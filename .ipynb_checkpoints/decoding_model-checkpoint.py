import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from scipy.optimize import linear_sum_assignment
from transformer import PositionalEncoding

class MultiLevelTokenizer(nn.Module):
    def __init__(self, in_channels=1, d_model=256, kernel_sizes=[20, 50, 100, 400], window_len=8000, token_size=16):
        super().__init__()
        num_tokens = window_len // token_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=d_model, 
                      kernel_size=k, 
                      stride=1, 
                      padding=k//2)
            for k in kernel_sizes
        ])
        
        # 2-Layer Conv MLP - Temporal Downsampling
        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=token_size, stride=token_size, padding=0, groups=d_model, bias=False),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1),
        )

        # Project to decoder feature dimension
        self.proj = nn.Conv1d(
            in_channels=len(kernel_sizes) * d_model,
            out_channels=d_model,
            kernel_size=1
        )

    def forward(self, x):
        conv_outs = []
        for conv in self.convs:
            feat = F.relu(conv(x))    # (B, d_model, 8000)
            feat = self.downsample(feat)   # [B, 500, d_model]
            conv_outs.append(feat)

        out = torch.cat(conv_outs, dim=1)  # (B, d_model * n_kernels, 500)
        out = self.proj(out)

        return out

class DecodingModel(nn.Module):
    def __init__(self, d_model=256, num_layers=4, num_heads=8, num_queries=8, window_len=8000):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.window_len = window_len

        self.tokenizer = MultiLevelTokenizer()
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.detect_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2), # [start_time, duration]
            nn.Sigmoid(),  # normalize to [0,1]
        )

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.objectness_head = nn.Sequential( # signal confidence
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, waveform_batch):
        B, in_channels, window_len = waveform_batch.shape
        
        x = self.tokenizer(waveform_batch) # [B, d_model, 500]
        x = x.transpose(1, 2) # [B, 500, d_model]
        x = self.positional_encoding(x)

        memory = x.transpose(0, 1)  # [500, B, d_model], encoder outputs
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, d_model]
        decoded = self.transformer_decoder(query, memory)  # [num_queries, B, d_model]
        decoded = decoded.transpose(0, 1)

        signal_output = self.detect_head(decoded)
        reg_output = self.reg_head(decoded)
        confidence_output = self.objectness_head(decoded)

        return signal_output, reg_output, confidence_output


def compute_cost_matrix(pred_signal, pred_photons, target, device='cuda', ph_weight=0.05):
    """
    pred_signal: [num_queries, 2] -> [start, duration] (normalized)
    pred_photons: [num_queries, 1]
    target: dict with keys 'start_time' (N,), 'photons' (N,)
    Returns: cost_matrix [num_queries, num_gt] (numpy)
    """
    pred_start = pred_signal[:, 0].unsqueeze(1)          # [Q,1]
    pred_ph = pred_photons[:, 0].unsqueeze(1)           # [Q,1]

    gt_start = target['start_time'].unsqueeze(0).to(device)  # [1, N]
    gt_ph = target['photons'].unsqueeze(0).to(device)        # [1, N]

    # simple L1 on normalized start times; add small photon term as tie-breaker
    cost_start = torch.abs(pred_start - gt_start)           # [Q, N]
    cost_ph = torch.abs(pred_ph - gt_ph)                    # [Q, N]

    cost = cost_start + ph_weight * cost_ph
    return cost.cpu().detach().numpy()

    
def hungarian_match(pred_signal, pred_photons, target, device='cuda', ph_weight=0.05):
    """
    Returns integer index arrays (row_ind, col_ind)
    - row_ind: indices of predictions (queries)
    - col_ind: indices of matched gt (same length)
    """
    if target['start_time'].numel() == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)

    cost_matrix = compute_cost_matrix(pred_signal, pred_photons, target, device=device, ph_weight=ph_weight)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)

    import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ------------ matching cost (uses only start time; optional photon bonus) -------------
def compute_cost_matrix(pred_signal, pred_photons, target, device='cuda', ph_weight=0.05):
    """
    pred_signal: [num_queries, 2] -> [start, duration] (normalized)
    pred_photons: [num_queries, 1]
    target: dict with keys 'start_time' (N,), 'photons' (N,)
    Returns: cost_matrix [num_queries, num_gt] (numpy)
    """
    pred_start = pred_signal[:, 0].unsqueeze(1)          # [Q,1]
    pred_ph = pred_photons[:, 0].unsqueeze(1)           # [Q,1]

    gt_start = target['start_time'].unsqueeze(0).to(device)  # [1, N]
    gt_ph = target['photons'].unsqueeze(0).to(device)        # [1, N]

    # simple L1 on normalized start times; add small photon term as tie-breaker
    cost_start = torch.abs(pred_start - gt_start)           # [Q, N]
    cost_ph = torch.abs(pred_ph - gt_ph)                    # [Q, N]

    cost = cost_start + ph_weight * cost_ph
    return cost.cpu().detach().numpy()


def hungarian_match(pred_signal, pred_photons, target, device='cuda', ph_weight=0.05):
    """
    Returns integer index arrays (row_ind, col_ind)
    - row_ind: indices of predictions (queries)
    - col_ind: indices of matched gt (same length)
    """
    if target['start_time'].numel() == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)

    cost_matrix = compute_cost_matrix(pred_signal, pred_photons, target, device=device, ph_weight=ph_weight)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)


# ------------ set-based loss -------------
def set_based_loss(signal_output, reg_output, confidence_output, targets,
                   lambda_start=5.0, lambda_photon=1e-4, lambda_conf=0.5,
                   device='cuda', pos_weight=None, logger=None):
    """
    signal_output: [B, Q, 2]  (start,duration normalized to [0,1])
    reg_output:    [B, Q, 1]  photon counts (raw scalar)
    confidence_output: [B, Q, 1] in [0,1] (sigmoid already applied in head)
    targets: list of length B, each dict with 'start_time' (k,), 'photons' (k,)
    Returns: total_loss (scalar tensor), dict of components for logging
    """
    B, Q, _ = signal_output.shape
    device = torch.device(device if isinstance(device, str) else device)

    total_loss = torch.tensor(0., device=device)
    total_start = torch.tensor(0., device=device)
    total_ph = torch.tensor(0., device=device)
    total_conf = torch.tensor(0., device=device)

    # pos_weight for BCE: helpful when many negative queries
    # PyTorch BCEWithLogitsLoss accepts pos_weight, but we already have sigmoid -> use manual weighting
    if pos_weight is None:
        # estimate pos_weight = (#negatives / #positives) ~ (Q - avg_gt)/avg_gt
        avg_gt = max(1.0, sum([t['start_time'].numel() for t in targets]) / max(1, B))
        est_pos_weight = (Q - avg_gt) / max(1.0, avg_gt)
        pos_weight = torch.tensor(est_pos_weight, device=device)

    for b in range(B):
        pred_signal = signal_output[b]      # [Q,2]
        pred_photons = reg_output[b]        # [Q,1]
        pred_conf = confidence_output[b]    # [Q,1]
        target = targets[b]

        # if no GT pulses: force all objectness to 0 (negatives)
        if target['start_time'].numel() == 0:
            # BCE loss to encourage zeros
            target_unmatched_conf = torch.zeros_like(pred_conf, device=device)
            bce = F.binary_cross_entropy(pred_conf, target_unmatched_conf, reduction='mean')
            total_conf += bce
            total_loss += lambda_conf * bce
            continue

        # --- matching (Hungarian) ---
        row_ind, col_ind = hungarian_match(pred_signal, pred_photons, target, device=device)
        if row_ind.numel() == 0:
            # should not usually happen if gt exists
            continue

        # row_ind = row_ind.to(device)
        # col_ind = col_ind.to(device)

        # matched predictions/gt
        matched_pred_start = pred_signal[row_ind, 0]         # [M]
        matched_pred_start = matched_pred_start.clamp(0.0, 1.0)

        matched_pred_ph = pred_photons[row_ind, 0]           # [M]
        matched_pred_conf = pred_conf[row_ind, 0]            # [M]

        matched_gt_start = target['start_time'][col_ind].to(device)   # [M]
        matched_gt_ph = target['photons'][col_ind].to(device)         # [M]

        loss_start = F.smooth_l1_loss(
            matched_pred_start, matched_gt_start, reduction='mean', beta=1e-2
        ) # smooth near zero
        
        loss_ph = F.l1_loss(matched_pred_ph, matched_gt_ph, reduction='mean')
        
        # matched queries
        target_pos = torch.ones_like(matched_pred_conf, device=device)
        loss_conf_pos = F.binary_cross_entropy(matched_pred_conf, target_pos, reduction='mean')
        
        # unmatched queries
        all_idx = torch.arange(Q, device=device)
        matched_idx = torch.tensor(row_ind, device=device)
        unmatched_mask = torch.ones(Q, dtype=torch.bool, device=device)
        unmatched_mask[matched_idx] = False
        
        if unmatched_mask.any():
            unmatched_pred_conf = pred_conf[unmatched_mask, 0]
            target_neg = torch.zeros_like(unmatched_pred_conf, device=device)
            loss_conf_neg = F.binary_cross_entropy(unmatched_pred_conf, target_neg, reduction='mean')
            # combine positive/negative losses with weighting
            confidence_loss = 0.7 * loss_conf_pos + 0.3 * loss_conf_neg
        else:
            confidence_loss = loss_conf_pos

        # accumulate
        total_start += loss_start
        total_ph += loss_ph
        total_conf += confidence_loss

        total_loss = total_loss + lambda_start * loss_start + lambda_photon * loss_ph + lambda_conf * confidence_loss

    # average over batch
    total_loss = total_loss / B
    stats = {
        'loss_total': total_loss.detach().cpu().item(),
        'loss_start_mean': (total_start / B).detach().cpu().item(),
        'loss_photon_mean': (total_ph / B).detach().cpu().item(),
        'loss_conf_mean': (total_conf / B).detach().cpu().item(),
    }
    if logger is not None:
        logger.log_dict(stats)

    return total_loss, stats['loss_start_mean'], stats['loss_photon_mean'], stats['loss_conf_mean']

# def set_based_loss(signal_output, reg_output, confidence_output, targets, lambda_signal=1.0, lambda_photon=0.00005, lambda_conf = 0.005, device='cuda', logger=None):
#     """
#     signal_output: [B, num_queries, 2]
#     reg_output: [B, num_queries, 1]
#     targets: list of length B, each dict has 'start_time', 'duration', 'photons'
#     """
#     B, num_queries, _ = signal_output.shape
#     total_loss = 0.0

#     for b in range(B):
#         pred_signal = signal_output[b]  # [num_queries, 2]
#         pred_photons = reg_output[b]    # [num_queries, 1]
#         pred_conf = confidence_output[b]
#         target = targets[b]

#         # This will never happen in the training set
#         if target['start_time'].numel() == 0:
#             continue

#         row_ind, col_ind = hungarian_match(pred_signal, pred_photons, target)
#         row_ind, col_ind = row_ind.to(device), col_ind.to(device)

#         # matched predictions
#         matched_pred_signal = pred_signal[row_ind, 0] # only start time
#         matched_pred_photon = pred_photons[row_ind]
#         matched_pred_conf = pred_conf[row_ind]

#         # matched ground truth
#         matched_gt_signal = target['start_time'][col_ind].to(device)
#         matched_gt_photon = target['photons'][col_ind].unsqueeze(1)

#         # confidence loss
#         target_conf = torch.ones_like(matched_pred_conf)
#         confidence_loss = F.binary_cross_entropy(matched_pred_conf, target_conf, reduction='mean')
#         # also include in confidence loss unmatched queries
#         all_indices = set(range(num_queries))
#         unmatched_indices = list(all_indices - set(row_ind.tolist()))
#         if unmatched_indices:
#             unmatched_pred_conf = pred_conf[unmatched_indices]
#             target_unmatched_conf = torch.zeros_like(unmatched_pred_conf)
#             confidence_loss += F.binary_cross_entropy(unmatched_pred_conf, target_unmatched_conf, reduction='mean')

#         # regression losses
#         loss_signal = F.l1_loss(matched_pred_signal, matched_gt_signal.to(device), reduction='mean')
#         loss_photon = F.l1_loss(matched_pred_photon, matched_gt_photon.to(device), reduction='mean')

#         total_loss += lambda_signal * loss_signal + lambda_photon * loss_photon + lambda_conf * confidence_loss

#     return total_loss / B, loss_signal, loss_photon, confidence_loss