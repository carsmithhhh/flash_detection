from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time

def mined_bce_loss(data, hit_times, photon_list, class_output, reg_output, epoch, device, include_reg=True, logger=None):
    # 8.27 - Sampling 500 hard negatives to resolve delayed waveform width
    
    data = data.squeeze(1)
    offset = 0
    N, L = data.shape
    target = torch.zeros((N, L), dtype=torch.float32).to(device)
    photon_target = torch.zeros((N, L), dtype=torch.float32).to(device)

    wf_width = 900 # 20 ns for non-delayed flashes
    rng = np.random.default_rng()
    sampled_indices = []

    # Build a mask tensor [N, L] for sampled indices (True = include in loss)
    sampled_indices = torch.zeros((N, L), dtype=torch.bool, device=device)

    for i, times in enumerate(hit_times):
        if (
            times is None 
            or (isinstance(times, (list, np.ndarray)) and len(times) == 0)
            or (isinstance(times, (list, np.ndarray)) and np.all(np.array(times) < 0))
        ):
            continue  # no flashes

        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        else:
            times = np.asarray(times).flatten()

        hit_indices = []
        for j, t in enumerate(times):
            if t < 0:
                pass
            else:
                t_idx = int(np.clip(t + offset, 0, L - 1))
                photon_num = photon_list[i][j]
                target[i, t_idx] = 1.0
                photon_target[i, t_idx] = photon_num
                sampled_indices[i, t_idx] = True
                hit_indices.append(t_idx)

        # Hard negative mining: 500 negative bins within wf_width of any hit time (but not the hit time itself)
        wf_neg_candidates = set()
        for t_idx in hit_indices:
            start = max(0, t_idx)
            end = min(L, t_idx + wf_width + 1)
            wf_neg_candidates.update(range(start, end))
            
        wf_neg_candidates.difference_update(hit_indices)
        wf_neg_candidates = list(wf_neg_candidates)
        if len(wf_neg_candidates) > 0:
            chosen_wf_neg = rng.choice(wf_neg_candidates, size=min(500, len(wf_neg_candidates)), replace=False)
            sampled_indices[i, chosen_wf_neg] = True

        # Random negative mining: 100 random bins outside wf_width of any hit and not a hit
        all_indices = set(range(L))
        forbidden = set(hit_indices).union(wf_neg_candidates)
        random_neg_candidates = list(all_indices - forbidden)
        if len(random_neg_candidates) > 0:
            chosen_rand_neg = rng.choice(random_neg_candidates, size=min(100, len(random_neg_candidates)), replace=False)
            sampled_indices[i, chosen_rand_neg] = True

    masked_class_output = class_output.squeeze(1)[sampled_indices]  # shape: [num_selected]
    masked_target = target[sampled_indices]# shape: [num_selected]

    # Mask the regression output to consider ONLY bins where softmax class > 0.5
    masked_reg_output = None
    masked_photon_target = None
    
    # Only calculate regression loss once classification has converged a bit to avoid seeing large # of 0's in early its
    if include_reg:
        mask = (torch.sigmoid(class_output) > 0.5).squeeze(1)
        masked_reg_output = reg_output.squeeze(1)[mask]
        masked_photon_target = photon_target[mask]

    # Calculate BCEWithLogits loss on masked values
    if masked_target.numel() == 0:
        class_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        # Dynamically calculate positive weighting per_batch
        n_pos = masked_target.sum().item()
        n_total = masked_target.numel()
        n_neg = n_total - n_pos
        pos_weight_val = n_neg / max(1, n_pos)
        
        pos_weight = torch.tensor([pos_weight_val], device=masked_target.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        class_loss = criterion(masked_class_output, masked_target)

        # regression_criterion = torch.nn.MSELoss() # torch.nn.L1Loss()
        regression_criterion = torch.nn.PoissonNLLLoss(log_input=True)
        reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if include_reg and masked_reg_output.numel() > 0:
            reg_loss = regression_criterion(masked_reg_output, masked_photon_target)
        scale_factor = 0.1

    loss = class_loss + (scale_factor * reg_loss)

    return loss, sampled_indices, masked_target, masked_class_output, class_output, target, masked_reg_output, masked_photon_target

def overall_class_acc(hit_times, class_output, device): # computed per batch
    '''
    Parameters:
        - hit_times: list of arrays, truth hit times for each sample
        - class_output: [B, 1, L] per-bin logits
    '''
    with torch.no_grad():
        B, _, L = class_output.shape
        class_output = class_output.squeeze(1)  # [B, L]
        mask = (torch.sigmoid(class_output) > 0.5)  # [B, L]

        batch_accs = []
        for i in range(B):
            # predicted hit indices for this sample
            pred_hits = set(torch.nonzero(mask[i], as_tuple=False).squeeze(1).tolist())

            # true hit indices for this sample
            true_hits = set(int(t) for t in hit_times[i] if t > 0)

            if len(true_hits) > 0:
                correct_hits = len(true_hits.intersection(pred_hits))
                acc = correct_hits / len(true_hits) # ACCURACY: DIVIDE BY # TRUE BINS
            else:
                acc = 0.0
            batch_accs.append(acc)

    return sum(batch_accs) / len(batch_accs)

def overall_class_purity(hit_times, class_output, device): # computed per batch
    with torch.no_grad():
        B, _, L = class_output.shape
        class_output = class_output.squeeze(1)  # [B, L]
        mask = (torch.sigmoid(class_output) > 0.5)  # [B, L]

        batch_accs = []
        for i in range(B):
            # predicted hit indices for this sample
            pred_hits = set(torch.nonzero(mask[i], as_tuple=False).squeeze(1).tolist())

            # true hit indices for this sample
            true_hits = set(int(t) for t in hit_times[i] if t > 0)

            if len(pred_hits) > 0:
                correct_hits = len(true_hits.intersection(pred_hits))
                acc = correct_hits / len(pred_hits) # PURITY -> DIVIDE BY # PREDICTED BINS
            else:
                acc = 0.0
            batch_accs.append(acc)
    
    return sum(batch_accs) / len(batch_accs)


def regression_rmse(hit_times, photon_bins, reg_output, class_output, device):
    """
    Compute regression RMSE over nonzero bins.

    Parameters:
        hit_times: list/array of true hit indices per sample (length B, each element iterable)
        photon_bins: [B, 1, L] per-bin binary indicators (1 if photons present, 0 otherwise)
        reg_output: [B, 1, L] per-bin regression values

    Returns:
        batch_rmse: average per-sample RMSE across true hit bins
    """
    with torch.no_grad():
        B, _, L = reg_output.shape
        reg_output = reg_output.squeeze(1).to(device)  # [B, L]
        photon_bins = photon_bins.squeeze(1).to(device) # [B, L]

        batch_rmses = []
        for i in range(B):
            # True hit indices for this sample
            true_hit_idx = [int(t) for t in hit_times[i] if t >= 0]

            if len(true_hit_idx) > 0:
                preds = torch.exp(reg_output[i, true_hit_idx])
                targets = photon_bins[i, true_hit_idx].float()
                mse = torch.mean((preds - targets) ** 2)
                rmse = torch.sqrt(mse)
                batch_rmses.append(rmse.item())
            else:
                # If no hits, skip or append NaN/0 depending on convention
                batch_rmses.append(0.0)

    return sum(batch_rmses) / len(batch_rmses) if batch_rmses else 0.0


# def merge_bins(class_output, skip_tol=1):
#     """
#     Vectorized version of merge_bins.
#     """
#     B, C, L = class_output.shape
#     device = class_output.device

#     active = (torch.sigmoid(class_output) > 0.5).int()  # [B, 1, L]

#     if skip_tol == 0:
#         return active

#     # Morphological closing: dilate → erode
#     # Expand gaps of length <= skip_tol
#     # Use a conv1d with a kernel of ones
#     kernel = torch.ones(1, 1, skip_tol + 1, device=device)
    
#     # Dilation: any 1 spreads to neighbors within skip_tol
#     dilated = F.conv1d(active.float(), kernel, padding=skip_tol//2) > 0

#     # Closing (approximation): fill gaps smaller than skip_tol
#     merged_mask = dilated.int()

#     return merged_mask
def merge_bins(class_output, skip_tol=1):
    """
    Morphological closing on 1D binary masks.
    Fills gaps smaller than skip_tol without thickening regions.
    """
    B, C, L = class_output.shape
    device = class_output.device

    active = (torch.sigmoid(class_output) > 0.5).int()  # [B, 1, L]

    if skip_tol == 0:
        return active

    # Structuring element length
    k = skip_tol * 2 + 1  
    kernel = torch.ones(1, 1, k, device=device)

    # ---- Dilation ----
    dilated = F.conv1d(active.float(), kernel, padding=skip_tol) > 0

    # ---- Erosion ----
    eroded = F.conv1d(dilated.float(), kernel, padding=skip_tol) == k

    return eroded.int()

def mask_to_intervals(mask_row):
    """
    Vectorized: convert 1D mask to list of (start,end).
    """
    mask = mask_row.cpu().numpy().astype(int)
    diff = mask[1:] - mask[:-1]

    starts = list(np.where(diff == 1)[0] + 1)
    ends   = list(np.where(diff == -1)[0])

    # If mask starts with 1, prepend start=0
    if mask[0] == 1:
        starts = [0] + starts
    # If mask ends with 1, append end=L-1
    if mask[-1] == 1:
        ends = ends + [len(mask)-1]

    return list(zip(starts, ends))

def merged_class_acc(merged_mask, hit_times, device):
    '''
    Parameters:
        - merged_mask: mask of shape [B, 1, L] with 1's for merged regions
        - hit_times: list of arrays, truth hit times for each sample
    '''
    with torch.no_grad():
        B, _, L = merged_mask.shape

        batch_accs = []
        for i in range(B):
            # predicted hit indices for this sample
            pred_hits = set(torch.nonzero(merged_mask[i], as_tuple=False).flatten().tolist())
            
             # true hit indices for this sample
            true_hits = set(int(t) for t in hit_times[i] if t > 0)

            if len(true_hits) > 0:
                correct_hits = len(true_hits.intersection(pred_hits))
                acc = correct_hits / len(true_hits) # ACCURACY: DIVIDE BY # TRUE BINS
            else:
                acc = 0.0
            batch_accs.append(acc)

    return sum(batch_accs) / len(batch_accs)

def merged_class_purity(merged_mask, hit_times, device, no_sum=False): # computed per batch
    with torch.no_grad():
        B, _, L = merged_mask.shape

        batch_accs = []
        for i in range(B):
            # within predicted window bins
            pred_hits = set(torch.nonzero(merged_mask[i], as_tuple=False).flatten().tolist())
            mask_row = merged_mask[i, 0]  # [L]
            intervals = mask_to_intervals(mask_row) #[[s1, e1], [s2, e2], ...]
            
            # true hit indices for this sample
            true_hits = set(int(t) for t in hit_times[i] if t > 0)

            if len(pred_hits) > 0:
                correct_hits = len(true_hits.intersection(pred_hits))
                acc = correct_hits / len(intervals) # PURITY -> DIVIDE BY # merged windows
            else:
                acc = 0.0
            batch_accs.append(acc)

    if no_sum:
        return batch_accs
    else:
        return sum(batch_accs) / len(batch_accs)

def sum_photons_in_intervals(photon_counts, merged_mask):
    """
    Sum predicted photon counts within each continuous interval.

    Parameters
    ----------
    photon_counts : torch.Tensor, shape [B, 1, L]
        Exponentiated regression outputs (log photon counts), or target
    merged_mask : torch.Tensor, shape [B, 1, L]
        Binary mask of intervals (after merging).

    Returns
    -------
    all_interval_sums : list of list of floats
        For each batch element, a list of interval photon sums.
    """
    B, C, L = photon_counts.shape

    all_interval_sums = []
    for b in range(B):
        mask = merged_mask[b, 0].cpu().numpy()
        counts = photon_counts[b, 0].detach().cpu().numpy()

        interval_sums = []
        in_interval = False
        start = 0

        for i in range(L):
            if mask[i] == 1 and not in_interval:
                # start of a new interval
                in_interval = True
                start = i
            elif mask[i] == 0 and in_interval:
                # end of interval
                in_interval = False
                interval_sums.append(counts[start:i].sum())
        if in_interval:  # handle if it ends at the last bin
            interval_sums.append(counts[start:].sum())

        all_interval_sums.append(interval_sums)

    return all_interval_sums

def sum_photons_in_intervals_vec(photon_counts, merged_mask):
    """
    Vectorized version (GPU-friendly) of sum_photons_in_intervals.
    Returns list of lists (still pythonic output), but avoids Python loops over L.
    """
    B, C, L = photon_counts.shape
    device = photon_counts.device
    all_interval_sums = []

    for b in range(B):
        mask = merged_mask[b, 0]  # [L]
        counts = photon_counts[b, 0]  # [L]

        # assign a unique interval id to each contiguous run of 1s
        interval_ids = torch.zeros_like(mask, dtype=torch.long, device=device)
        interval_counter = 0
        prev = 0
        for i in range(L):
            if mask[i] == 1:
                if i == 0 or mask[i-1] == 0:
                    interval_counter += 1
                interval_ids[i] = interval_counter

        # sum counts per interval
        if interval_counter == 0:
            all_interval_sums.append([])
        else:
            sums = torch.zeros(interval_counter, device=device, dtype=counts.dtype).scatter_add_(0, interval_ids[mask>0]-1, counts[mask>0])
            all_interval_sums.append(sums.cpu().tolist())

    return all_interval_sums

def interval_rmse(pred_sums_list, true_sums_list):
    errors = []
    for pred, true in zip(pred_sums_list, true_sums_list):
        if len(pred) > 0:
            errors.append(torch.tensor(pred, device='cpu') - torch.tensor(true, device='cpu'))
    if not errors:
        return None
    errors = torch.cat(errors)
    return float(torch.sqrt((errors**2).mean()))

def merged_twoflash_acc(merged_mask, hit_times, device, no_sum=False):
    '''
    Parameters:
        - merged_mask: mask of shape [B, 1, L] with 1's for merged regions
        - hit_times: list of arrays or lists, truth hit times for each sample [time1, time2]
    Returns:
        - avg_flash1_acc, avg_flash2_acc
    '''
    with torch.no_grad():
        B, _, L = merged_mask.shape

        flash1_accs, flash2_accs = [], []

        for i in range(B):
            pred_hits = set(torch.nonzero(merged_mask[i], as_tuple=False).flatten().tolist())

            # extract truth times, enforce exactly two flashes
            true_times = [int(t) for t in hit_times[i] if t > 0]
            if len(true_times) < 2:
                # pad with dummy values if needed
                true_times = true_times + [None] * (2 - len(true_times))

            # check flash 1
            if true_times[0] is not None:
                flash1_accs.append(1.0 if true_times[0] in pred_hits else 0.0)
            else:
                flash1_accs.append(0.0)

            # check flash 2
            if true_times[1] is not None:
                flash2_accs.append(1.0 if true_times[1] in pred_hits else 0.0)
            else:
                flash2_accs.append(0.0)

        # return averages across batch
        if no_sum:
            return flash1_accs, flash2_accs
        else:
            return sum(flash1_accs)/B, sum(flash2_accs)/B
        
