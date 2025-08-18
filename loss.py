from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def mined_bce_loss(data, hit_times, photon_list, class_output, reg_output, epoch, device):
    # Instead of using all negative samples, we wil pick 150
    # CURRENTLY, BUMPING UP STATS TO 167 BINS PER SAMPLE

    data = data.squeeze(1)
    offset = 0
    N, L = data.shape
    target = torch.zeros((N, L), dtype=torch.float32).to(device)
    photon_target = torch.zeros((N, L), dtype=torch.float32).to(device)

    wf_width = 20 # ns, eyeball for now
    rng = np.random.default_rng()
    sampled_indices = []

    # Build a mask tensor [N, L] for sampled indices (True = include in loss)
    sampled_indices = torch.zeros((N, L), dtype=torch.bool, device=device)

    for i, times in enumerate(hit_times):
        # Always include the true hit times
        if times is None or (isinstance(times, (list, np.ndarray)) and len(times) == 0):
            continue  # no flashes

        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        else:
            times = np.asarray(times).flatten()

        hit_indices = []
        for j, t in enumerate(times):
            t_idx = int(np.clip(t + offset, 0, L - 1))
            photon_num = photon_list[i][j]
            target[i, t_idx] = 1.0
            photon_target[i, t_idx] = photon_num
            sampled_indices[i, t_idx] = True
            hit_indices.append(t_idx)

        # Hard negative mining: 50 negative bins within wf_width of any hit time (but not the hit time itself)
        wf_neg_candidates = set()
        for t_idx in hit_indices:
            start = max(0, t_idx)
            end = min(L, t_idx + wf_width + 1)
            wf_neg_candidates.update(range(start, end))
            
        wf_neg_candidates.difference_update(hit_indices)
        wf_neg_candidates = list(wf_neg_candidates)
        if len(wf_neg_candidates) > 0:
            chosen_wf_neg = rng.choice(wf_neg_candidates, size=min(50, len(wf_neg_candidates)), replace=False)
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
    if epoch > 2: # Only calculate regression loss once classification has converged a bit to avoid seeing large # of 0's in early its
        mask = (torch.sigmoid(class_output) > 0.5).squeeze(1)   # shape [batch]
        masked_reg_output = reg_output.squeeze(1)[mask]
        masked_photon_target = photon_target[mask]
    
    # Calculate BCEWithLogits loss on masked values
    if masked_target.numel() == 0:
        # No positive or negative samples selected, return zero loss
        class_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        pos_weight_val = 100.0 
        pos_weight = torch.tensor([pos_weight_val], device=masked_target.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        class_loss = criterion(masked_class_output, masked_target)

        regression_criterion = torch.nn.MSELoss()
        reg_loss = 0.0
        if epoch > 2:
            reg_loss = regression_criterion(masked_reg_output, masked_photon_target)

        scale_factor = 1.0

    loss = class_loss + (scale_factor * reg_loss)

    return loss, sampled_indices, masked_target, masked_class_output, class_output, target, masked_reg_output, masked_photon_target

def bce_loss(data, hit_times, class_output, device): # works
    data = data.squeeze(1)
    N, L = data.shape
    offset = 0
    target = torch.zeros((N, L), dtype=torch.float32).to(device)

    for i, times in enumerate(hit_times):
        if times is None or (isinstance(times, (list, np.ndarray)) and len(times) == 0):
            continue  # no flashes

        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        else:
            times = np.asarray(times).flatten()

        for t in times:
            t_idx = int(np.clip(t + offset, 0, L - 1))
            target[i, t_idx] = 1.0

    # calculate weights
    pos_weight_val = L
    pos_weight = torch.tensor([pos_weight_val], device=target.device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(class_output.squeeze(1), target)

    return loss, target

def val_bce(data, hit_times, class_output, device):
    data = data.squeeze(1)
    N, L = data.shape
    offset = 0

    # Build target
    target = torch.zeros((N, L), dtype=torch.float32).to(device)
    hit_indices_list = []
    for i, times in enumerate(hit_times):
        indices = []
        if times is None or (isinstance(times, (list, np.ndarray)) and len(times) == 0):
            hit_indices_list.append(indices)
            continue

        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        else:
            times = np.asarray(times).flatten()

        for t in times:
            t_idx = int(np.clip(t + offset, 0, L - 1))
            target[i, t_idx] = 1.0
            indices.append(t_idx)

        hit_indices_list.append(indices)

    # Compute accuracy
    class_output_flat = class_output.view(N, -1)
    accuracy_per_sample = []

    for i in range(N):
        true_indices = set(hit_indices_list[i])
        n_hits = len(true_indices) # works
        # print(n_hits, len(hit_indices_list[i]))

        if n_hits == 0:
            continue  # skip samples with no true hits

        # Get top-n prediction indices
        topk_indices = torch.topk(class_output_flat[i], k=n_hits).indices.detach().cpu().numpy()
        topk_set = set(topk_indices)

        # Count how many predictions are correct
        correct_hits = len(true_indices.intersection(topk_set))
        accuracy = correct_hits / n_hits
        accuracy_per_sample.append(accuracy)

    # Final accuracy: mean over samples with hits
    final_accuracy = sum(accuracy_per_sample) / len(accuracy_per_sample) if accuracy_per_sample else 0.0
    return final_accuracy #, data[i], class_output_flat[i], true_indices, topk_set 


def regression_loss_fast(data, hit_times, class_output, device, strategy='closest'):
    """
    Ultra-fast vectorized regression loss for time differences.
    Processes entire batch at once using vectorized operations.
    
    Args:
        data: Input data tensor of shape [B, 1, n_bins]
        hit_times: List of lists where each element contains hit times for that sample
        class_output: Model class_output tensor of shape [B, n_bins]
        device: torch device
        strategy: How to handle multiple hit times per sample
            - 'closest': For each bin, use the closest hit time
    """
    time_res = 1 # ns
    B, _, n_bins = data.shape
    bin_centers = torch.arange(time_res / 2, n_bins * time_res, time_res, device=device)
    offset = 0

    data = data.squeeze(1) # [B, 16000] ok
    bin_centers = bin_centers.expand(B, -1)  # shape: [B, 16000] ok
    
    # Pre-process all hit times at once
    max_hit_times = max(len(times) if times is not None else 0 for times in hit_times)
    if max_hit_times == 0:
        # No hit times in batch
        target_diffs = torch.zeros((B, n_bins), device=device)
    else:
        # Create padded tensor of hit times: [B, max_hit_times]
        hit_times_padded = torch.full((B, max_hit_times), float('inf'), device=device)
        hit_times_mask = torch.zeros((B, max_hit_times), dtype=torch.bool, device=device)
        
        for i, times in enumerate(hit_times):
            if times is None or (isinstance(times, (list, np.ndarray)) and len(times) == 0):
                continue
                
            # Convert to list if it's a single time
            if np.isscalar(times):
                times = [times]
            else:
                times = np.asarray(times.cpu()).flatten()
            
            # Apply offset
            times = torch.tensor(times) + offset
            
            # Pad to max_hit_times
            num_times = len(times)
            hit_times_padded[i, :num_times] = torch.tensor(times, device=device, dtype=torch.float32)
            hit_times_mask[i, :num_times] = True
        
        # Expand for vectorized computation: [B, max_hit_times, n_bins]
        hit_times_expanded = hit_times_padded.unsqueeze(2).expand(-1, -1, n_bins)
        bin_centers_expanded = bin_centers.unsqueeze(1).expand(-1, max_hit_times, -1)
        
        # Calculate all differences: [B, max_hit_times, n_bins]
        all_diffs = bin_centers_expanded - hit_times_expanded
        
        # Apply mask to ignore invalid hit times
        all_diffs = torch.where(hit_times_mask.unsqueeze(2), all_diffs, torch.tensor(float('inf'), device=device))
            
        # strategy == 'closest'
        # Find minimum absolute difference
        abs_diffs = torch.abs(all_diffs)
        # Replace inf with large values for argmin
        abs_diffs = torch.where(torch.isfinite(abs_diffs), abs_diffs, torch.tensor(1e10, device=device))
        min_indices = torch.argmin(abs_diffs, dim=1)  # [B, n_bins]
        
        # Gather the corresponding differences
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_bins)
        target_diffs = all_diffs[batch_indices, min_indices, torch.arange(n_bins, device=device)]
        
        # Handle cases with no hit times
        no_hit_mask = ~hit_times_mask.any(dim=1)
        target_diffs = torch.where(no_hit_mask.unsqueeze(1), torch.zeros_like(target_diffs), target_diffs)
    
    # Calculate MSE loss
    loss = F.mse_loss(class_output.squeeze(1), torch.log1p(abs_diffs))

    return loss, abs_diffs

def val_regression(data, hit_times, class_output, device, strategy='closest'):
    class_output = class_output.squeeze(1) # Shape [B, 16000]
    offset = 0
    
    B, L = class_output.shape

    # Get number of ground truth hits per sample
    k_per_sample = torch.tensor([len(ht) if ht is not None else 0 for ht in hit_times], device=class_output.device)

    pred_indices, _ = k_local_minima(class_output, k_per_sample)  # [B, max_k]

    total_hits = 0
    correct_matches = 0

    # for each sample, correct local minima / total hits
    for i in range(B):
        gt_times = hit_times[i]
        if gt_times is None or len(gt_times) == 0:
            continue

        gt_indices = torch.tensor([max(0, min(int(t + offset), L - 1)) for t in gt_times], device=device)

        pred = pred_indices[i]
        pred = pred[pred != -1]  # Remove padding

        if strategy == 'closest':
            # Match each GT hit to the closest predicted one, count unique matches
            matched = set()
            for gt in gt_indices:
                if len(pred) == 0:
                    continue
                closest_idx = torch.argmin(torch.abs(pred - gt))
                matched.add(pred[closest_idx].item())
            correct_matches += len(matched)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        total_hits += len(gt_indices)
        acc = correct_matches / total_hits if total_hits > 0 else 0.0

    return acc, class_output[i], gt_indices, pred
    

def k_local_minima(batch_tensor, k_per_sample):
    """
    batch_tensor: Tensor of shape [B, L]
    k_per_sample: LongTensor of shape [B] (e.g. tensor([2, 3, 1]))

    Returns:
        topk_indices: [B, max_k] (with unused entries set to -1)
        topk_values: [B, max_k] (with unused entries set to +inf)
    """
    B, L = batch_tensor.shape
    max_k = torch.max(k_per_sample).item()

    # Find local minima mask
    x = batch_tensor
    left  = x[:, 1:-1] < x[:, :-2]
    right = x[:, 1:-1] < x[:, 2:]
    local_minima_mask = left & right  # [B, L-2]

    # Candidate indices and values
    candidate_indices = torch.arange(1, L - 1, device=x.device).unsqueeze(0).expand(B, -1)  # [B, L-2]
    minima_indices = candidate_indices.clone()
    minima_indices[~local_minima_mask] = L  # dummy invalid index
    minima_values = x[:, 1:-1].clone()
    minima_values[~local_minima_mask] = float('inf')

    # topk (deepest = smallest values)
    topk_values, topk_pos = torch.topk(-minima_values, k=max_k, dim=1, largest=True, sorted=True)
    topk_indices = torch.gather(minima_indices, 1, topk_pos)
    topk_values = -topk_values

    # Mask out values beyond each sample's k
    range_k = torch.arange(max_k, device=x.device).unsqueeze(0).expand(B, -1)  # [B, max_k]
    k_per_sample_expanded = k_per_sample.unsqueeze(1)  # [B, 1]
    valid_mask = range_k < k_per_sample_expanded  # [B, max_k]

    topk_indices[~valid_mask] = -1
    topk_values[~valid_mask] = float('inf')

    return topk_indices, topk_values
