"""Evaluation metrics and prediction post-processing.

Two groups of helpers:
  * Per-bin metrics computed directly from the classification/regression maps
    (accuracy, purity, regression RMSE).
  * "Merged" metrics: predicted positive bins are first closed into contiguous
    intervals (:func:`merge_bins`), then photon counts are summed per interval and
    compared against truth. This matches how flashes are scored downstream.

Unless noted, ``class_output``/``reg_output`` are ``[B, 1, L]`` and ``hit_times`` is a
length-``B`` iterable of true hit-bin indices per sample.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _compute_grad_norm(parameters, norm_type=2.0):
    """Total gradient norm over ``parameters`` (for logging training stability)."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    if norm_type == float("inf"):
        return max(g.abs().max() for g in grads)
    return torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)


# --------------------------------------------------------------------------- #
# Per-bin classification / regression metrics
# --------------------------------------------------------------------------- #
def overall_class_acc(hit_times, class_output, device):
    """Mean per-sample recall: fraction of true hit bins predicted positive (>0.5)."""
    with torch.no_grad():
        B = class_output.shape[0]
        mask = torch.sigmoid(class_output.squeeze(1)) > 0.5  # [B, L]
        batch_accs = []
        for i in range(B):
            pred_hits = set(torch.nonzero(mask[i], as_tuple=False).squeeze(1).tolist())
            true_hits = {int(t) for t in hit_times[i] if t > 0}
            if len(true_hits) > 0:
                batch_accs.append(len(true_hits & pred_hits) / len(true_hits))
            else:
                batch_accs.append(0.0)
    return sum(batch_accs) / len(batch_accs)


def overall_class_purity(hit_times, class_output, device):
    """Mean per-sample precision: fraction of predicted positive bins that are true hits."""
    with torch.no_grad():
        B = class_output.shape[0]
        mask = torch.sigmoid(class_output.squeeze(1)) > 0.5  # [B, L]
        batch_pure = []
        for i in range(B):
            pred_hits = set(torch.nonzero(mask[i], as_tuple=False).squeeze(1).tolist())
            true_hits = {int(t) for t in hit_times[i] if t > 0}
            if len(pred_hits) > 0:
                batch_pure.append(len(true_hits & pred_hits) / len(pred_hits))
            else:
                batch_pure.append(0.0)
    return sum(batch_pure) / len(batch_pure)


def regression_rmse(hit_times, photon_bins, reg_output, class_output, device, mse=False):
    """Mean per-sample RMSE of predicted vs true photon counts at true hit bins.

    ``mse=True`` decodes the regression output with ``expm1`` (log1p target convention);
    otherwise ``exp`` (Poisson log-rate convention).
    """
    predict_fn = torch.expm1 if mse else torch.exp
    with torch.no_grad():
        B = reg_output.shape[0]
        reg_output = reg_output.squeeze(1).to(device)    # [B, L]
        photon_bins = photon_bins.squeeze(1).to(device)  # [B, L]
        batch_rmses = []
        for i in range(B):
            true_hit_idx = [int(t) for t in hit_times[i] if t >= 0]
            if len(true_hit_idx) > 0:
                preds = predict_fn(reg_output[i, true_hit_idx])
                targets = photon_bins[i, true_hit_idx].float()
                batch_rmses.append(torch.sqrt(torch.mean((preds - targets) ** 2)).item())
            else:
                batch_rmses.append(0.0)
    return sum(batch_rmses) / len(batch_rmses) if batch_rmses else 0.0


# --------------------------------------------------------------------------- #
# Interval merging and merged metrics
# --------------------------------------------------------------------------- #
def merge_bins(class_output, skip_tol=1):
    """Morphological closing of the per-bin positive mask (dilate then erode).

    Fills gaps smaller than ``skip_tol`` so a single flash's spread-out positive bins
    become one contiguous interval, without thickening isolated regions. Returns an
    int mask ``[B, 1, L]``.
    """
    device = class_output.device
    active = (torch.sigmoid(class_output) > 0.5).int()  # [B, 1, L]
    if skip_tol == 0:
        return active
    k = skip_tol * 2 + 1
    kernel = torch.ones(1, 1, k, device=device)
    dilated = F.conv1d(active.float(), kernel, padding=skip_tol) > 0
    eroded = F.conv1d(dilated.float(), kernel, padding=skip_tol) == k
    return eroded.int()


def mask_to_intervals(mask_row):
    """Convert a 1D binary mask into a list of ``(start, end)`` inclusive intervals."""
    mask = mask_row.cpu().numpy().astype(int)
    diff = mask[1:] - mask[:-1]
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])
    if mask[0] == 1:
        starts = [0] + starts
    if mask[-1] == 1:
        ends = ends + [len(mask) - 1]
    return list(zip(starts, ends))


def merged_class_acc(merged_mask, hit_times, device):
    """Mean recall after merging: true hit bins covered by a predicted interval."""
    with torch.no_grad():
        B = merged_mask.shape[0]
        batch_accs = []
        for i in range(B):
            pred_hits = set(torch.nonzero(merged_mask[i], as_tuple=False).flatten().tolist())
            true_hits = {int(t) for t in hit_times[i] if t > 0}
            if len(true_hits) > 0:
                batch_accs.append(len(true_hits & pred_hits) / len(true_hits))
            else:
                batch_accs.append(0.0)
    return sum(batch_accs) / len(batch_accs)


def merged_class_purity(merged_mask, hit_times, device, no_sum=False):
    """Mean purity after merging: true hits found per predicted interval.

    With ``no_sum=True`` returns the per-sample list instead of the batch mean.
    """
    with torch.no_grad():
        B = merged_mask.shape[0]
        batch_pures = []
        for i in range(B):
            pred_hits = set(torch.nonzero(merged_mask[i], as_tuple=False).flatten().tolist())
            intervals = mask_to_intervals(merged_mask[i, 0])
            true_hits = {int(t) for t in hit_times[i] if t > 0}
            if len(pred_hits) > 0:
                batch_pures.append(len(true_hits & pred_hits) / len(intervals))
            else:
                batch_pures.append(0.0)
    return batch_pures if no_sum else sum(batch_pures) / len(batch_pures)


def merged_twoflash_acc(merged_mask, hit_times, device, no_sum=False):
    """Per-flash recall for exactly-two-flash samples.

    Returns ``(flash1_acc, flash2_acc)`` batch means, or the per-sample lists when
    ``no_sum=True``. Each flash scores 1.0 if its true bin lies in a predicted interval.
    """
    with torch.no_grad():
        B = merged_mask.shape[0]
        flash1_accs, flash2_accs = [], []
        for i in range(B):
            pred_hits = set(torch.nonzero(merged_mask[i], as_tuple=False).flatten().tolist())
            true_times = [int(t) for t in hit_times[i] if t > 0]
            if len(true_times) < 2:
                true_times = true_times + [None] * (2 - len(true_times))
            flash1_accs.append(1.0 if true_times[0] in pred_hits else 0.0)
            flash2_accs.append(1.0 if true_times[1] in pred_hits else 0.0)
    if no_sum:
        return flash1_accs, flash2_accs
    return sum(flash1_accs) / B, sum(flash2_accs) / B


def sum_photons_in_intervals_vecwgrad(photon_counts, merged_mask, keep_grads=True):
    """Sum per-bin photon counts within each merged interval (autograd-friendly).

    Returns a length-``B`` list; element ``b`` is ``[interval_sums]`` (a 1-element list
    holding a 1D tensor of per-interval sums). Set ``keep_grads=False`` to detach.
    """
    B = photon_counts.shape[0]
    results = []
    for b in range(B):
        mask = merged_mask[b, 0]          # [L]
        counts = photon_counts[b, 0]      # [L]
        # Assign a contiguous interval id to each positive bin; background = -1.
        interval_ids = mask * (mask.diff(prepend=mask.new_zeros(1)) == 1).cumsum(0)
        interval_ids[mask == 0] = -1
        valid_ids = interval_ids[interval_ids >= 0]
        valid_counts = counts[interval_ids >= 0]
        num_intervals = interval_ids.max().item() + 1 if valid_ids.numel() > 0 else 0
        interval_sums = counts.new_zeros(num_intervals)
        interval_sums.scatter_add_(0, valid_ids, valid_counts)
        results.append([interval_sums] if keep_grads else [interval_sums.detach()])
    return results


def max_photons_in_intervals(photon_counts, merged_mask, keep_grads=False):
    """Like :func:`sum_photons_in_intervals_vecwgrad` but takes the max per interval.

    Returns per-sample tensors (``keep_grads=True``) or detached CPU lists.
    """
    B = photon_counts.shape[0]
    results = []
    for b in range(B):
        mask = merged_mask[b, 0]
        counts = photon_counts[b, 0]
        interval_ids = mask * (mask.diff(prepend=mask.new_zeros(1)) == 1).cumsum(0)
        interval_ids[mask == 0] = -1
        valid_ids = interval_ids[interval_ids >= 0]
        valid_counts = counts[interval_ids >= 0]
        num_intervals = interval_ids.max().item() + 1 if valid_ids.numel() > 0 else 0
        if num_intervals == 0:
            results.append([])
            continue
        interval_max = torch.full(
            (num_intervals,), float("-inf"), device=counts.device, dtype=torch.float32
        )
        interval_max.scatter_reduce_(0, valid_ids, valid_counts.float(), reduce="amax", include_self=True)
        results.append(interval_max if keep_grads else interval_max.detach().cpu().tolist())
    return results


def interval_rmse(pred_sums_list, true_sums_list):
    """RMSE between predicted and true per-interval photon sums across the batch."""
    errors = []
    for pred, true in zip(pred_sums_list, true_sums_list):
        if len(pred) > 0:
            errors.append(torch.tensor(pred, device="cpu") - torch.tensor(true, device="cpu"))
    if not errors:
        return None
    errors = torch.cat(errors)
    return float(torch.sqrt((errors ** 2).mean()))
