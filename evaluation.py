'''
Handles model output post-processing (merging bins), and evaluation of model performance.
'''

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time

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

    k = skip_tol * 2 + 1  
    kernel = torch.ones(1, 1, k, device=device)

    dilated = F.conv1d(active.float(), kernel, padding=skip_tol) > 0
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

        batch_pures = []
        for i in range(B):
            # within predicted window bins
            pred_hits = set(torch.nonzero(merged_mask[i], as_tuple=False).flatten().tolist())
            mask_row = merged_mask[i, 0]  # [L]
            #print("mask row: ", mask_row)
            intervals = mask_to_intervals(mask_row) #[[s1, e1], [s2, e2], ...]
            #print("intervals: ", intervals)
            true_hits = set(int(t) for t in hit_times[i] if t > 0)
            #print("true hits: ", true_hits)

            if len(pred_hits) > 0:
                correct_hits = len(true_hits.intersection(pred_hits))
                pur = correct_hits / len(intervals) # PURITY -> DIVIDE BY # merged windows
            else:
                pur = 0.0
            batch_pures.append(pur)

    if no_sum:
        return batch_pures
    else:
        return sum(batch_pures) / len(batch_pures)

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

def merged_twoflash_acc(merged_mask, hit_times, device, no_sum=False):
    '''
    Parameters:
        - merged_mask: mask of shape [B, 1, L] with 1's for merged regions
        - hit_times: list of arrays or lists, truth hit times for each sample [time1, time2]
    Returns:
        - avg_flash1_acc, avg_flash2_acc (or arrays of flash1_accs, flash2_accs)
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

def interval_rmse(pred_sums_list, true_sums_list):
    errors = []
    for pred, true in zip(pred_sums_list, true_sums_list):
        if len(pred) > 0:
            errors.append(torch.tensor(pred, device='cpu') - torch.tensor(true, device='cpu'))
    if not errors:
        return None
    errors = torch.cat(errors)
    return float(torch.sqrt((errors**2).mean()))