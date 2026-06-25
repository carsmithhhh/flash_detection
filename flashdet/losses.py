"""Training losses for joint flash detection + photon regression.

The headline loss is :func:`mined_bce_loss`. Because true flash bins are extremely
rare (a handful of positive bins among thousands), it trains the classifier on a
*mined* subset of bins per sample: the true hits, plus hard negatives near each hit
(where the waveform tail makes the decision hard), plus a few random negatives. The
regression term is gated on confidently-positive bins and added with a small weight.
"""

import numpy as np
import torch
import torch.nn.functional as F


# Hard-negative window (samples) around each hit, ~ the delayed waveform width.
WF_WIDTH = 900
N_HARD_NEG = 500   # hard negatives sampled within WF_WIDTH of a hit
N_RAND_NEG = 100   # random negatives sampled elsewhere
REG_SCALE = 0.1    # weight of the regression term relative to classification


def mined_bce_loss(
    data, hit_times, photon_list, class_output, reg_output, epoch, device,
    include_reg=True, logger=None,
):
    """Class-balanced mined BCE (+ optional Poisson regression) loss.

    Args:
        data: input waveforms ``[B, 1, L]`` (used only for shape).
        hit_times: length-``B`` iterable of true hit-bin indices per sample.
        photon_list: parallel iterable of photon counts per hit.
        class_output: per-bin classification logits ``[B, 1, L]``.
        reg_output: per-bin regression output ``[B, 1, L]`` (log-rate; softplus-ed here).
        epoch: current epoch (kept for schedule hooks; unused by default).
        include_reg: add the regression term (gated on sigmoid(class) > 0.5).

    Returns an 8-tuple whose first element is the scalar loss; the remaining elements
    (sampled mask, masked targets/outputs, full targets) are exposed for debugging and
    plotting and are otherwise ignored by the training loop.
    """
    data = data.squeeze(1)
    offset = 0
    N, L = data.shape
    target = torch.zeros((N, L), dtype=torch.float32, device=device)
    photon_target = torch.zeros((N, L), dtype=torch.float32, device=device)
    rng = np.random.default_rng()

    # Boolean mask [N, L] of bins that contribute to the classification loss.
    sampled_indices = torch.zeros((N, L), dtype=torch.bool, device=device)

    for i, times in enumerate(hit_times):
        if (
            times is None
            or (isinstance(times, (list, np.ndarray)) and len(times) == 0)
            or (isinstance(times, (list, np.ndarray)) and np.all(np.array(times) < 0))
        ):
            continue  # no flashes in this waveform

        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        else:
            times = np.asarray(times).flatten()

        # True hits: always included, with their photon-count targets.
        hit_indices = []
        for j, t in enumerate(times):
            if t < 0:
                continue
            t_idx = int(np.clip(t + offset, 0, L - 1))
            target[i, t_idx] = 1.0
            photon_target[i, t_idx] = photon_list[i][j]
            sampled_indices[i, t_idx] = True
            hit_indices.append(t_idx)

        # Hard negatives: bins within WF_WIDTH of a hit (excluding the hits themselves).
        wf_neg_candidates = set()
        for t_idx in hit_indices:
            wf_neg_candidates.update(range(max(0, t_idx), min(L, t_idx + WF_WIDTH + 1)))
        wf_neg_candidates.difference_update(hit_indices)
        wf_neg_candidates = list(wf_neg_candidates)
        if wf_neg_candidates:
            chosen = rng.choice(wf_neg_candidates, size=min(N_HARD_NEG, len(wf_neg_candidates)), replace=False)
            sampled_indices[i, chosen] = True

        # Random negatives: bins far from any hit.
        forbidden = set(hit_indices).union(wf_neg_candidates)
        random_neg_candidates = list(set(range(L)) - forbidden)
        if random_neg_candidates:
            chosen = rng.choice(random_neg_candidates, size=min(N_RAND_NEG, len(random_neg_candidates)), replace=False)
            sampled_indices[i, chosen] = True

    masked_class_output = class_output.squeeze(1)[sampled_indices]
    masked_target = target[sampled_indices]

    # Regression is computed only on bins the classifier is already confident about.
    masked_reg_output = None
    masked_photon_target = None
    if include_reg:
        mask = (torch.sigmoid(class_output) > 0.5).squeeze(1)
        masked_reg_output = reg_output.squeeze(1)[mask]
        masked_photon_target = photon_target[mask]

    if masked_target.numel() == 0:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss, sampled_indices, masked_target, masked_class_output, class_output, target, masked_reg_output, masked_photon_target

    # Balance positives/negatives dynamically within the mined subset.
    n_pos = masked_target.sum().item()
    pos_weight = torch.tensor([(masked_target.numel() - n_pos) / max(1, n_pos)], device=device)
    class_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(masked_class_output, masked_target)

    reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    if include_reg and masked_reg_output.numel() > 0:
        # log_input=False with softplus keeps the predicted rate positive.
        reg_loss = torch.nn.PoissonNLLLoss(log_input=False)(
            F.softplus(masked_reg_output), masked_photon_target
        )

    loss = class_loss + REG_SCALE * reg_loss
    return loss, sampled_indices, masked_target, masked_class_output, class_output, target, masked_reg_output, masked_photon_target


def bce_loss(data, hit_times, class_output, device):
    """Plain class-balanced BCE over *all* bins (no mining, no regression).

    Returns ``(loss, target)`` where ``target`` is the per-bin binary label map.
    Positive bins are up-weighted by ``L`` to counter the extreme class imbalance.
    """
    data = data.squeeze(1)
    N, L = data.shape
    offset = 0
    target = torch.zeros((N, L), dtype=torch.float32, device=device)

    for i, times in enumerate(hit_times):
        if (
            times is None
            or (isinstance(times, (list, np.ndarray)) and len(times) == 0)
            or (isinstance(times, (list, np.ndarray)) and np.all(np.array(times) < 0))
        ):
            continue
        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        else:
            times = np.asarray(times).flatten()
        for t in times:
            if t < 0:
                continue
            target[i, int(np.clip(t + offset, 0, L - 1))] = 1.0

    pos_weight = torch.tensor([L], device=device)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(class_output.squeeze(1), target)
    return loss, target
