#!/usr/bin/env python
"""Benchmark trained models and save per-model performance statistics.

Usage:
    python evaluate.py --config configs/evaluation.yaml

Two studies (toggle each in the config):

  single_flash  - on a single-flash dataset, scan metrics vs photon count:
                  merged accuracy / purity, merged interval width, and the
                  reconstructed-photon fraction (predicted / true photons).

  double_flash  - on a two-flash dataset with a fixed photon count, scan metrics
                  vs the time gap Δt between the two flashes: per-flash merged
                  accuracy, purity, and reconstructed-photon fraction.

Results are written as ``.npy`` dicts under ``output_dir`` for plotting (see
``examples/03_plots_and_figures.ipynb``).
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from flashdet import metrics
from flashdet.data import make_dataloader
from flashdet.utils import instantiate_models, load_config, set_seed


def single_flash_study(models, data_path, study_cfg, device, seed=42):
    """Scan merged metrics and reconstructed-photon fraction vs photon count.

    The single-flash dataset is ordered by photon count in blocks of
    ``batches_per_photon`` batches; metrics are averaged within each block so the
    returned lists are indexed by increasing photon count.
    """
    loader = make_dataloader(data_path, seed=seed, batch_size=25, shuffle=False)
    batches_per_photon = study_cfg.get("batches_per_photon", 4)
    results = {n: {"merge_acc": [], "merge_pure": [], "interval": [], "reco_frac": []} for n in models}

    interval_bins = {n: [] for n in models}
    acc = {n: 0.0 for n in models}
    pure = {n: 0.0 for n in models}
    reco = {n: 0.0 for n in models}

    with torch.no_grad():
        for i, (data, _, hit_times, photon_target, _) in enumerate(tqdm(loader, desc="single-flash")):
            data, photon_target = data.to(device), photon_target.to(device)

            for name, (model, reg_loss) in models.items():
                predict_fn = torch.expm1 if reg_loss == "mse" else torch.exp
                class_output, reg_output = model(data, mode="bce")
                merged_mask = metrics.merge_bins(class_output, skip_tol=5)

                if study_cfg.get("merged_window_width"):
                    for b in range(merged_mask.shape[0]):
                        intervals = metrics.mask_to_intervals(merged_mask[b, 0])
                        interval_bins[name].extend(e - s + 1 for (s, e) in intervals)

                if study_cfg.get("reco_frac"):
                    pred_sums = metrics.sum_photons_in_intervals_vecwgrad(predict_fn(reg_output), merged_mask, keep_grads=False)
                    true_sums = metrics.sum_photons_in_intervals_vecwgrad(photon_target, merged_mask, keep_grads=False)
                    pred = torch.stack([x[0].sum() for x in pred_sums])
                    true = torch.stack([x[0].sum() for x in true_sums])
                    mask = true > 0
                    if mask.any():
                        reco[name] += torch.mean(pred[mask] / true[mask]).item()

                if study_cfg.get("merged_acc"):
                    acc[name] += metrics.merged_class_acc(merged_mask, hit_times, device)
                if study_cfg.get("merged_pure"):
                    pure[name] += metrics.merged_class_purity(merged_mask, hit_times, device)

            if (i + 1) % batches_per_photon == 0:
                for name in models:
                    results[name]["reco_frac"].append(reco[name] / batches_per_photon)
                    results[name]["merge_acc"].append(acc[name] / batches_per_photon)
                    results[name]["merge_pure"].append(pure[name] / batches_per_photon)
                    avg_interval = np.mean(interval_bins[name]) if interval_bins[name] else 0.0
                    results[name]["interval"].append(avg_interval)
                    interval_bins[name], acc[name], pure[name], reco[name] = [], 0.0, 0.0, 0.0

    return results


def double_flash_study(models, data_path, study_cfg, device, seed=42, max_dt=1501):
    """Scan per-flash merged metrics vs the inter-flash time gap Δt.

    Accumulates each metric into a length-``max_dt`` array indexed by Δt (in bins),
    then divides by the per-Δt sample count to produce mean curves.
    """
    loader = make_dataloader(data_path, seed=seed, batch_size=25, shuffle=False)
    results = {
        n: {
            "bin_counts": torch.zeros(max_dt),
            "reco_frac_flash1": torch.zeros(max_dt),
            "reco_frac_flash2": torch.zeros(max_dt),
            "merge_acc_flash1": torch.zeros(max_dt),
            "merge_acc_flash2": torch.zeros(max_dt),
            "merge_pure": torch.zeros(max_dt),
        }
        for n in models
    }

    with torch.no_grad():
        for data, _, hit_times, photon_target, _ in tqdm(loader, desc="double-flash"):
            data, photon_target = data.to(device), photon_target.to(device)
            dt_index = torch.tensor([int(t[1] - t[0]) for t in hit_times], dtype=torch.long)

            for name, (model, reg_loss) in models.items():
                predict_fn = torch.expm1 if reg_loss == "mse" else torch.exp
                class_output, reg_output = model(data, mode="bce")
                merged_mask = metrics.merge_bins(class_output, skip_tol=5)

                pred_sums = true_sums = None
                if study_cfg.get("reco_frac"):
                    pred_sums = metrics.sum_photons_in_intervals_vecwgrad(predict_fn(reg_output), merged_mask, keep_grads=False)
                    true_sums = metrics.sum_photons_in_intervals_vecwgrad(photon_target, merged_mask, keep_grads=False)
                f1, f2 = ([], [])
                if study_cfg.get("merged_acc"):
                    f1, f2 = metrics.merged_twoflash_acc(merged_mask, hit_times, device, no_sum=True)
                purity_ls = []
                if study_cfg.get("merged_pure"):
                    purity_ls = metrics.merged_class_purity(merged_mask, hit_times, device, no_sum=True)

                for b, idx in enumerate(dt_index):
                    results[name]["bin_counts"][idx] += 1
                    if study_cfg.get("reco_frac"):
                        valid = torch.where(true_sums[b][0] > 0)[0].flatten()
                        if valid.numel() >= 2:
                            results[name]["reco_frac_flash1"][idx] += (pred_sums[b][0][valid[0]] / true_sums[b][0][valid[0]]).item()
                            results[name]["reco_frac_flash2"][idx] += (pred_sums[b][0][valid[1]] / true_sums[b][0][valid[1]]).item()
                    if study_cfg.get("merged_acc"):
                        results[name]["merge_acc_flash1"][idx] += f1[b]
                        results[name]["merge_acc_flash2"][idx] += f2[b]
                    if study_cfg.get("merged_pure"):
                        results[name]["merge_pure"][idx] += purity_ls[b]

    for name in models:
        counts = results[name]["bin_counts"].clone()
        counts[counts == 0] = 1  # avoid div-by-zero for empty Δt bins
        for key in results[name]:
            if key != "bin_counts":
                results[name][key] /= counts
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda")
    output_dir = cfg.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    models = instantiate_models(cfg["models"], device)
    print("evaluating:", list(models.keys()))

    studies = cfg.get("studies", {})
    single = studies.get("single_flash", {})
    if single.get("enabled"):
        results = single_flash_study(models, cfg["data"]["single_flash"], single, device, cfg.get("seed", 42))
        out = os.path.join(output_dir, "single_flash.npy")
        np.save(out, results, allow_pickle=True)
        print(f"saved {out}")

    double = studies.get("double_flash", {})
    if double.get("enabled"):
        path = cfg["data"]["double_flash_template"].format(photon=double.get("fixed_photon", "random"))
        results = double_flash_study(models, path, double, device, cfg.get("seed", 42))
        out = os.path.join(output_dir, "double_flash.npy")
        np.save(out, results, allow_pickle=True)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
