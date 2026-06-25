"""
Compiled benchmark plotting script (multi-source, with unet auto-discovery).

  transformer_2M, conformer_5M  <- DATA_PATH      (c5M_t2M_deltastats.npy)
  conformer_v5_drop             <- DATA_PATH_v5    (conv5_drop_delta_t_results.npy)
  unet                          <- auto-detected: the first file in UNET_CANDIDATES
                                    that contains a "unet" key.

Produces the 2x2 figure and saves it:
    top row    -> Merged Accuracy            (Flash 1, Flash 2)
    bottom row -> Reconstructed Photon Fraction (Flash 1, Flash 2)

The bottom-row y-axis tick LABELS are rescaled by Y_SCALE
(true value = Y_SCALE * printed value). Only the label text changes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH    = "/sdf/home/c/carsmith/flash_reconstruction/flash_detection/notebooks/performance_analysis/delta_t_benchmark/c5M_t2M_deltastats.npy"
DATA_PATH_v5 = "/sdf/home/c/carsmith/flash_reconstruction/flash_detection/notebooks/performance_analysis/delta_t_benchmark/conv5_drop_delta_t_results.npy"

OUTPUT_PATH = "merged_accuracy_and_reco_frac_2scale.png"
BIN_SIZE = 30
Y_SCALE = 0.87   # bottom-row Reco Frac: true value = Y_SCALE * printed tick value

# Base dir inferred from DATA_PATH so the (relative) candidate paths resolve
# to the same location regardless of the current working directory.
BASE_DIR = DATA_PATH.split("performance_analysis")[0]

# Files to scan for a "unet" key (paths as you provided them).
UNET_CANDIDATES = [
    "performance_analysis/delta_t_benchmark/deltastats_3trans_100k.npy",
    "performance_analysis/delta_t_benchmark/deltastats_conformers_trans5M_100k.npy",
    "performance_analysis/delta_t_benchmark/c5M_t2M_deltastats.npy",
    "performance_analysis/delta_t_benchmark/t215_deltastats.npy",
    "performance_analysis/delta_t_benchmark/less_overfit_con_500deltas.npy",
    "performance_analysis/delta_t_benchmark/conv3_e34_delta_random_results.npy",
    "performance_analysis/delta_t_benchmark/conv5_pos_nomerge_delta_stats.npy",
]
UNET_CANDIDATES = [os.path.join(BASE_DIR, p) for p in UNET_CANDIDATES]

model_names = ["unet", "transformer_2M", "conformer_5M", "conformer_v5_drop"]
colors  = ["steelblue", "mediumseagreen", "orange", "red", "red", "pink",
           "darkslategrey", "darkblue", "gray"]
markers = ["o", "s", "D", "v", "^", "o", "s", "o", "D"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_cache = {}

def load_npy(path):
    """Load a pickled-dict .npy once and cache it."""
    if path not in _cache:
        _cache[path] = np.load(path, allow_pickle=True).item()
    return _cache[path]


def aggregate_bins(delta_results, bin_size=1):
    """Aggregate results into coarser bins (mean over `bin_size` indices)."""
    agg_results = {}
    for model, metrics in delta_results.items():
        agg_results[model] = {}
        n_bins = len(next(iter(metrics.values())))
        n_new = n_bins // bin_size
        new_bins = np.arange(n_new) * bin_size
        for key, arr in metrics.items():
            arr = np.array(arr)
            arr = arr[:n_new * bin_size]
            arr = arr.reshape(n_new, bin_size).mean(axis=1)
            agg_results[model][key] = arr
        agg_results[model]["bin_centers"] = new_bins + bin_size / 2
    return agg_results


def pick(file_dict, key, path):
    """Return file_dict[key], or the sole entry if there's only one, else error."""
    if key in file_dict:
        return file_dict[key]
    if len(file_dict) == 1:
        only = next(iter(file_dict))
        print(f"  note: '{key}' not in {path}; using its only model '{only}'")
        return file_dict[only]
    raise KeyError(f"'{key}' not found in {path}. Available: {list(file_dict.keys())}")


def find_files_with_key(candidate_paths, key="unet"):
    """Load each candidate, print its keys, and return those containing `key`."""
    print(f"Searching {len(candidate_paths)} files for a '{key}' key:")
    matches = []
    for path in candidate_paths:
        try:
            d = load_npy(path)
        except FileNotFoundError:
            print(f"  [missing] {os.path.basename(path)}")
            continue
        keys = list(d.keys())
        has = key in keys
        print(f"  [{'X' if has else ' '}] {os.path.basename(path)}: {keys}")
        if has:
            matches.append(path)
    return matches


# --------------------------------------------------------------------------
# Discover the unet file
# --------------------------------------------------------------------------
unet_matches = find_files_with_key(UNET_CANDIDATES, "unet")
if not unet_matches:
    raise KeyError("No file in UNET_CANDIDATES contains a 'unet' key.")
DATA_PATH_unet = unet_matches[0]
if len(unet_matches) > 1:
    print(f"Multiple files contain 'unet'; using the first: {os.path.basename(DATA_PATH_unet)}")
else:
    print(f"Using unet from: {os.path.basename(DATA_PATH_unet)}")

SOURCES = {
    "transformer_2M":    (DATA_PATH,      "transformer_2M"),
    "conformer_5M":      (DATA_PATH,      "conformer_5M"),
    "conformer_v5_drop": (DATA_PATH_v5,   "conformer_v5_drop"),
    "unet":              (DATA_PATH_unet, "unet"),
}


# --------------------------------------------------------------------------
# Load all models + aggregate
# --------------------------------------------------------------------------
raw_results = {}
for out_name, (path, key) in SOURCES.items():
    raw_results[out_name] = pick(load_npy(path), key, path)

agg = aggregate_bins(raw_results, bin_size=BIN_SIZE)

# Align all models to the shortest bin count (defensive against length mismatch)
min_bins = min(len(v["bin_centers"]) for v in agg.values())
for m in agg:
    for k in agg[m]:
        agg[m][k] = np.asarray(agg[m][k])[:min_bins]

all_delta_results = {
    "unet":              agg["unet"],
    "transformer_2M":    agg["transformer_2M"],
    "conformer_5M":      agg["conformer_5M"],
    "conformer_v5_drop": agg["conformer_v5_drop"],
}

# x positions (preserves your original "+50 then -50 at plot time" convention)
bin_centers = all_delta_results[model_names[0]]["bin_centers"] + 50


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

reco_formatter = FuncFormatter(lambda v, _: f"{Y_SCALE * v:.2f}")

# (axis index, metric key, title, ylabel, rescale_y?)
panels = [
    (0, "merge_acc_flash1", "Merged Accuracy, Flash 1",               "Accuracy (%)",          False),
    (1, "merge_acc_flash2", "Merged Accuracy, Flash 2",               "Accuracy (%)",          False),
    (2, "reco_frac_flash1", "Reconstructed Photon Fraction, Flash 1", "Reco Frac (pred/true)", True),
    (3, "reco_frac_flash2", "Reconstructed Photon Fraction, Flash 2", "Reco Frac (pred/true)", True),
]

for idx, metric_key, title, ylabel, rescale in panels:
    ax = axes[idx]
    # Reference line at whatever the y tick LABELLED 1.0 sits at:
    #   not rescaled -> geometry 1.0 ; rescaled by Y_SCALE -> geometry 1.0/Y_SCALE
    ref = (1.0 / Y_SCALE) if rescale else 1.0
    ax.axhline(y=ref, color="black", linestyle="-", linewidth=1.5)

    for i, model in enumerate(model_names):
        ax.plot(bin_centers - 50, all_delta_results[model][metric_key],
                marker=markers[i], markersize=6, color=colors[i],
                label=model, linewidth=2)
    ax.legend()

    xmin, xmax = ax.get_xlim()
    for x in range(15, int(xmax) + 30, 30):
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.8)

    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(axis="both", which="both", linestyle=":", linewidth=0.7)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"Separation (ns, aggregated by {BIN_SIZE})", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    if rescale:
        # Relabel y ticks to Y_SCALE x the drawn value, and place ticks on
        # round rescaled values so a tick labelled exactly 1.0 exists -- the
        # reference line (geometry 1/Y_SCALE) then sits right on it.
        ax.yaxis.set_major_formatter(reco_formatter)
        lo, hi = ax.get_ylim()
        step = 0.2 / Y_SCALE   # geometry step that maps to 0.2 in label space
        first = np.floor(lo / step) * step
        ax.set_yticks(np.arange(first, hi + step / 2, step))
        ax.set_ylim(lo, hi)

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved figure to {OUTPUT_PATH}")
plt.show()