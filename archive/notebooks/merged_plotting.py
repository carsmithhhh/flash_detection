"""
Compiled benchmark plotting script (multi-source).

Models come from three files:
    transformer_2M, conformer_5M  <- DATA_PATH      (c5M_t2M_deltastats.npy)
    conformer_v5_drop             <- DATA_PATH_v5    (conv5_drop_delta_t_results.npy)
    unet                          <- DATA_PATH_unet  (test_deltastats.npy)

Produces the 2x2 figure and saves it:
    top row    -> Merged Accuracy            (Flash 1, Flash 2)
    bottom row -> Reconstructed Photon Fraction (Flash 1, Flash 2)

The bottom-row y-axis tick LABELS are rescaled by Y_SCALE
(true value = Y_SCALE * printed value). Only the label text changes -- curves,
gridlines, and the black reference line stay exactly where they are drawn.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH      = "/sdf/home/c/carsmith/flash_reconstruction/flash_detection/notebooks/performance_analysis/delta_t_benchmark/c5M_t2M_deltastats.npy"
DATA_PATH_v5   = "/sdf/home/c/carsmith/flash_reconstruction/flash_detection/notebooks/performance_analysis/delta_t_benchmark/conv5_drop_delta_t_results.npy"
DATA_PATH_unet = "/sdf/home/c/carsmith/flash_reconstruction/flash_detection/notebooks/performance_analysis/delta_t_benchmark/deltastats_3trans_100k.npy"

OUTPUT_PATH = "merged_accuracy_and_reco_frac.png"
BIN_SIZE = 30
Y_SCALE = 0.8   # bottom-row Reco Frac: true value = Y_SCALE * printed tick value

# Which key to pull from each file for each output model name.
# If a key is missing and the file has exactly one model, that one is used.
SOURCES = {
    "transformer_2M":    (DATA_PATH,      "transformer_2M"),
    "conformer_5M":      (DATA_PATH,      "conformer_5M"),
    "conformer_v5_drop": (DATA_PATH_v5,   "conformer_v5_drop"),
    "unet":              (DATA_PATH_unet, "unet"),
}

model_names = ["unet", "transformer_2M", "conformer_5M", "conformer_v5_drop"]
colors  = ["steelblue", "mediumseagreen", "orange", "red", "red", "pink",
           "darkslategrey", "darkblue", "gray"]
markers = ["o", "s", "D", "v", "^", "o", "s", "o", "D"]


# --------------------------------------------------------------------------
# Aggregation helper (unchanged from your original)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Load all models + aggregate
# --------------------------------------------------------------------------
_cache = {}
raw_results = {}
for out_name, (path, key) in SOURCES.items():
    if path not in _cache:
        _cache[path] = np.load(path, allow_pickle=True).item()
        print(f"{path}\n  keys: {list(_cache[path].keys())}")
    raw_results[out_name] = pick(_cache[path], key, path)

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
    ax.axhline(y=1.2, color="black", linestyle="-", linewidth=1.5)

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
        # Relabel y ticks to Y_SCALE x the drawn value; geometry untouched.
        ax.yaxis.set_major_formatter(reco_formatter)

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved figure to {OUTPUT_PATH}")
plt.show()