# Archive

Original research material, kept for reference. **Not** part of the clean library —
nothing here is needed to use `flashdet`.

- `notebooks/`, `tutorial_notebooks/` — the original development notebooks, plus
  saved result artifacts (`*.npy`, `*.png`, `animations/*.gif`) that
  `examples/03_plots_and_figures.ipynb` can read.
- `model_list.yaml`, `oct_models.yaml`, `performance_analysis_config.yaml` — the old
  evaluation configs, now consolidated into `../configs/evaluation.yaml`.
- `wandb/`, `wandb_data/` — Weights & Biases run logs.

These notebooks import the original flat modules (`model.py`, `data_utils.py`,
`hybrid_loss.py`, `evaluation.py`, `utils.py`, etc.), which were refactored into the
`flashdet/` package and removed from the repo root. The deleted files remain in git
history (`git log --diff-filter=D --name-only`) if you need to consult them; the
equivalent functionality now lives in `flashdet/`.
