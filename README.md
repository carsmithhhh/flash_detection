# flash_detection

Per-time-bin **flash detection** (classification) and **photon regression** on
LArTPC optical waveforms. Three architectures are provided -- a 1D U-Net, a
transformer encoder, and a conformer (plus a conformer variant) -- all sharing one
data pipeline, loss, training loop, and evaluation suite.

Each waveform may contain several flashes. A model predicts, for every time bin,
(1) whether a flash starts there and (2) how many photons it carries.

## Layout

```
flashdet/                 importable library
├── models/
│   ├── layers.py         shared blocks: ResidualBlock1D, PositionalEncoding, MultiLevelTokenizer
│   ├── unet.py           UNet1D
│   ├── transformer.py    TransformerModel
│   ├── conformer.py      ConformerModel, ConformerModelv2
│   ├── conformer_block.py  from-scratch Conformer encoder (no torchaudio)
│   └── __init__.py       MODEL_REGISTRY + build_model(name, **args)
├── data.py               WaveformDataset, dataloaders, train/val split
├── losses.py             mined_bce_loss (mining + Poisson regression), bce_loss
├── metrics.py            per-bin + merged-interval metrics, prediction post-processing
├── engine.py             train / validate / save_checkpoint
└── utils.py              config loading, seeding, optimizer/scheduler, checkpoint loading

train.py                  CLI: train a model from a YAML config
evaluate.py               CLI: benchmark trained models, save .npy stats
configs/                  one YAML per model + evaluation.yaml
examples/                 example notebooks (simulate, train, lr search, plots)
archive/                  the original research notebooks and scripts (kept for reference)
```

## Install
** I haven't extensively tested environment setup with the new refactor so you may have to play around with this a little. **
```bash
pip install -r requirements.txt
# optional, makes `import flashdet` work from anywhere:
pip install -e .
```

You may not need to install anything if running on SDF (many packages are already installed), or can just install as you try to run and it throws errors.

## Quickstart

**1. Get data.** Generate a dataset with `examples/00_simulate_dataset.ipynb` (uses
the sibling [`waveforms`](../waveforms) simulation package), or point a config at an
existing `.npy` file. A dataset is a pickled dict with `waveforms`, `arrival_times`,
and `num_photons` (see `flashdet/data.py`).

Adjust paths in the `config` file, and make sure `wandb` is setup correctly to route to your account. Everything is still hardcoded to my directories. If there are
files that are buried in my directories I can move them elsewhere if it is more convenient, just let me know.

**2. Train.** Pick a config and run:

```bash
python train.py --config configs/conformer.yaml
# quick overrides for experiments:
python train.py --config configs/conformer.yaml --epochs 5 --lr 3e-4 --no-wandb
```

On the cluster, submit a batch job (defaults to `configs/conformer.yaml`):

```bash
sbatch training_job.sbatch configs/unet.yaml
```
This is the most convenient way to submit long training jobs and let them run while you are away from the computer! For small tests/debugging, it is usually better
to work in a jupyter notebook or terminal session.

**3. Evaluate.** Edit `configs/evaluation.yaml` to list trained checkpoints, then:

```bash
python evaluate.py --config configs/evaluation.yaml   # writes results/*.npy
```

**4. Plot.** `examples/03_plots_and_figures.ipynb` loads the `results/*.npy` files
and reproduces the key benchmark figures.

## Configs

Each `configs/<model>.yaml` is self-contained: `model` (registry `name` + `args`),
`data` (path, batch size, split ratios), `optim` (lr, weight decay, scheduler),
`train` (epochs, mode, device, checkpoint locations), and `wandb` (logging toggle).
Paths default to the SDF cluster — update `data.path` / `*_checkpoint*` for your
environment. The training loss `mode` is `mined_bce` (hard-negative mining +
regression) or `bce` (plain class-balanced BCE).

## Adding a model

1. Implement it in a new file under `flashdet/models/` returning
   `(class_logits, reg_logits)` of shape `[B, 1, L]`.
2. Register it in `flashdet/models/__init__.py` (`MODEL_REGISTRY`).
3. Copy a config in `configs/`, set `model.name` to the new key, and train.

## Examples

| Notebook | Shows |
| --- | --- |
| `00_simulate_dataset.ipynb` | Generate and inspect a waveform dataset. |
| `01_train_models.ipynb`     | Build and train each architecture via the library. |
| `02_lr_search.ipynb`        | A small learning-rate sweep. |
| `03_plots_and_figures.ipynb`| Reproduce the benchmark figures from `evaluate.py` output. |
