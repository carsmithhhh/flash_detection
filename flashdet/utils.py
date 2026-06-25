"""Config loading, reproducibility, and object-construction helpers.

These glue functions let ``train.py``/``evaluate.py`` stay thin: they turn the plain
dicts parsed from a YAML config into seeded RNGs, optimizers, schedulers, and loaded
checkpoints.
"""

import random

import numpy as np
import torch
import yaml

from .models import build_model


def set_seed(seed=42):
    """Seed Python, NumPy and Torch (incl. CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    """Parse a YAML config file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer(model, optim_cfg):
    """Build an Adam optimizer from ``{lr, weight_decay}`` (sensible defaults)."""
    return torch.optim.Adam(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-4),
        weight_decay=optim_cfg.get("weight_decay", 1e-6),
    )


def build_scheduler(optimizer, sched_cfg):
    """Build a ``ReduceLROnPlateau`` scheduler from ``{factor, patience}``."""
    sched_cfg = sched_cfg or {}
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 2),
    )


def instantiate_models(models_cfg, device="cuda"):
    """Build and load a set of trained models from a ``{name: entry}`` dict.

    Each entry provides ``class`` (a registry name), ``args``, ``checkpoint``, and an
    optional ``reg_loss`` tag. Entries with ``include: False`` are skipped. Returns
    ``{name: [model, reg_loss]}`` with each model in ``eval`` mode on ``device``.
    """
    models = {}
    for name, cfg in models_cfg.items():
        if not isinstance(cfg, dict) or not cfg.get("include", True):
            continue
        model = build_model(cfg["class"], **(cfg.get("args") or {})).to(device)
        checkpoint = torch.load(cfg["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[name] = [model, cfg.get("reg_loss")]
    return models


def load_models(config_path, device="cuda"):
    """Convenience wrapper: load a YAML model-list file and instantiate its models."""
    with open(config_path, "r") as f:
        return instantiate_models(yaml.safe_load(f), device)
