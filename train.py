#!/usr/bin/env python
"""Train a flash-detection model from a YAML config.

Usage:
    python train.py --config configs/conformer.yaml

The config fully specifies the run (model, data, optimizer, schedule, logging,
checkpoint location); see any file in ``configs/`` for the expected structure.
CLI flags override the matching config fields for quick experiments.
"""

import argparse

import torch

from flashdet import engine
from flashdet.data import make_train_val_loaders
from flashdet.models import build_model
from flashdet.utils import build_optimizer, build_scheduler, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to a YAML training config.")
    p.add_argument("--epochs", type=int, help="Override train.epochs.")
    p.add_argument("--lr", type=float, help="Override optim.lr.")
    p.add_argument("--data", help="Override data.path.")
    p.add_argument("--device", help="Override train.device (e.g. cuda, cpu).")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides.
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.data is not None:
        cfg["data"]["path"] = args.data
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = cfg["train"].get("device", "cuda")

    # Data
    data_cfg = cfg["data"]
    train_loader, val_loader, _ = make_train_val_loaders(
        data_cfg["path"],
        seed=seed,
        batch_size=data_cfg.get("batch_size", 25),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.0),
        num_workers=data_cfg.get("num_workers", 0),
    )
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    # Model
    model = build_model(cfg["model"]["name"], **(cfg["model"].get("args") or {})).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg['model']['name']}  parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["optim"].get("scheduler"))

    # Optional Weights & Biases logging
    logger = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb

        logger = wandb.init(
            project=wandb_cfg.get("project", "flash_detection"),
            name=wandb_cfg.get("name"),
            config=cfg,
        )
        wandb.watch(model, log="all", log_freq=100)

    # Train
    train_cfg = cfg["train"]
    engine.train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device=device,
        epochs=train_cfg["epochs"],
        mode=train_cfg.get("mode", "mined_bce"),
        mse=train_cfg.get("mse", False),
        checkpoint_dir=train_cfg.get("checkpoint_dir"),
        val_every=train_cfg.get("val_every", 5),
        save_every=train_cfg.get("save_every", 5),
        logger=logger,
    )

    # Save final checkpoint
    final_path = train_cfg.get("final_checkpoint")
    if final_path:
        engine.save_checkpoint(model, optimizer, scheduler, final_path)
        print(f"saved final checkpoint to {final_path}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
