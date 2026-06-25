"""Training / validation loop for the detection + regression models.

:func:`train` runs the full schedule and returns a ``results`` dict of per-epoch
metric lists. Validation and checkpointing happen on fixed intervals. Logging is
optional: pass any object with a ``.log(dict)`` method (e.g. a ``wandb`` run) as
``logger``.
"""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import metrics
from .losses import bce_loss, mined_bce_loss


def _match_length(class_output, reg_output, target_len):
    """Pad/crop model outputs to ``target_len`` (U-Net lengths need not be powers of 2)."""
    if class_output.shape[-1] == target_len:
        return class_output, reg_output
    diff = target_len - class_output.shape[-1]
    if diff > 0:
        return F.pad(class_output, (0, diff)), F.pad(reg_output, (0, diff))
    return class_output[..., :target_len], reg_output[..., :target_len]


def save_checkpoint(model, optimizer, scheduler, path):
    """Save model/optimizer/scheduler state to ``path`` (parent dirs created)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


@torch.no_grad()
def validate(model, val_loader, device, mode="mined_bce", mse=False):
    """Run one validation pass, returning ``(loss, acc, purity, reg_rmse)`` means."""
    model.eval()
    totals = {"loss": 0.0, "acc": 0.0, "pure": 0.0, "rmse": 0.0}

    for data, _, hit_times, photon_target, photon_list in val_loader:
        data, photon_target = data.to(device), photon_target.to(device)
        class_output, reg_output = model(data, mode="bce")
        class_output, reg_output = _match_length(class_output, reg_output, data.shape[-1])

        if mode == "mined_bce":
            loss, *_ = mined_bce_loss(data, hit_times, photon_list, class_output, reg_output, 0, device)
            totals["rmse"] += metrics.regression_rmse(hit_times, photon_target, reg_output, class_output, device, mse=mse)
        else:  # 'bce'
            loss, _ = bce_loss(data, hit_times, class_output, device)

        totals["loss"] += loss.item()
        totals["acc"] += metrics.overall_class_acc(hit_times, class_output, device)
        totals["pure"] += metrics.overall_class_purity(hit_times, class_output, device)

    n = max(1, len(val_loader))
    model.train()
    return totals["loss"] / n, totals["acc"] / n, totals["pure"] / n, totals["rmse"] / n


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    mode="mined_bce",
    mse=False,
    checkpoint_dir=None,
    val_every=5,
    save_every=5,
    logger=None,
):
    """Train ``model`` for ``epochs`` and return a dict of per-epoch metric lists.

    Args:
        train_loader / val_loader: yield ``(data, arrival, hit_times, photon_bin, photon_list)``.
        optimizer / scheduler: standard PyTorch objects; ``scheduler.step(val_loss)`` is
            called after each validation pass (use a plateau scheduler).
        mode: ``'mined_bce'`` (mined loss + regression) or ``'bce'`` (plain BCE).
        mse: regression decode convention passed through to the RMSE metric.
        checkpoint_dir: if set, ``{epoch}.pth`` is written every ``save_every`` epochs.
        val_every / save_every: epoch intervals for validation / checkpointing.
        logger: optional object with ``.log(dict)`` (e.g. a wandb run).
    """
    model.train()
    optimizer.zero_grad()
    results = {k: [] for k in [
        "train_loss", "train_acc", "train_pure", "train_reg_rmse",
        "eval_loss", "eval_acc", "eval_pure", "eval_reg_rmse",
    ]}

    for epoch in range(epochs):
        running = {"loss": 0.0, "acc": 0.0, "pure": 0.0, "rmse": 0.0}
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, verbose=True)

        for data, _, hit_times, photon_target, photon_list in progress:
            data, photon_target = data.to(device), photon_target.to(device)
            class_output, reg_output = model(data, mode="bce")
            class_output, reg_output = _match_length(class_output, reg_output, data.shape[-1])

            loss, *_ = mined_bce_loss(data, hit_times, photon_list, class_output, reg_output, epoch, device)
            acc = metrics.overall_class_acc(hit_times, class_output, device)
            purity = metrics.overall_class_purity(hit_times, class_output, device)
            rmse = metrics.regression_rmse(hit_times, photon_target, reg_output, class_output, device, mse=mse)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running["loss"] += loss.item()
            running["acc"] += acc
            running["pure"] += purity
            running["rmse"] += rmse
            progress.set_postfix({"loss": running["loss"], "acc": running["acc"]})

        n = max(1, len(train_loader))
        train_loss, train_acc = running["loss"] / n, running["acc"] / n
        train_pure, train_rmse = running["pure"] / n, running["rmse"] / n
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_pure"].append(train_pure)
        results["train_reg_rmse"].append(train_rmse)

        if logger is not None:
            logger.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_pure": train_pure,
                "train_reg_rmse": train_rmse,
                "grad_norm": metrics._compute_grad_norm(model.parameters()),
            })

        if val_loader is not None and len(val_loader) > 0 and (epoch + 1) % val_every == 0:
            val_loss, val_acc, val_pure, val_rmse = validate(model, val_loader, device, mode, mse)
            results["eval_loss"].append(val_loss)
            results["eval_acc"].append(val_acc)
            results["eval_pure"].append(val_pure)
            results["eval_reg_rmse"].append(val_rmse)
            if logger is not None:
                logger.log({
                    "epoch": epoch, "eval_loss": val_loss, "eval_acc": val_acc,
                    "eval_pure": val_pure, "eval_reg_rmse": val_rmse,
                })
            scheduler.step(val_loss)

        if checkpoint_dir and (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, os.path.join(checkpoint_dir, f"{epoch}.pth"))

    return results
