"""flashdet: flash detection + photon regression on LArTPC optical waveforms.

Public surface:
    models  - UNet1D, TransformerModel, ConformerModel(v2), build_model, MODEL_REGISTRY
    data    - WaveformDataset, make_dataloader, make_train_val_loaders, split_dataset
    losses  - mined_bce_loss, bce_loss
    metrics - per-bin and merged-interval evaluation metrics
    engine  - train, validate, save_checkpoint
    utils   - set_seed, load_config, build_optimizer/scheduler, load_models
"""

from . import data, engine, losses, metrics, utils
from .models import (
    MODEL_REGISTRY,
    ConformerModel,
    ConformerModelv2,
    TransformerModel,
    UNet1D,
    build_model,
)

__all__ = [
    "data",
    "engine",
    "losses",
    "metrics",
    "utils",
    "build_model",
    "MODEL_REGISTRY",
    "UNet1D",
    "TransformerModel",
    "ConformerModel",
    "ConformerModelv2",
]
