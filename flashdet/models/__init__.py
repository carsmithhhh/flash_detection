"""Model architectures and a name-based registry.

``build_model("conformer", d_model=256, ...)`` lets the training/eval scripts
instantiate a model purely from its YAML config (a string name + an ``args`` dict).
"""

from .conformer import ConformerModel, ConformerModelv2
from .layers import MultiLevelTokenizer, PositionalEncoding, ResidualBlock1D
from .transformer import TransformerModel
from .unet import UNet1D

MODEL_REGISTRY = {
    "unet": UNet1D,
    "transformer": TransformerModel,
    "conformer": ConformerModel,
    "conformer_v2": ConformerModelv2,
}


def build_model(name, **kwargs):
    """Instantiate a registered model by name with constructor ``kwargs``."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "UNet1D",
    "TransformerModel",
    "ConformerModel",
    "ConformerModelv2",
    "ResidualBlock1D",
    "PositionalEncoding",
    "MultiLevelTokenizer",
]
