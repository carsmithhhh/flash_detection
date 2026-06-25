"""1D U-Net for per-bin flash detection + photon regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualBlock1D


class UNet1D(nn.Module):
    """Symmetric 1D U-Net with residual blocks and two output heads.

    A standard encoder/bottleneck/decoder with skip connections operates at full
    waveform resolution. Two 2-layer 1x1-conv heads share the decoder features:
    a classification head (per-bin flash logits) and a regression head (per-bin
    photon count). Both outputs have shape ``[B, 1, L]``.

    Args:
        in_channels: input channels (1 for a raw waveform).
        base_channels: channel count of the first encoder stage; doubles each level.
        depth: number of down/up-sampling levels.
    """

    def __init__(self, in_channels=1, base_channels=32, depth=4):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Encoder: two residual blocks per level, then halve the resolution.
        channels = in_channels
        for d in range(depth):
            self.encoders.append(
                nn.Sequential(
                    ResidualBlock1D(channels, base_channels * 2 ** d),
                    ResidualBlock1D(base_channels * 2 ** d, base_channels * 2 ** d),
                )
            )
            self.pools.append(nn.MaxPool1d(2))
            channels = base_channels * 2 ** d

        self.bottleneck = nn.Sequential(
            ResidualBlock1D(channels, channels * 2),
            ResidualBlock1D(channels * 2, channels * 2),
        )

        # Decoder: upsample, concatenate the matching skip, then two residual blocks.
        for d in reversed(range(depth)):
            self.upsamples.append(nn.ConvTranspose1d(channels * 2, channels, 2, stride=2))
            self.decoders.append(
                nn.Sequential(
                    ResidualBlock1D(channels * 2, channels),
                    ResidualBlock1D(channels, channels),
                )
            )
            channels = channels // 2

        # Classification head (2-layer 1x1-conv MLP) -> raw logits for BCEWithLogitsLoss.
        self.class_l1 = nn.Conv1d(base_channels, base_channels // 2, 1)
        self.class_l2 = nn.Conv1d(base_channels // 2, 1, 1)

        # Regression head (2-layer 1x1-conv MLP) -> per-bin photon count.
        self.reg_l1 = nn.Conv1d(base_channels, base_channels // 2, 1)
        self.reg_l2 = nn.Conv1d(base_channels // 2, 1, 1)

    def forward(self, x, mode="bce"):
        """``x``: ``[B, 1, L]`` -> ``(class_logits, photon_reg)``, each ``[B, 1, L]``.

        ``mode`` is accepted for a common interface with the other models; the U-Net
        always returns raw logits plus the regression map.
        """
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # Crop/pad to handle waveform lengths that are not powers of two.
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                skip = skip[..., : x.shape[-1]] if diff > 0 else F.pad(skip, (0, -diff))
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        class_logits = self.class_l2(F.relu(self.class_l1(x)))  # raw logits
        photon_reg = self.reg_l2(F.relu(self.reg_l1(x)))
        return class_logits, photon_reg
