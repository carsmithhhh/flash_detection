"""Lightweight transformer-encoder model for flash detection + regression."""

import torch.nn as nn
import torch.nn.functional as F

from .layers import PositionalEncoding


class TransformerModel(nn.Module):
    """Conv tokenizer -> transformer encoder -> upsample -> class & reg heads.

    The waveform is split into non-overlapping ``token_size``-sample tokens by a
    strided conv, encoded with a standard ``nn.TransformerEncoder``, linearly
    interpolated back to ``window_len`` resolution, and read out by two 2-layer
    1x1-conv heads. Returns ``(class_logits, reg_logits)``, each ``[B, 1, window_len]``.

    Args:
        in_channels: input channels (1 for a raw waveform).
        d_model: token embedding / attention dimension.
        num_heads: attention heads per encoder layer.
        num_layers: number of transformer encoder layers.
        token_size: samples per token (tokenizer kernel/stride).
        window_len: waveform length the model upsamples back to.
    """

    def __init__(
        self,
        in_channels=1,
        d_model=128,
        num_heads=8,
        num_layers=4,
        token_size=100,
        window_len=8000,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_len = window_len
        self.token_size = token_size

        self.tokenizer = nn.Conv1d(
            in_channels=1, out_channels=d_model, kernel_size=token_size, stride=token_size
        )
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_layers,
        )
        self.upsample = nn.Linear(d_model, d_model)

        # Classification head (2-layer MLP)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

        # Regression head (2-layer MLP)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x, mode="mined_bce"):
        """``x``: ``[B, 1, window_len]`` -> ``(class_logits, reg_logits)``."""
        x = self.tokenizer(x)               # [B, d_model, T]
        x = x.permute(2, 0, 1)              # [T, B, d_model]
        x = self.positional_encoding(x)
        x = self.encoder(x)                 # [T, B, d_model]

        x = x.permute(1, 0, 2)             # [B, T, d_model]
        x = self.upsample(x)                # [B, T, d_model]

        x = x.permute(0, 2, 1)             # [B, d_model, T]
        x = F.interpolate(x, size=self.window_len, mode="linear", align_corners=False)

        class_logits = self.class_l2(F.relu(self.class_l1(x)))
        reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))
        return class_logits, reg_logits
