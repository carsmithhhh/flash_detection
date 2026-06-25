"""Conformer models for flash detection + regression.

Two variants:
    ``ConformerModel``   - flexible tokenizer/upsampler, the workhorse architecture.
    ``ConformerModelv2`` - fixed multi-level tokenizer with an Upsample-based decoder.

Both use the from-scratch :class:`~flashdet.models.conformer_block.ConformerEncoder`
(no ``torchaudio`` dependency) and return ``(class_logits, reg_logits)``.
"""

import torch.nn as nn
import torch.nn.functional as F

from .conformer_block import ConformerEncoder
from .layers import MultiLevelTokenizer, PositionalEncoding


class ConformerModel(nn.Module):
    """Tokenizer -> Conformer encoder -> upsampler -> class & reg heads.

    Args:
        in_channels: input channels (1 for a raw waveform).
        d_model: feature / attention dimension.
        num_heads: attention heads per conformer block.
        num_layers: number of conformer blocks.
        token_size: samples per token.
        window_len: waveform length the model upsamples back to.
        tokens: ``'multi-level'`` uses :class:`MultiLevelTokenizer`; anything else
            falls back to a single strided conv tokenizer.
        kernel_sizes: temporal kernels for the multi-level tokenizer.
        mlp: if True, decode tokens with a learnable conv-transpose MLP; otherwise
            project per-token and linearly interpolate to ``window_len``.
        ffn_factor: conformer feed-forward width = ``ffn_factor * d_model``.
        conv_kernel_size: depthwise conv kernel in each conformer block (odd; the
            default 21 ~ the ~20 ns rising edge, the sharpest waveform feature).
        downsample: tokenizer downsampling mode (see :class:`MultiLevelTokenizer`).
        dropout: conformer dropout probability.

    Output: ``(class_logits, reg_logits)``, each ``[B, 1, window_len]``.
    """

    def __init__(
        self,
        in_channels=1,
        d_model=48,
        num_heads=4,
        num_layers=2,
        token_size=100,
        window_len=8000,
        tokens="multi-level",
        kernel_sizes=(20, 50, 100, 400),
        mlp=False,
        ffn_factor=8,
        conv_kernel_size=21,
        downsample="conv_mlp_lite",
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_len = window_len
        self.token_size = token_size
        self.tokens = tokens
        self.mlp = mlp

        if tokens == "multi-level":
            self.tokenizer = MultiLevelTokenizer(
                in_channels=1,
                hidden_dim=d_model,
                kernel_sizes=kernel_sizes,
                window_len=window_len,
                token_size=token_size,
                downsample=downsample,
            )
        else:
            self.tokenizer = nn.Conv1d(
                in_channels=1, out_channels=d_model, kernel_size=token_size, stride=token_size
            )

        # Absolute encoding on the tokens; the encoder adds relative encoding internally.
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)

        self.conformer = ConformerEncoder(
            embed_dim=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            conv_kernel_size=conv_kernel_size,
            ffn_factor=ffn_factor,
            dropout=dropout,
        )

        if mlp:
            # Learnable upsampler: grouped conv-transpose (cheap) + pointwise mixing + refine.
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(d_model, d_model, kernel_size=token_size, stride=token_size, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=2),
            )
        else:
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
        x = x.permute(2, 0, 1)             # [T, B, d_model]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)             # [B, T, d_model]

        x = self.conformer(x)               # [B, T, d_model]

        if self.mlp:
            x = x.transpose(1, 2)          # [B, d_model, T]
            x = self.upsample(x)            # [B, d_model, window_len]
        else:
            x = self.upsample(x)            # [B, T, d_model]
            x = x.permute(0, 2, 1)         # [B, d_model, T]
            x = F.interpolate(x, size=self.window_len, mode="linear", align_corners=False)

        class_logits = self.class_l2(F.relu(self.class_l1(x)))
        reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))
        return class_logits, reg_logits


class ConformerModelv2(nn.Module):
    """Conformer variant with a fixed multi-level tokenizer and Upsample decoder.

    Differs from :class:`ConformerModel` by always using the multi-level tokenizer
    (``conv_mlp_lite`` downsampling) and decoding with a non-learnable ``nn.Upsample``
    followed by conv refinement. Output: ``(class_logits, reg_logits)``, ``[B, 1, window_len]``.
    """

    def __init__(
        self,
        in_channels=1,
        d_model=256,
        num_heads=8,
        num_layers=4,
        token_size=100,
        window_len=8000,
        kernel_sizes=(20, 50, 100, 400),
        ffn_factor=4,
        conv_kernel_size=21,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_size = token_size
        self.window_len = window_len

        self.tokenizer = MultiLevelTokenizer(
            in_channels=1,
            hidden_dim=d_model,
            kernel_sizes=kernel_sizes,
            window_len=window_len,
            token_size=token_size,
        )
        # Absolute encoding on the tokens; the encoder adds relative encoding internally.
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)
        self.conformer = ConformerEncoder(
            embed_dim=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            conv_kernel_size=conv_kernel_size,
            ffn_factor=ffn_factor,
            dropout=dropout,
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=token_size, mode="linear"),  # non-learnable
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=2),
        )

        # Classification head (2-layer MLP)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

        # Regression head (2-layer MLP)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x, mode="mined_bce"):
        """``x``: ``[B, 1, window_len]`` -> ``(class_logits, reg_logits)``."""
        x = self.tokenizer(x)               # [B, d_model, T]
        x = x.permute(2, 0, 1)             # [T, B, d_model]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)             # [B, T, d_model]

        x = self.conformer(x)               # [B, T, d_model]

        x = x.permute(0, 2, 1)             # [B, d_model, T]
        x = self.upsample(x)                # [B, d_model, window_len]

        class_logits = self.class_l2(F.relu(self.class_l1(x)))
        reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))
        return class_logits, reg_logits
