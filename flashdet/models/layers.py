"""Shared building blocks used by more than one model architecture.

Tensor-shape convention throughout the package:
    B = batch size, C = channels, L = waveform length (time bins),
    d_model = transformer/conformer feature dimension, T = number of tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Two 3-tap conv layers with a residual connection (used by ``UNet1D``).

    A 1x1 conv projects the skip path when ``in_channels != out_channels``.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings.

    The table is stored as a (non-trainable) buffer so it moves with the module
    across devices. ``forward`` expects sequence-first input ``[T, B, d_model]``.
    """

    def __init__(self, d_model, max_len=16000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # not trained

    def forward(self, x):
        """Add positional encoding to ``x`` of shape ``[T, B, d_model]``."""
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].unsqueeze(1).to(x.device)  # [T, 1, d_model]


class MultiLevelTokenizer(nn.Module):
    """Tokenize a raw waveform into ``window_len // token_size`` tokens.

    Several parallel 1D convs (one per entry in ``kernel_sizes``) capture features
    at different temporal scales; each branch is downsampled to the token rate, the
    branches are concatenated, then a 1x1 conv projects back to ``hidden_dim``.

    Args:
        in_channels: input feature channels (1 for a raw waveform).
        hidden_dim: embedding size per conv branch (== model ``d_model``).
        kernel_sizes: temporal kernel sizes (in samples) for the parallel branches.
        pool_stride: stride used by the ``'pool'``/``'single_conv'`` downsamplers.
        window_len: waveform length in samples.
        token_size: samples collapsed into one token (token rate = window_len/token_size).
        downsample: one of ``'pool'``, ``'linear_mlp'``, ``'single_conv'``, ``'conv_mlp_lite'``.

    Output: ``[B, hidden_dim, window_len // token_size]``.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        kernel_sizes=(20, 50, 100, 400),
        pool_stride=100,
        window_len=8000,
        token_size=100,
        downsample="conv_mlp_lite",
    ):
        super().__init__()
        num_tokens = window_len // token_size

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )

        if downsample == "pool":
            self.downsample = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        elif downsample == "linear_mlp":
            self.downsample = nn.Sequential(
                nn.Linear(window_len, num_tokens * 5),
                nn.GELU(),
                nn.Linear(num_tokens * 5, num_tokens),
            )
        elif downsample == "single_conv":
            # padding=0 yields exactly window_len // token_size tokens
            self.downsample = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=pool_stride,
                stride=pool_stride,
                padding=0,
            )
        elif downsample == "conv_mlp_lite":
            # Lightweight 2-layer conv MLP. The depthwise + pointwise pair replaces a
            # single dense stride-token_size conv (which alone cost ~6M params).
            self.downsample = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
                nn.GELU(),
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=token_size,
                    stride=token_size,
                    padding=0,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            )
        else:
            raise ValueError(f"Unknown downsample mode: {downsample!r}")

        self.proj = nn.Conv1d(
            in_channels=len(kernel_sizes) * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )

    def forward(self, x):
        """``x``: ``[B, in_channels, window_len]`` -> ``[B, hidden_dim, num_tokens]``."""
        conv_outs = []
        for conv in self.convs:
            feat = F.relu(conv(x))        # [B, hidden_dim, window_len]
            feat = self.downsample(feat)  # [B, hidden_dim, num_tokens]
            conv_outs.append(feat)

        out = torch.cat(conv_outs, dim=1)  # [B, n_kernels * hidden_dim, num_tokens]
        out = self.proj(out)               # [B, hidden_dim, num_tokens]
        return out
