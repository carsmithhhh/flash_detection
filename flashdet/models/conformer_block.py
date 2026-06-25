"""From-scratch Conformer encoder blocks (Gulati et al., 2020, arxiv 2005.08100).

This is a self-contained replacement for ``torchaudio.models.Conformer``. A
:class:`ConformerEncoder` is a stack of :class:`ConformerBlock`s, each a macaron
sandwich:

    x -> x + 1/2 FFN(x) -> x + MHSA(x) -> x + Conv(x) -> x + 1/2 FFN(x) -> LayerNorm

Self-attention uses Transformer-XL style *relative* positional encoding, so the
encoder is position-aware on its own (the relative encoding matrix ``R`` is computed
once per forward and shared across blocks).

Shapes: tokens are ``[B, T, embed_dim]`` throughout, where T = number of tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding matrix ``R`` of shape ``[T, embed_dim]``.

    Positions count down ``[T-1, ..., 0]`` so that, after the relative shift in
    :class:`RelativeMultiHeadAttention`, score column offsets map to relative distances.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: [B, T, embed_dim]; returns R: [T, embed_dim]
        n = x.size(1)
        pos = torch.arange(n - 1, -1, -1, dtype=torch.float, device=x.device)
        dim = torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=x.device)
        div_term = torch.exp(dim * -(math.log(10000.0) / self.embed_dim))
        R = torch.zeros(n, self.embed_dim, device=x.device)
        R[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
        R[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)
        return R


class FeedForwardModule(nn.Module):
    """Pre-norm feed-forward block with Swish activation (added at half weight)."""

    def __init__(self, embed_dim, ffn_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * ffn_factor)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * ffn_factor, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dropout1(self.swish(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


class ConvolutionModule(nn.Module):
    """Pre-norm pointwise-GLU -> depthwise conv -> BatchNorm/Swish -> pointwise conv."""

    def __init__(self, embed_dim, kernel_size=31, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd to preserve sequence length"
        self.ln = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size, padding=kernel_size // 2, groups=embed_dim
        )
        self.bn = nn.BatchNorm1d(embed_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, embed_dim]
        x = self.ln(x)
        x = x.transpose(1, 2)              # [B, embed_dim, T] for Conv1d
        x = self.glu(self.pointwise_conv1(x))
        x = self.swish(self.bn(self.depthwise_conv(x)))
        x = self.dropout(self.pointwise_conv2(x))
        return x.transpose(1, 2)           # back to [B, T, embed_dim]


class RelativeMultiHeadAttention(nn.Module):
    """Transformer-XL style relative multi-head self-attention.

    Scores combine content and position terms with learned global biases ``u``, ``v``::

        A = (Q + u) K^T + rel_shift((Q + v) R^T)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.r_proj = nn.Linear(embed_dim, embed_dim)  # projects relative encoding R
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.u = nn.Parameter(torch.zeros(num_heads, self.head_dim))  # content bias
        self.v = nn.Parameter(torch.zeros(num_heads, self.head_dim))  # position bias
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x):
        """Shift position scores so column j encodes relative distance. [B,H,T,T]->[B,H,T,T]."""
        B, H, T, _ = x.shape
        x = F.pad(x, (1, 0)).view(B, H, T + 1, T)
        return x[:, :, 1:, :]

    def forward(self, x, R, key_padding_mask=None):
        # x: [B, T, embed_dim], R: [T, embed_dim]
        B, T, _ = x.shape

        def heads(t):  # [*, embed_dim] -> [*, H, head_dim] -> head-major
            return t.view(t.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = heads(self.q_proj(x))                       # [B, H, T, head_dim]
        K = heads(self.k_proj(x))
        V = heads(self.v_proj(x))
        R_ = self.r_proj(R).view(-1, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)  # [1, H, T, head_dim]

        content = torch.matmul(Q + self.u.unsqueeze(0).unsqueeze(2), K.transpose(-2, -1))   # [B, H, T, T]
        position = torch.matmul(Q + self.v.unsqueeze(0).unsqueeze(2), R_.transpose(-2, -1))
        attn = (content + self._rel_shift(position)) * self.scale

        if key_padding_mask is not None:  # [B, T], True = ignore
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)


class MultiHeadedSelfAttentionModule(nn.Module):
    """Pre-norm wrapper around :class:`RelativeMultiHeadAttention` with dropout."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention = RelativeMultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, R, key_padding_mask=None):
        x = self.attention(self.layer_norm(x), R, key_padding_mask=key_padding_mask)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """One conformer block: macaron FFN, relative MHSA, conv module, FFN, final norm."""

    def __init__(self, embed_dim, num_heads, kernel_size=31, ffn_factor=4, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(embed_dim, ffn_factor, dropout)
        self.mhsa = MultiHeadedSelfAttentionModule(embed_dim, num_heads, dropout)
        self.conv = ConvolutionModule(embed_dim, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(embed_dim, ffn_factor, dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, R, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, R, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.layer_norm(x)


class ConformerEncoder(nn.Module):
    """Stack of :class:`ConformerBlock`s; a drop-in for ``torchaudio.models.Conformer``.

    Args:
        embed_dim: token/model dimension.
        num_heads: attention heads per block.
        num_layers: number of conformer blocks.
        conv_kernel_size: depthwise conv kernel (odd; ~ the sharpest waveform feature).
        ffn_factor: feed-forward expansion factor.
        dropout: dropout probability used throughout.

    ``forward([B, T, embed_dim]) -> [B, T, embed_dim]``. Unlike torchaudio's Conformer
    it takes no ``lengths`` argument (sequences here are fixed-length and unpadded); an
    optional ``key_padding_mask`` is still accepted for completeness.
    """

    def __init__(self, embed_dim, num_heads, num_layers, conv_kernel_size=31, ffn_factor=4, dropout=0.1):
        super().__init__()
        self.pos_enc = RelativePositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [ConformerBlock(embed_dim, num_heads, conv_kernel_size, ffn_factor, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, key_padding_mask=None):
        R = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, R, key_padding_mask=key_padding_mask)
        return x
