import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math

from transformer import *
from torchaudio.models import Conformer

class MultiLevelTokenizer(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_sizes=[20, 50, 100, 400], pool_stride=100, window_len=8000, token_size=100, downsample='conv_mlp_lite'):
        """
        Args:
            in_channels: input feature channels (e.g., 1 for waveform)
            hidden_dim: embedding size per conv branch
            kernel_sizes: list of temporal kernel sizes (in samples)
            stride: how many samples = 100 ns (controls downsampling rate)
        """
        super().__init__()
        num_tokens = window_len // token_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=hidden_dim, 
                      kernel_size=k, 
                      stride=1, 
                      padding=k//2)
            for k in kernel_sizes
        ])

        if downsample == 'pool':
            self.downsample = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        elif downsample == 'linear_mlp':
            self.downsample = nn.Sequential(
                nn.Linear(window_len, num_tokens*5),
                nn.GELU(),
                nn.Linear(num_tokens*5, num_tokens),
            )
        elif downsample == 'single_conv':
            self.downsample = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=pool_stride,
                stride=pool_stride,
                padding=0 # want exactly window_len // token_size tokens
            )
        # 2-Layer Conv MLP
        elif downsample == 'conv_mlp_lite':
            self.downsample = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
                nn.GELU(),
                # This layer has HUGE PARAMS (6M)
                # nn.Conv1d(hidden_dim, hidden_dim, kernel_size=pool_stride, stride=pool_stride, padding=0)
                # Replacement depthwise + pointwise Conv
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=token_size, stride=token_size, padding=0, groups=hidden_dim, bias=False),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            )
        
        self.proj = nn.Conv1d(
            in_channels=len(kernel_sizes) * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )

    def forward(self, x):
        """
        x: (B, in_channels, L)
        """
        conv_outs = []
        for conv in self.convs:
            feat = F.relu(conv(x))    # (B, d_model, 8000)
            feat = self.downsample(feat)   # [B, 80, d_model]
            conv_outs.append(feat)

        out = torch.cat(conv_outs, dim=1)  # (B, d_model * n_kernels, 80)
        out = self.proj(out)

        return out

class ConformerModel(nn.Module):
    def __init__(self, in_channels=1, d_model=48, num_heads=4, num_layers=2, token_size=100, window_len = 8000, tokens='multi-level', kernel_sizes=[20, 50, 100, 400], mlp=False, ffn_factor=8, downsample='conv_mlp_lite', dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.window_len = window_len
        self.token_size = token_size
        self.tokens = tokens
        self.kernel_sizes = kernel_sizes
        self.mlp = mlp

        # Embed into tokens
        if self.tokens == 'multi-level':
            self.tokenizer = MultiLevelTokenizer(
                in_channels=1,
                hidden_dim=d_model,
                kernel_sizes=kernel_sizes,
                window_len=window_len,
                token_size=token_size,
                downsample=downsample
            )
        else:
            self.tokenizer = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=100, stride=100)

        # Compute sinusoidal positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)

        # kernel_size roughly scaled to data's sharpest features (~20 ns for initial peak)
        self.conformer = Conformer(input_dim=d_model, num_heads=num_heads, ffn_dim=(ffn_factor*d_model), num_layers=num_layers, depthwise_conv_kernel_size=21, dropout=dropout) #[B, L, d_model]

        # Upsampling back to original bin resolution [batch_size, 48, 8000] w/ interpolation after this layer
        if mlp:
            num_tokens = window_len // token_size
            # 2-Layer Linear MLP
            # self.upsample = nn.Sequential(
            #     nn.Linear(num_tokens, num_tokens*5),   # [B, d_model, 80] → [B, d_model, 8000]
            #     nn.ReLU(),
            #     nn.Linear(num_tokens*5, window_len)  # optional refinement
            # )
            # 2-Layer Conv MLP
            self.upsample = nn.Sequential(
                # ANOTHER HUGE LAYER (6M PARAMS)
                # nn.ConvTranspose1d(d_model, d_model, kernel_size=token_size, stride=token_size),  # upsample
                # Lightweight replacement
                nn.ConvTranspose1d(d_model, d_model, kernel_size=token_size, stride=token_size, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),  # pointwise mixing
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=2)     # refine
            )
        else:
            self.upsample = nn.Linear(d_model, d_model)

        # Classification Head (2-layer MLP)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

        # Regression Head (2-layer MLP)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x, mode='mined_bce'):
        B, in_channels, window_len = x.shape

        x = self.tokenizer(x)          # [B, d_model, 80]
        L_tokens = x.size(-1)
        
        x = x.permute(2, 0, 1)         # [80, B, d_model]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)         #[B, 80, d_model]

        lengths = torch.full((B,), window_len // self.token_size, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)          # [B, 80, d_model]

        if self.mlp:
            x = x.transpose(1, 2)   # [B, d_model, 80]
        
        x = self.upsample(x)        # [B, d_model, 8000]

        if not self.mlp:
            x = x.permute(0, 2, 1)         # [B, d_model, 80]
            x = F.interpolate(x, size=8000, mode="linear", align_corners=False)

        # Class & Reg Heads
        class_logits = self.class_l2(F.relu(self.class_l1(x)))
        reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))

        return class_logits, reg_logits

class ConformerModelv2(nn.Module):
    def __init__(self, in_channels=1, d_model=256, num_heads=8, num_layers=4, token_size=100, window_len = 8000, kernel_sizes=[20, 50, 100, 400], ffn_factor=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.token_size=token_size
        self.window_len = window_len
        self.kernel_sizes = kernel_sizes
        self.ffn_factor = ffn_factor

        self.tokenizer = MultiLevelTokenizer(
            in_channels=1,
            hidden_dim=d_model,
            kernel_sizes=kernel_sizes,
            window_len=window_len,
            token_size=token_size
        )

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)
        self.conformer = Conformer(input_dim=d_model, num_heads=num_heads, ffn_dim=(ffn_factor*d_model), num_layers=num_layers, depthwise_conv_kernel_size=21, dropout=dropout) #[B, L, d_model]

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=100, mode="linear"),  # non-learnable
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=2)
        )

         # Classification Head (2-layer MLP)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

        # Regression Head (2-layer MLP)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x, mode='mined_bce'):
        B, in_channels, window_len = x.shape

        x = self.tokenizer(x)                      # [B, d_model, 80]
        L_tokens = x.size(-1)

        x = x.permute(2, 0, 1)                     # [80, B, d_model]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)                     #[B, 80, d_model]

        lengths = torch.full((B,), window_len // self.token_size, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)          # [B, 80, d_model]

        x = x.permute(0, 2, 1)
        x = self.upsample(x)                       # [B, d_model, 8000]

        # Class & Reg Heads
        class_logits = self.class_l2(F.relu(self.class_l1(x)))
        reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))

        return class_logits, reg_logits

        

        
        

        