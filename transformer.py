import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math

# Standard sinusoidal positional encoding
# Data is [batch_size, channels, window_len]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # won’t be trained

    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].unsqueeze(1).to(x.device)  # [seq_len, 1, d_model]

class TransformerModel(nn.Module):
    def __init__(self, in_channels=1, d_model=128, num_heads=8, num_layers=4, token_size=100, window_len = 8000):
        super().__init__()

        self.d_model = d_model
        self.window_len = window_len

        # Embed 100 non-overlapping ns bins into tokens
        self.tokenizer = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=100, stride=100)

        # Compute sinusoidal positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), 
            num_layers=num_layers
        ) # [batch_size, 48, 800]

        # Upsampling back to original bin resolution [batch_size, 48, 8000] w/ interpolation after this layer
        self.upsample = nn.Linear(d_model, d_model)

        # Classification Head (2-layer MLP)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

        # Regression Head (2-layer MLP)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x, mode='mined_bce'):
        batch_size, in_channels, window_len = x.shape

        x = self.tokenizer(x)          # [B, d_model, 80]
        x = x.permute(2, 0, 1)         # [80, B, d_model]
        x = self.positional_encoding(x)
        x = self.encoder(x)            # [80, B, d_model]

        x = x.permute(1, 0, 2)         # [B, 80, d_model]
        x = self.upsample(x)           # [B, 80, d_model]

        x = x.permute(0, 2, 1)         # [B, d_model, 80]
        x = F.interpolate(x, size=8000, mode="linear", align_corners=False)

        # Class & Reg Heads
        class_logits = self.class_l2(F.relu(self.class_l1(x)))
        reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))

        return class_logits, reg_logits

        
        
        