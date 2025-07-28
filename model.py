import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, depth=4):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Encoder
        channels = in_channels
        for d in range(depth):
            self.encoders.append(nn.Sequential(
                ResidualBlock1D(channels, base_channels * 2**d),
                ResidualBlock1D(base_channels * 2**d, base_channels * 2**d)
            ))
            self.pools.append(nn.MaxPool1d(2))
            channels = base_channels * 2**d

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(channels, channels * 2),
            ResidualBlock1D(channels * 2, channels * 2)
        )

        # Decoder
        for d in reversed(range(depth)):
            self.upsamples.append(nn.ConvTranspose1d(channels * 2, channels, 2, stride=2))
            self.decoders.append(nn.Sequential(
                ResidualBlock1D(channels * 2, channels),
                ResidualBlock1D(channels, channels)
            ))
            channels = channels // 2

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

        # exclude sigmoid for more stability when doing binary x-tent loss
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.final_conv(x) # excluding sigmoid makes model output raw logits, shape (batch_size, 16000, 1)


################ Lightweight Transformer Model for Same Purpose ################

# Pipeline for an example input (array of length 16000)
# 1. Tokenize into 16000 tokens
# 2. Apply positional encoding to each token, embed each token, and then combine these embeddings into a single tensor
# 3. Apply a transformer encoder to the combined tensor
# 4. Apply a transformer decoder to the combined tensor
# 5. Apply a linear layer to the output of the decoder to classify each original time bin as flash or not flash
# 6. Output the classification for each time bin

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
        

class TransformerModel(nn.Module):
    def __init__(self, in_channels=1, d_model=48, num_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # tokenize 500 bins as 1 token:
        self.tokenizer = nn.Unfold(kernel_size=(500, 1), stride=(500, 1))
        self.input_embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=16000)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), 
            num_layers=num_layers
        )
        
        # predict flash probability for each time bin
        self.output_layer = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 16000) from data loader

        batch_size, in_channels, seq_len = x.shape  # (B, 1, 16000)

        # tokenize 500 bins as 1 token:
        # START HERE: FIX TOKENIZATION
        # x = self.tokenizer(x)

        # Remove channel dimension for transformer (treat as (B, seq_len, 1))
        x = x.permute(0, 2, 1)  # (B, 16000, 1)

        # Embed each time bin: (B, 16000, 1) -> (B, 16000, d_model)
        x = self.input_embedding(x)

        # Add positional encoding: (B, 16000, d_model)
        pe = self.positional_encoding.pe[:seq_len, :].to(x.device)  # (16000, d_model)
        x = x + pe.unsqueeze(0)  # (B, 16000, d_model)

        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)  # (16000, B, d_model)

        # Encoder
        x = self.encoder(x)  # (16000, B, d_model)

        # Back to (B, 16000, d_model) for output layer
        x = x.transpose(0, 1)  # (B, 16000, d_model)

        # Predict flash probability for each time bin
        output = self.output_layer(x)  # (B, 16000, 1)

        # To match UNet1D output: (B, 1, 16000)
        output = output.permute(0, 2, 1)  # (B, 1, 16000)
        return output