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

        # Classification & Regression (photons per bin)
        self.final_conv = nn.Conv1d(base_channels, 1, 1) # [B, 1, 16000] # for yes/no signal
        # self.reg_head = nn.Conv1d(base_channels, 1, 1) # For how many photons
        
        self.regression_head = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x, mode='bce'):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)

            # adding cropping
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                skip = skip[..., :x.shape[-1]] if diff > 0 else F.pad(skip, (0, -diff))

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # For regression task, use regression head
        # Sigmoid is included in BCEWithLogits loss (don't include it as a layer here)
        class_logits = self.final_conv(x)
        # photon_reg = self.reg_head(x)
        if mode == 'bce':
            # return class_logits, photon_reg
            return class_logits
        elif mode == 'regression':
            return self.regression_head(x)

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
    def __init__(self, in_channels=1, d_model=48, num_heads=4, num_layers=2, token_size=100):
        super().__init__()
        self.d_model = d_model
        
        # embed 100 non-overlapping bins into tokens
        self.tokenizer = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=100, stride=100)
        # self.input_embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=1600)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), 
            num_layers=num_layers
        )
        # Yes, you need a layer to go back to the original bin resolution (16000) if you want to predict at the original time resolution.
        # For example, you can use an upsampling layer or a linear projection to expand from token-level (e.g., 160 tokens) back to 16000 bins.
        self.upsample = nn.Linear(int(16000 / token_size), 16000)
        
        # predict flash probability for each time bin
        self.output_layer = nn.Linear(d_model, 1)
    
    def forward(self, x, mode='regression'):
        # x shape: (batch_size, 1, 16000) from data loader

        batch_size, in_channels, seq_len = x.shape  # (B, 1, 16000)

        # tokenize 500 bins as 1 token:
        # START HERE: FIX TOKENIZATION
        x = self.tokenizer(x) # should be shape [B, d_model, 1600]

        # Remove channel dimension for transformer (treat as (B, seq_len, 1))
        x = x.permute(0, 2, 1)  # (B, 16000, d_model)

        # Add positional encoding: (B, 16000, d_model)
        pe = self.positional_encoding.pe[:seq_len, :].to(x.device)  # (1600, d_model)
        
        x = x + pe.unsqueeze(0)  # (B, 16000, d_model)

        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)  # (16000, B, d_model)

        # Encoder
        x = self.encoder(x)  # (16000, B, d_model)

        # Back to (B, 16000, d_model) for output layer
        x = x.transpose(0, 1)  # (B, 16000, d_model)

        # Predict flash probability for each time bin
        x = self.upsample(x)
        output = self.output_layer(x)  # (B, 16000, 1)

        # To match UNet1D output: (B, 1, 16000)
        output = output.permute(0, 2, 1)  # (B, 1, 16000)
        return output



class Transformers2(nn.Module):
    # embedding dimension 48, 2 self-attention layers with 4 heads each - seems reasonable?
    def __init__(self, in_channels=1, d_model=48, num_heads=4, num_layers=2, token_size=10, window_size=1000):
        super().__init__()

        # Hyperparameters
        self.d_model = d_model
        self.token_size = token_size
        self.window_size = window_size
        assert (window_size % token_size == 0)

        # Layers
        # embedding tokens - token_size non-overlapping chunks
        self.tokenizer = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=self.token_size, stride=self.token_size)

        # positional encoding for tokens - using sinusoidal encoding (see above)
        # max_len should be (time window length / token_size)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=int(window_size / token_size))

        # encoder
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

        # ultimately we want per-bin classification, so upsample back to correct resolution
        # self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=100, stride=100)
        self.upsample = nn.Linear(in_features=int(window_size / token_size), out_features=window_size)  # output 100 logits per token
        self.regression_head = nn.Linear(in_features=d_model, out_features=1)
        
    def forward(self, x, mode='regression'):
        # Input x has shape [B, 1, 16000]
        B, _, L = x.shape
        assert (L % self.token_size == 0)

        tokens = self.tokenizer(x) # [B, d_model, 160]
        tokens = tokens.permute(0, 2, 1) # [B, 160, d_model]
        pos_encoding = self.positional_encoding.pe[:L, :].to(x.device)

        x = tokens + pos_encoding.unsqueeze(0)

        x = x.transpose(0, 1) # [160, B, d_model]

        x = self.encoder(x)
        
        x = x.permute(1, 2, 0) #[B, d_model, 1600]
        x = self.upsample(x) # upsample expects shape (*, channels) -> [B, d_model, 16000]

        x = x.permute(0, 2, 1) # [B, 16000, d_model]

        output = self.regression_head(x)

        return output.squeeze(2) # [B, 16000]

