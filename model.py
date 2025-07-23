import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
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

        # BINARY classification of each bin in the output: start of flash or not start of flash
        self.sigmoid = nn.Sigmoid()

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
        return self.sigmoid(self.final_conv(x))
        