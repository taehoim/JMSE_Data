"""Lightweight Transformer-encoder forecaster.

Projects the F input channels to `hidden`, adds sinusoidal positional encoding,
runs `layers` TransformerEncoder blocks, and regresses the horizon from the last
timestep's representation. Kept small (few heads/layers) to suit the ~5k-step files.
"""
import math

import torch
from torch import nn

from jmse.models.base import BaseForecaster


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x):                               # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(BaseForecaster):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float,
                 horizon: int, nhead: int = 4):
        super().__init__()
        # ensure hidden is divisible by nhead
        while hidden % nhead != 0 and nhead > 1:
            nhead -= 1
        self.input_proj = nn.Linear(n_features, hidden)
        self.pos = _PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(layers, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, horizon),
        )

    def forward(self, x):                               # x: (B, L, F) -> (B, horizon)
        h = self.pos(self.input_proj(x))
        h = self.encoder(h)
        return self.head(h[:, -1, :])                   # last timestep
