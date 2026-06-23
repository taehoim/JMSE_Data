"""Temporal Convolutional Network forecaster (dilated causal convolutions).

A stack of residual blocks with exponentially growing dilation gives a receptive
field that covers the full lookback window without recurrence. `layers` sets the
number of residual blocks (dilation 1, 2, 4, ...); `hidden` is the channel width.
"""
import torch
from torch import nn
from torch.nn import functional as F

from jmse.models.base import BaseForecaster


class _CausalConv1d(nn.Module):
    """Causal Conv1d for short windows, implemented without cuDNN's small-kernel overhead."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel))
        self.bias = nn.Parameter(torch.empty(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_ch * kernel
        bound = fan_in ** -0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        length = x.shape[-1]
        pad = (self.kernel - 1) * self.dilation
        x_pad = F.pad(x, (pad, 0))
        taps = [
            x_pad[:, :, j * self.dilation:j * self.dilation + length]
            for j in range(self.kernel)
        ]
        stacked = torch.stack(taps, dim=-1)  # (B, C, L, K)
        return torch.einsum("bclk,ock->bol", stacked, self.weight) + self.bias[None, :, None]


class _Chomp1d(nn.Module):
    """Trim the right padding so the convolution stays strictly causal."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(BaseForecaster):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float,
                 horizon: int, kernel: int = 3):
        super().__init__()
        blocks = []
        in_ch = n_features
        for i in range(max(layers, 1)):
            blocks.append(_TemporalBlock(in_ch, hidden, kernel, dilation=2 ** i, dropout=dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, horizon),
        )

    def forward(self, x):                       # x: (B, L, F) -> (B, horizon)
        h = self.tcn(x.transpose(1, 2))         # -> (B, hidden, L)
        return self.head(h[:, :, -1])           # last (causal) timestep
