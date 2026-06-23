"""Recurrent forecasters (LSTM, GRU). Both mirror the original paper's MLP head."""
from torch import nn

from jmse.models.base import BaseForecaster


def _mlp_head(hidden: int, dropout: float, horizon: int) -> nn.Sequential:
    """Shared two-layer regression head (last-step hidden state -> horizon targets)."""
    return nn.Sequential(
        nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(hidden, horizon),
    )


class LSTMForecaster(BaseForecaster):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = _mlp_head(hidden, dropout, horizon)

    def forward(self, x):                       # x: (B, L, F) -> (B, horizon)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRUForecaster(BaseForecaster):
    """GRU counterpart to the reference LSTM: same head, fewer gates (cheaper, often competitive)."""

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float, horizon: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = _mlp_head(hidden, dropout, horizon)

    def forward(self, x):                       # x: (B, L, F) -> (B, horizon)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])
