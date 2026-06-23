"""Quantile-regression uncertainty: predict a set of quantiles via the pinball loss.

A single LSTM backbone (the reference architecture) emits horizon x n_quantiles outputs;
training minimizes the pinball (quantile) loss so the heads learn the conditional
quantiles directly. The 0.05/0.5/0.95 set gives a 90% prediction interval plus a median
point forecast. Quantile crossing is repaired post-hoc by sorting along the quantile axis.
"""
import numpy as np
import torch
from torch import nn

from jmse.models.base import BaseForecaster

DEFAULT_TAUS = (0.05, 0.5, 0.95)


class QuantileForecaster(BaseForecaster):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float,
                 horizon: int, taus=DEFAULT_TAUS):
        super().__init__()
        self.horizon = horizon
        self.taus = tuple(taus)
        self.n_q = len(self.taus)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, horizon * self.n_q),
        )

    def forward(self, x):                             # x: (B, L, F) -> (B, horizon, n_q)
        out, _ = self.lstm(x)
        flat = self.head(out[:, -1, :])
        return flat.view(-1, self.horizon, self.n_q)

    @torch.no_grad()
    def predict_quantiles(self, X, device="cpu", batch_size=1024, target_scaler=None):
        """Return monotonic quantile predictions (N, horizon, n_q), inverse-scaled.

        target_scaler is applied per quantile slice (StandardScaler is monotone, so the
        quantile ordering is preserved); quantiles are sorted to remove any crossing.
        """
        self.eval()
        self.to(device)
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        outs = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            outs.append(self(xb).cpu().numpy())
        q = np.concatenate(outs, axis=0) if outs else np.empty((0, self.horizon, self.n_q))
        if target_scaler is not None:
            for k in range(self.n_q):                 # inverse-scale each quantile slice
                q[:, :, k] = target_scaler.inverse_transform(q[:, :, k])
        return np.sort(q, axis=-1)                     # enforce monotonicity


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, taus) -> torch.Tensor:
    """Mean pinball (quantile) loss.

    pred:   (B, horizon, n_q)
    target: (B, horizon)
    taus:   iterable of length n_q.
    """
    taus_t = torch.as_tensor(taus, dtype=pred.dtype, device=pred.device)
    err = target.unsqueeze(-1) - pred                 # (B, horizon, n_q)
    loss = torch.maximum(taus_t * err, (taus_t - 1.0) * err)
    return loss.mean()


def sort_quantiles(q):
    """Sort quantile predictions ascending along the last axis (fixes crossing)."""
    if isinstance(q, torch.Tensor):
        return torch.sort(q, dim=-1).values
    return np.sort(np.asarray(q), axis=-1)
