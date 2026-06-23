"""Model registry: maps config `model` name -> constructor(n_features, ..., horizon).

Holds the gradient-trained (nn.Module) forecasters benchmarked in P1. Classical
baselines (AR / Kalman) have their own fit/predict path and are not registered here.
"""
from jmse.models.rnn import GRUForecaster, LSTMForecaster
from jmse.models.tcn import TCNForecaster
from jmse.models.transformer import TransformerForecaster

_REGISTRY = {
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "tcn": TCNForecaster,
    "transformer": TransformerForecaster,
}


def build_model(name: str, n_features: int, horizon: int, *, hidden: int, layers: int, dropout: float):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](n_features, hidden, layers, dropout, horizon)


def available_models():
    return sorted(_REGISTRY)
