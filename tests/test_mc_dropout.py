import numpy as np
import torch

from jmse.models.rnn import LSTMForecaster
from jmse.uq.mc_dropout import enable_dropout, mc_dropout_predict


def test_mc_dropout_produces_variance_when_dropout_on():
    torch.manual_seed(0)
    m = LSTMForecaster(n_features=8, hidden=32, layers=1, dropout=0.5, horizon=5)
    X = np.float32(np.random.default_rng(0).standard_normal((64, 20, 8)))
    mean, std = mc_dropout_predict(m, X, T=30, device="cpu")
    assert mean.shape == (64, 5) and std.shape == (64, 5)
    assert np.all(std >= 0)
    assert std.mean() > 1e-4                          # head dropout injects spread


def test_mc_dropout_zero_dropout_is_deterministic():
    torch.manual_seed(0)
    m = LSTMForecaster(8, 32, 1, dropout=0.0, horizon=5)
    X = np.float32(np.random.default_rng(1).standard_normal((32, 20, 8)))
    mean, std = mc_dropout_predict(m, X, T=20, device="cpu")
    assert std.max() < 1e-6                            # no dropout -> no variance


def test_mc_dropout_restores_eval_mode():
    m = LSTMForecaster(8, 16, 1, 0.3, 5).eval()
    assert not m.training
    mc_dropout_predict(m, np.float32(np.zeros((4, 20, 8))), T=3)
    assert not m.training                              # eval mode restored after MC passes


def test_enable_dropout_only_toggles_dropout_layers():
    m = LSTMForecaster(8, 16, 1, 0.3, 5).eval()
    enable_dropout(m)
    dropouts = [mod.training for mod in m.modules() if isinstance(mod, torch.nn.Dropout)]
    assert dropouts and all(dropouts)                  # every Dropout is now in train mode
