import numpy as np
import torch

from jmse.uq.quantile import (
    DEFAULT_TAUS, QuantileForecaster, pinball_loss, sort_quantiles,
)


def test_forward_shape():
    taus = (0.05, 0.5, 0.95)
    m = QuantileForecaster(n_features=8, hidden=32, layers=1, dropout=0.0, horizon=5, taus=taus)
    out = m(torch.randn(16, 20, 8))
    assert out.shape == (16, 5, 3)                    # (B, horizon, n_quantiles)


def test_pinball_minimized_at_empirical_quantile():
    rng = np.random.default_rng(0)
    y = torch.tensor(rng.standard_normal(20_000), dtype=torch.float32).reshape(-1, 1)
    tau = 0.7
    q_star = float(np.quantile(y.numpy(), tau))
    taus = (tau,)

    def loss_at(q):
        pred = torch.full((y.shape[0], 1, 1), q)
        return pinball_loss(pred, y, taus).item()

    base = loss_at(q_star)
    assert base <= loss_at(q_star + 0.1) and base <= loss_at(q_star - 0.1)


def test_pinball_loss_value_known_case():
    # single point, tau=0.5: pinball = 0.5*|err|; err = y - pred = 2 -> 1.0
    pred = torch.zeros(1, 1, 1)
    target = torch.full((1, 1), 2.0)
    assert abs(pinball_loss(pred, target, (0.5,)).item() - 1.0) < 1e-6


def test_sort_quantiles_fixes_crossing():
    # quantiles out of order along the last axis must come out ascending
    q = torch.tensor([[[0.9, 0.1, 0.5]]])             # (1,1,3)
    s = sort_quantiles(q)
    assert torch.allclose(s, torch.tensor([[[0.1, 0.5, 0.9]]]))


def test_default_taus():
    assert DEFAULT_TAUS == (0.05, 0.5, 0.95)


def test_overfit_tiny_batch_pinball():
    torch.manual_seed(0)
    taus = (0.05, 0.5, 0.95)
    m = QuantileForecaster(8, 32, 1, 0.0, 5, taus=taus)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    x = torch.randn(8, 20, 8)
    y = torch.randn(8, 5)
    loss = None
    for _ in range(400):
        opt.zero_grad()
        loss = pinball_loss(m(x), y, taus)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05                          # can fit the tiny batch
