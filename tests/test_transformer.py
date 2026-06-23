import torch

from jmse.models.transformer import TransformerForecaster


def test_forward_shape():
    m = TransformerForecaster(n_features=8, hidden=64, layers=2, dropout=0.1, horizon=5)
    out = m(torch.randn(16, 20, 8))
    assert out.shape == (16, 5)


def test_nhead_divisibility_fallback():
    # hidden not divisible by default nhead=4 must not raise (nhead is reduced)
    m = TransformerForecaster(n_features=6, hidden=30, layers=1, dropout=0.0, horizon=3)
    out = m(torch.randn(4, 10, 6))
    assert out.shape == (4, 3)


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = TransformerForecaster(8, 32, 1, 0.0, 5)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    x = torch.randn(8, 20, 8)
    y = torch.randn(8, 5)
    loss = None
    for _ in range(500):
        opt.zero_grad()
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2
