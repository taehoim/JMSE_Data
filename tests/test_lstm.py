import torch

from jmse.models.rnn import GRUForecaster, LSTMForecaster


def test_forward_shape():
    m = LSTMForecaster(n_features=8, hidden=128, layers=2, dropout=0.3, horizon=5)
    out = m(torch.randn(16, 20, 8))
    assert out.shape == (16, 5)


def test_overfit_tiny_batch():
    # the model must be able to drive loss -> ~0 on a tiny fixed batch (it can learn)
    torch.manual_seed(0)
    m = LSTMForecaster(8, 32, 1, 0.0, 5)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    x = torch.randn(8, 20, 8)
    y = torch.randn(8, 5)
    loss = None
    for _ in range(300):
        opt.zero_grad()
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3


def test_gru_forward_shape():
    m = GRUForecaster(n_features=8, hidden=128, layers=2, dropout=0.3, horizon=5)
    out = m(torch.randn(16, 20, 8))
    assert out.shape == (16, 5)


def test_gru_overfit_tiny_batch():
    torch.manual_seed(0)
    m = GRUForecaster(8, 32, 1, 0.0, 5)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    x = torch.randn(8, 20, 8)
    y = torch.randn(8, 5)
    loss = None
    for _ in range(300):
        opt.zero_grad()
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3
