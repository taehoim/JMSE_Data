import numpy as np

from jmse.uq.run import _aleatoric_std, _total_std


def test_aleatoric_std_per_horizon_rmse():
    # residuals with known per-horizon spread
    y = np.zeros((1000, 3))
    rng = np.random.default_rng(0)
    mean = np.column_stack([
        0.5 * rng.standard_normal(1000),
        1.0 * rng.standard_normal(1000),
        2.0 * rng.standard_normal(1000),
    ])
    aleat = _aleatoric_std(mean, y)
    assert aleat.shape == (3,)
    assert np.allclose(aleat, [0.5, 1.0, 2.0], atol=0.05)


def test_total_std_quadrature():
    epi = np.full((4, 3), 3.0)
    aleat = np.array([4.0, 0.0, 4.0])
    total = _total_std(epi, aleat)
    assert total.shape == (4, 3)
    assert np.allclose(total[:, 0], 5.0)               # sqrt(9+16)
    assert np.allclose(total[:, 1], 3.0)               # aleatoric 0 -> epistemic only
