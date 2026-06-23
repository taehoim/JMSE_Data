import numpy as np

from jmse.models.feature_baselines import GBMWindowForecaster, RidgeWindowForecaster


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 4, 3))
    w = rng.standard_normal((12, 2))
    y = X.reshape(600, -1) @ w                      # exact linear target, 2 horizons
    pred = RidgeWindowForecaster(alpha=1e-8).fit(X, y).predict(X)
    assert pred.shape == (600, 2)
    assert np.corrcoef(pred.ravel(), y.ravel())[0, 1] > 0.99


def test_gbm_shapes_and_learns_signal():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5, 2))
    y = np.column_stack([X[:, -1, 0], X[:, -1, 0] + X[:, -2, 1]])   # (400, 2)
    pred = GBMWindowForecaster(seed=0, max_iter=30).fit(X, y).predict(X)
    assert pred.shape == (400, 2)
    # in-sample it should track the signal far better than predicting the mean
    assert np.mean((pred - y) ** 2) < 0.5 * np.mean((y - y.mean(0)) ** 2)
