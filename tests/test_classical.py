import numpy as np

from jmse.models.classical import ARForecaster, KalmanForecaster, build_classical, is_classical


def test_registry():
    assert is_classical("ar") and is_classical("kalman")
    assert not is_classical("lstm")
    assert isinstance(build_classical("ar", order=4), ARForecaster)
    assert isinstance(build_classical("kalman"), KalmanForecaster)


def test_ar_predict_shape():
    rng = np.random.default_rng(0)
    yhist = rng.standard_normal((50, 20))
    m = ARForecaster(order=5).fit(yhist)
    out = m.predict(yhist, horizon=5)
    assert out.shape == (50, 5)
    assert m.coef_.shape == (5,)


def test_ar_recovers_known_process():
    # x_t = 0.6 x_{t-1} (AR(1)); recursive forecast must decay geometrically.
    n, L = 200, 40
    a = 0.6
    rng = np.random.default_rng(1)
    series = np.zeros((n, L))
    series[:, 0] = rng.standard_normal(n)
    for t in range(1, L):
        series[:, t] = a * series[:, t - 1] + 0.01 * rng.standard_normal(n)
    m = ARForecaster(order=1).fit(series)
    assert abs(m.coef_[0] - a) < 0.05
    assert abs(m.intercept_) < 0.05
    out = m.predict(series, horizon=3)
    # one-step from last value ~ a*x_t; recursion feeds the prediction back exactly
    last = series[:, -1]
    assert np.allclose(out[:, 0], a * last, atol=0.05 * np.abs(last).mean() + 0.02)
    assert np.allclose(out[:, 1], m.intercept_ + m.coef_[0] * out[:, 0], atol=1e-9)


def test_kalman_predict_shape_and_flat():
    rng = np.random.default_rng(2)
    yhist = rng.standard_normal((30, 20))
    m = KalmanForecaster().fit(yhist)
    out = m.predict(yhist, horizon=5)
    assert out.shape == (30, 5)
    # local-level forecast is flat across the horizon
    assert np.allclose(out, out[:, [0]])
    assert m.R > 0 and m.q > 0


def test_kalman_denoises_constant_plus_noise():
    # constant level + observation noise: filtered level should beat the raw last obs
    rng = np.random.default_rng(3)
    true = 0.4
    yhist = true + 0.3 * rng.standard_normal((500, 25))
    m = KalmanForecaster().fit(yhist)
    pred = m.predict(yhist, horizon=1)[:, 0]
    last = yhist[:, -1]
    assert np.mean((pred - true) ** 2) < np.mean((last - true) ** 2)
