import numpy as np
from scipy.stats import norm

from jmse.earlywarning import alarm


def test_exceedance_labels_strict_gt():
    # M3 guardrail: comparator must be strict '>' (matches dataset_summary)
    y = np.array([[0.10, 0.20], [0.30, 0.30]])
    theta = 0.30
    lab = alarm.exceedance_labels(y, theta)
    assert lab.dtype == bool
    assert lab.tolist() == [[False, False], [False, False]]   # 0.30 is NOT > 0.30
    assert alarm.exceedance_labels(np.array([[0.31]]), theta).tolist() == [[True]]


def test_prob_exceed_gaussian_matches_normal_sf():
    mean = np.array([[0.0, 1.0]])
    std = np.array([[1.0, 2.0]])
    theta = 0.5
    p = alarm.prob_exceed_gaussian(mean, std, theta)
    expected = norm.sf((theta - mean) / std)             # P(Y>theta)
    assert np.allclose(p, expected)
    assert np.all((p >= 0) & (p <= 1))


def test_prob_exceed_gaussian_zero_std_is_step():
    mean = np.array([[0.4, 0.6]])
    std = np.array([[0.0, 0.0]])
    p = alarm.prob_exceed_gaussian(mean, std, 0.5)
    assert p.tolist() == [[0.0, 1.0]]                     # degenerate -> hard step at theta


def test_prob_exceed_quantile_approximates_gaussian():
    # quantiles of N(mu,sigma) -> interpolated P(Y>theta) should track the Gaussian SF
    mu, sigma = 0.2, 0.5
    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q = (mu + sigma * norm.ppf(taus)).reshape(1, 1, -1)   # (1,1,Q)
    for theta in (0.0, 0.2, 0.5):
        p = alarm.prob_exceed_quantile(q, taus, theta)[0, 0]
        assert abs(p - norm.sf((theta - mu) / sigma)) < 0.06


def test_prob_exceed_quantile_monotone_in_theta():
    taus = np.array([0.05, 0.5, 0.95])
    q = np.array([0.0, 1.0, 2.0]).reshape(1, 1, -1)
    ps = [alarm.prob_exceed_quantile(q, taus, t)[0, 0] for t in np.linspace(-1, 3, 20)]
    assert all(a >= b - 1e-9 for a, b in zip(ps, ps[1:]))   # non-increasing as theta rises
    assert ps[0] >= 0.99 and ps[-1] <= 0.01


def test_decisions_threshold_alpha():
    score = np.array([[0.1, 0.6], [0.9, 0.4]])
    fired = alarm.decisions(score, alpha=0.5)
    assert fired.tolist() == [[False, True], [True, False]]


def test_trend_forecast_linear_history_extrapolates_exactly():
    # a perfectly linear history continues exactly along its line (S2 naive baseline)
    L, H = 20, 5
    a, b = 0.10, 0.02
    hist = (a + b * np.arange(L)).reshape(1, L)
    fc = alarm.trend_forecast(hist, H)
    assert fc.shape == (1, H)
    assert np.allclose(fc[0], a + b * np.arange(L, L + H), atol=1e-9)


def test_trend_forecast_constant_is_persistence():
    hist = np.full((3, 10), 0.4)
    fc = alarm.trend_forecast(hist, 4)
    assert fc.shape == (3, 4)
    assert np.allclose(fc, 0.4)                          # zero slope -> hold last value


def test_trend_forecast_window_limits_history_to_recent_samples():
    # flat then rising: a window of 5 sees only the rising tail (slope 1, last value 4)
    hist = np.concatenate([np.zeros((1, 5)), np.arange(5).reshape(1, 5)], axis=1)  # (1,10)
    fc = alarm.trend_forecast(hist, 1, window=5)
    assert np.isclose(fc[0, 0], 5.0)                     # 4 + slope(1)*1
