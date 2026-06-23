import numpy as np

from jmse.eval import diagnostics as dg


def test_acf_white_noise_near_zero():
    rng = np.random.default_rng(0)
    a = dg.acf(rng.standard_normal(20000), 10)
    assert abs(a[0] - 1.0) < 1e-9
    assert np.all(np.abs(a[1:]) < 0.05)


def test_acf_ar1_matches_geometric():
    rng = np.random.default_rng(1)
    phi, n = 0.7, 50000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    a = dg.acf(x, 5)
    assert np.allclose(a[:4], [phi ** k for k in range(4)], atol=0.03)


def test_pacf_ar1_cuts_off_after_lag1():
    rng = np.random.default_rng(2)
    phi, n = 0.6, 50000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    p = dg.pacf_levinson(dg.acf(x, 6))
    assert abs(p[1] - phi) < 0.03
    assert np.all(np.abs(p[2:]) < 0.05)


def test_dominant_period_recovers_sinusoid():
    fs = 1.0
    t = np.arange(4096) / fs
    per = dg.dominant_period(np.sin(2 * np.pi * t / 8.0), fs)
    assert abs(per - 8.0) < 0.5
