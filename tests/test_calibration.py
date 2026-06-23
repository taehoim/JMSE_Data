import numpy as np
from scipy.stats import norm

from jmse.uq import calibration as cal


def test_picp_of_perfectly_calibrated_gaussian():
    rng = np.random.default_rng(0)
    n = 200_000
    y = rng.standard_normal(n)                       # true ~ N(0,1)
    mean = np.zeros(n)
    std = np.ones(n)
    lo, hi = cal.gaussian_interval(mean, std, coverage=0.90)
    picp = cal.picp(y, lo, hi)
    assert abs(picp - 0.90) < 0.01                   # ~90% fall inside


def test_gaussian_interval_uses_correct_z():
    lo, hi = cal.gaussian_interval(np.array([0.0]), np.array([1.0]), coverage=0.90)
    assert abs(hi[0] - norm.ppf(0.95)) < 1e-9        # upper = z_0.95 = 1.6449
    assert abs(lo[0] + norm.ppf(0.95)) < 1e-9


def test_mpiw_basic_and_normalized():
    lo = np.array([0.0, 1.0])
    hi = np.array([2.0, 4.0])                          # widths 2 and 3 -> mean 2.5
    assert abs(cal.mpiw(lo, hi) - 2.5) < 1e-12
    y = np.array([0.0, 4.0])                           # range 4 -> normalized 0.625
    assert abs(cal.mpiw(lo, hi, y_range=y.max() - y.min()) - 0.625) < 1e-12


def test_crps_gaussian_closed_form_at_mean():
    # CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi) = 0.233692...
    val = cal.crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0]
    assert abs(val - (2 * norm.pdf(0) - 1 / np.sqrt(np.pi))) < 1e-9


def test_crps_gaussian_matches_ensemble_energy_form():
    rng = np.random.default_rng(1)
    mu, sigma, y = 0.3, 0.7, 0.5
    samples = mu + sigma * rng.standard_normal((1, 40_000))
    closed = cal.crps_gaussian(np.array([y]), np.array([mu]), np.array([sigma]))[0]
    energy = cal.crps_ensemble(np.array([y]), samples)[0]
    assert abs(closed - energy) < 5e-3


def test_crps_scales_with_sigma_and_penalizes_miss():
    # wider sigma at the mean costs more; a far miss costs more than a near one
    near = cal.crps_gaussian(np.array([0.1]), np.array([0.0]), np.array([1.0]))[0]
    far = cal.crps_gaussian(np.array([3.0]), np.array([0.0]), np.array([1.0]))[0]
    assert far > near


def test_reliability_curve_diagonal_when_calibrated():
    rng = np.random.default_rng(2)
    n = 100_000
    y = rng.standard_normal(n)
    mean, std = np.zeros(n), np.ones(n)
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    nominal, empirical = cal.reliability_curve(y, mean, std, levels)
    assert np.allclose(nominal, levels)
    assert np.max(np.abs(empirical - nominal)) < 0.01    # close to diagonal


def test_reliability_detects_overconfidence():
    # under-dispersed (std too small) -> empirical coverage below nominal
    rng = np.random.default_rng(3)
    n = 100_000
    y = rng.standard_normal(n)
    mean, std = np.zeros(n), np.full(n, 0.5)             # half the true spread
    _, empirical = cal.reliability_curve(y, mean, std, np.array([0.9]))
    assert empirical[0] < 0.9


def test_crps_ensemble_from_quantiles_consistency():
    # quantile-based CRPS approx on a Gaussian's quantiles is near the closed form
    mu, sigma, y = 0.0, 1.0, 0.4
    taus = np.linspace(0.01, 0.99, 99)
    q = mu + sigma * norm.ppf(taus)                      # (Q,)
    approx = cal.crps_from_quantiles(np.array([y]), q[None, :], taus)[0]
    closed = cal.crps_gaussian(np.array([y]), np.array([mu]), np.array([sigma]))[0]
    assert abs(approx - closed) < 0.02
