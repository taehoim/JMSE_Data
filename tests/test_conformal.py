"""Tests for split-conformal and CQR interval calibration (P5, C1 rigor)."""
import numpy as np

from jmse.uq import conformal as cf


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_split_conformal_quantile_finite_sample_level():
    # The conformal quantile is the ceil((n+1)(1-alpha))/n empirical order statistic.
    r = _rng(1)
    resid = np.sort(r.standard_normal(99))            # n = 99, alpha = 0.10 -> 0.90
    q = cf.conformal_quantile(resid, alpha=0.10)
    # level = ceil((n+1)(1-alpha))/n = ceil(100*0.9)/99 = 90/99
    expected = np.quantile(resid, 90 / 99, method="higher")
    assert np.isclose(q, expected)


def test_split_conformal_covers_at_nominal_rate():
    # Gaussian point forecast with constant bias; calibrate on a calib split, test on fresh draws.
    r = _rng(2)
    n = 4000
    y_cal = r.standard_normal(n)
    pred_cal = np.zeros(n)
    y_te = r.standard_normal(n)
    pred_te = np.zeros(n)
    alpha = 0.10
    q = cf.conformal_quantile(np.abs(y_cal - pred_cal), alpha)
    lo, hi = pred_te - q, pred_te + q
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert abs(cov - (1 - alpha)) < 0.02          # near nominal, distribution-free


def test_split_conformal_per_horizon_shapes_and_coverage():
    r = _rng(3)
    H = 5
    y_cal = r.standard_normal((2000, H)) * np.arange(1, H + 1)     # growing scale per horizon
    pred_cal = np.zeros_like(y_cal)
    y_te = r.standard_normal((2000, H)) * np.arange(1, H + 1)
    pred_te = np.zeros_like(y_te)
    lo, hi = cf.split_conformal(pred_cal, y_cal, pred_te, alpha=0.10)
    assert lo.shape == hi.shape == pred_te.shape
    # wider intervals at later horizons (scale grows)
    widths = (hi - lo).mean(axis=0)
    assert np.all(np.diff(widths) > 0)
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert abs(cov - 0.90) < 0.03


def test_cqr_corrects_undercoverage():
    # Quantile predictions that are too tight -> CQR widens them to nominal coverage.
    r = _rng(4)
    n = 4000
    y_cal = r.standard_normal(n)
    # nominal 90% band but deliberately too narrow (50% band quantiles)
    qlo_cal = np.full(n, -0.674)
    qhi_cal = np.full(n, 0.674)
    y_te = r.standard_normal(n)
    qlo_te = np.full(n, -0.674)
    qhi_te = np.full(n, 0.674)
    base_cov = np.mean((y_te >= qlo_te) & (y_te <= qhi_te))
    lo, hi = cf.cqr(qlo_cal, qhi_cal, y_cal, qlo_te, qhi_te, alpha=0.10)
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert base_cov < 0.6                              # started badly under-covered
    assert abs(cov - 0.90) < 0.03                      # CQR restored nominal coverage


def test_cqr_can_tighten_overcovered_band():
    # If the supplied band is too wide, CQR shifts inward (negative offset) toward nominal.
    r = _rng(5)
    n = 4000
    y_cal = r.standard_normal(n)
    qlo_cal = np.full(n, -3.0)
    qhi_cal = np.full(n, 3.0)
    y_te = r.standard_normal(n)
    lo, hi = cf.cqr(qlo_cal, qhi_cal, y_cal, np.full(n, -3.0), np.full(n, 3.0), alpha=0.10)
    assert (hi - lo).mean() < 6.0                      # tightened from the 6.0-wide input
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert abs(cov - 0.90) < 0.03


def test_calibration_mask_splits_each_group_in_half():
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    mask = cf.first_half_mask(groups)
    # group 0: 4 windows -> 2 calib; group 1: 6 -> 3 calib
    assert mask.tolist() == [True, True, False, False, True, True, True, False, False, False]
