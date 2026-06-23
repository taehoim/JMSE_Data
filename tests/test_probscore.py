"""Tests for probabilistic-calibration scores (ECE / Brier) and conditional coverage (P5)."""
import numpy as np

from jmse.uq import conditional as cond
from jmse.uq import probscore as ps


def test_brier_perfect_and_worst():
    label = np.array([1, 0, 1, 0])
    assert ps.brier_score(np.array([1.0, 0.0, 1.0, 0.0]), label) == 0.0
    assert ps.brier_score(np.array([0.0, 1.0, 0.0, 1.0]), label) == 1.0
    assert np.isclose(ps.brier_score(np.full(4, 0.5), label), 0.25)


def test_ece_zero_for_calibrated_forecast():
    # A forecast whose predicted probability equals the empirical frequency in each bin.
    r = np.random.default_rng(0)
    prob = r.uniform(size=200000)
    label = (r.uniform(size=prob.size) < prob).astype(int)   # P(label=1)=prob exactly
    ece = ps.expected_calibration_error(prob, label, n_bins=10)
    assert ece < 0.01                                         # well-calibrated -> near zero


def test_ece_detects_overconfidence():
    # Always predicts 0.9 but is only right half the time -> ECE ~ 0.4.
    prob = np.full(1000, 0.9)
    label = np.zeros(1000, int)
    label[:500] = 1
    ece = ps.expected_calibration_error(prob, label, n_bins=10)
    assert abs(ece - 0.4) < 0.02


def test_reliability_bins_partition_counts():
    prob = np.array([0.05, 0.15, 0.95, 0.96])
    label = np.array([0, 0, 1, 1])
    centers, conf, acc, count = ps.reliability_bins(prob, label, n_bins=10)
    assert count.sum() == 4
    assert count[0] == 1 and count[1] == 1 and count[9] == 2   # last bin holds both 0.9x


def test_coverage_width_basic():
    y = np.array([[0.0, 0.0], [2.0, 2.0]])
    lo = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    hi = np.array([[1.0, 1.0], [1.0, 1.0]])
    cov, w = cond.coverage_width(y, lo, hi)
    assert cov == 0.5                                          # first row inside, second outside
    assert w == 2.0


def test_coverage_by_horizon_and_key():
    y = np.zeros((4, 2))
    lo = np.full((4, 2), -1.0)
    hi = np.full((4, 2), 1.0)
    y[0, 0] = 5.0                                              # one miss at horizon 0
    perh = cond.coverage_by_horizon(y, lo, hi)
    assert perh["coverage"].tolist() == [0.75, 1.0]
    keys = np.array([10, 10, 20, 20])
    bykey = cond.coverage_by_key(y, lo, hi, keys, key_name="tonnage")
    row10 = bykey[bykey["tonnage"] == 10].iloc[0]
    assert row10["coverage"] == 0.75                          # group 10 has the single miss (0.5 of its 4 cells)


def test_coverage_by_regime_splits_on_threshold():
    y = np.array([[0.1], [0.5]])
    lo = np.array([[0.0], [0.0]])
    hi = np.array([[0.2], [1.0]])
    out = cond.coverage_by_regime(y, lo, hi, theta=0.3)
    below = out[out["regime"] == "below"].iloc[0]
    above = out[out["regime"] == "above"].iloc[0]
    assert below["coverage"] == 1.0 and below["n"] == 1
    assert above["coverage"] == 1.0 and above["n"] == 1
