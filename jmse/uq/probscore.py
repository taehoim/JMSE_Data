"""Probabilistic-calibration scores for the exceedance-probability forecast (P5, C1/C2).

The early-warning layer turns the predictive distribution into a probability
P(theta_TIA(t+k) > theta_d). These scores judge that probability directly:
  Brier score  mean squared error of the probability vs the {0,1} exceedance outcome
  ECE          expected calibration error -- |confidence - accuracy| averaged over
               equal-width probability bins (Naeini et al. 2015; Guo et al. 2017)
  reliability  per-bin (confidence, accuracy, count) for a probability reliability diagram

All inputs are flat arrays of matched probability and binary-label values.
"""
import numpy as np


def brier_score(prob: np.ndarray, label: np.ndarray) -> float:
    """Mean squared error between forecast probability and binary outcome (lower better)."""
    p = np.asarray(prob, float).ravel()
    y = np.asarray(label, float).ravel()
    return float(np.mean((p - y) ** 2))


def reliability_bins(prob: np.ndarray, label: np.ndarray, n_bins: int = 10):
    """Equal-width binning of [0,1]; returns (centers, confidence, accuracy, count) per bin.

    confidence[b] = mean predicted probability in bin b; accuracy[b] = empirical event
    frequency in bin b; count[b] = number of samples. Empty bins yield NaN confidence/accuracy
    and zero count. The last bin is closed on the right so prob == 1 is included.
    """
    p = np.asarray(prob, float).ravel()
    y = np.asarray(label, float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, int)
    for b in range(n_bins):
        m = idx == b
        count[b] = int(m.sum())
        if count[b]:
            conf[b] = p[m].mean()
            acc[b] = y[m].mean()
    return centers, conf, acc, count


def expected_calibration_error(prob: np.ndarray, label: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) * |accuracy_b - confidence_b| over non-empty bins."""
    _, conf, acc, count = reliability_bins(prob, label, n_bins)
    n = count.sum()
    if n == 0:
        return float("nan")
    nz = count > 0
    return float(np.sum(count[nz] / n * np.abs(acc[nz] - conf[nz])))
