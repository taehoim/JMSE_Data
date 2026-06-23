"""Alarm-scoring primitives for the early-warning decision layer (C2).

The danger event for window i at horizon k is the *strict* exceedance Xacc(t+k) > theta;
`EXCEEDS` is the single source of truth for that comparator and is kept identical to
`data.curate.dataset_summary` (which uses `>`), so labels here match the dataset's
reported exceedance frequencies (M3 guardrail).

Two alarm families are scored against those labels:
  - probabilistic: score = P(Xacc(t+k) > theta) from a predictive distribution
      * Gaussian (mean, std)        -> exact via the normal survival function
      * quantiles (q at taus)       -> piecewise-linear CDF, linearly extrapolated + clipped
  - point: score = the point/median forecast itself (degenerate sigma -> 0 case)

All score/label arrays are shaped (N, horizon); thresholds theta are in the target units
(radians, matching y).
"""
import operator

import numpy as np
from scipy.stats import norm

EXCEEDS = operator.gt          # strict '>' — matches data.curate.dataset_summary (M3)


def exceedance_labels(y_true: np.ndarray, theta: float) -> np.ndarray:
    """Binary danger labels Xacc > theta (strict), shape == y_true."""
    return EXCEEDS(np.asarray(y_true, float), theta)


def prob_exceed_gaussian(mean: np.ndarray, std: np.ndarray, theta: float) -> np.ndarray:
    """P(Y > theta) for Y ~ N(mean, std). Zero-std degenerates to a hard step at theta."""
    mean = np.asarray(mean, float)
    std = np.asarray(std, float)
    out = np.where(EXCEEDS(mean, theta), 1.0, 0.0)        # step where std == 0
    nz = std > 0
    out = np.where(nz, norm.sf((theta - mean) / np.where(nz, std, 1.0)), out)
    return out


def prob_exceed_quantile(quantiles: np.ndarray, taus, theta: float) -> np.ndarray:
    """P(Y > theta) from predicted quantiles via a piecewise-linear predictive CDF.

    quantiles: (N, H, Q) ascending in tau; taus: (Q,). The CDF is linear through the
    (q_j, tau_j) points; outside [q_min, q_max] the nearest segment's slope is extended
    and the result clipped to [0, 1]. Returns (N, H) = 1 - CDF(theta).
    """
    q = np.asarray(quantiles, float)
    taus = np.asarray(taus, float)
    N, H, Q = q.shape
    flat = q.reshape(-1, Q)                                # (N*H, Q)
    cdf = np.empty(flat.shape[0])
    for i in range(flat.shape[0]):
        cdf[i] = _interp_cdf(flat[i], taus, theta)
    p = 1.0 - cdf.reshape(N, H)
    return np.clip(p, 0.0, 1.0)


def _interp_cdf(qrow: np.ndarray, taus: np.ndarray, theta: float) -> float:
    """CDF(theta) for one window's quantiles: linear interp, linear-extrapolate, clip."""
    if theta <= qrow[0]:                                  # below lowest quantile -> extrapolate down
        slope = (taus[1] - taus[0]) / max(qrow[1] - qrow[0], 1e-12)
        return float(np.clip(taus[0] + slope * (theta - qrow[0]), 0.0, 1.0))
    if theta >= qrow[-1]:                                 # above highest quantile -> extrapolate up
        slope = (taus[-1] - taus[-2]) / max(qrow[-1] - qrow[-2], 1e-12)
        return float(np.clip(taus[-1] + slope * (theta - qrow[-1]), 0.0, 1.0))
    return float(np.interp(theta, qrow, taus))            # interior: monotone piecewise-linear


def decisions(score: np.ndarray, alpha: float) -> np.ndarray:
    """Fire the alarm where score >= alpha (operating point on the probabilistic score)."""
    return np.asarray(score, float) >= alpha


def trend_forecast(yhist: np.ndarray, horizon: int, window: int = None) -> np.ndarray:
    """Learning-free domain-rule forecast: linear-trend extrapolation of the observed angle (S2).

    Fits a least-squares line to the last `window` samples of each window's target history
    (Xacc over t-L+1..t) and extends it `horizon` steps ahead. This is the "current angle plus
    recent trend" rule a bridge watch already applies, and it needs no model or training. It serves
    as a domain baseline for the early-warning alarm: its point forecast is scored against the same
    exceedance labels as the learned point/probabilistic alarms. `window` defaults to the full
    lookback L; window < 2 degenerates to persistence (slope 0).

    yhist: (N, L) target history aligned with the model inputs; returns (N, horizon) forecasts in
    the same units as yhist (radians).
    """
    yhist = np.asarray(yhist, float)
    N, L = yhist.shape
    w = L if window is None else int(min(window, L))
    yy = yhist[:, -w:]                                     # (N, w) most-recent samples
    last = yy[:, -1]                                       # (N,) current angle Xacc(t)
    if w < 2:
        slope = np.zeros(N)                                # not enough history -> persistence
    else:
        x = np.arange(w, dtype=float)
        xc = x - x.mean()
        slope = (yy - yy.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)   # (N,) LS slope per step
    k = np.arange(1, horizon + 1, dtype=float)            # forecast offsets 1..H from t
    return last[:, None] + slope[:, None] * k[None, :]    # (N, horizon)
