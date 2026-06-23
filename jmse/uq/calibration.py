"""Calibration and sharpness metrics for probabilistic forecasts (C1 evaluation).

All functions are pure and array-based; they take predictive distributions in one of
two forms and the observed targets, and return scalar/array scores:
  - Gaussian form: (mean, std) per point  -> gaussian_interval, crps_gaussian
  - Sample form:   (N, M) ensemble draws  -> crps_ensemble
  - Quantile form: (N, Q) quantiles + taus -> crps_from_quantiles

Metrics
  PICP   prediction-interval coverage probability  (target = nominal coverage)
  MPIW   mean prediction-interval width            (sharpness; lower better)
  CRPS   continuous ranked probability score       (proper score; lower better)
  reliability_curve  empirical vs nominal central coverage (reliability diagram)

Units follow the inputs; the caller converts to degrees for reporting.
"""
import numpy as np
from scipy.stats import norm


def gaussian_interval(mean: np.ndarray, std: np.ndarray, coverage: float = 0.90):
    """Central prediction interval [lo, hi] of a Gaussian for the given coverage."""
    z = norm.ppf(0.5 + coverage / 2.0)               # e.g. coverage 0.90 -> z_0.95
    mean = np.asarray(mean, float)
    std = np.asarray(std, float)
    return mean - z * std, mean + z * std


def picp(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations falling within [lower, upper] (inclusive)."""
    y = np.asarray(y, float)
    inside = (y >= np.asarray(lower, float)) & (y <= np.asarray(upper, float))
    return float(np.mean(inside))


def mpiw(lower: np.ndarray, upper: np.ndarray, y_range: float = None) -> float:
    """Mean interval width; if y_range is given, returns the normalized width (NMPIW)."""
    w = float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))
    return w / y_range if y_range else w


def crps_gaussian(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian forecast (Gneiting & Raftery, 2007), per point.

    CRPS(N(mu,sigma), y) = sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
    z = (y - mu)/sigma.
    """
    y = np.asarray(y, float)
    mean = np.asarray(mean, float)
    std = np.maximum(np.asarray(std, float), 1e-12)
    z = (y - mean) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Energy-form CRPS from ensemble samples, per point.

    CRPS = E|X - y| - 0.5 E|X - X'|, estimated from M draws:
        (1/M) sum_m |x_m - y| - (1/(2 M^2)) sum_{m,n} |x_m - x_n|.
    samples: (N, M). Returns (N,).
    """
    y = np.asarray(y, float).reshape(-1, 1)
    s = np.asarray(samples, float)
    M = s.shape[1]
    term1 = np.mean(np.abs(s - y), axis=1)
    # mean absolute pairwise difference per row, vectorized
    diff = np.abs(s[:, :, None] - s[:, None, :])
    term2 = diff.sum(axis=(1, 2)) / (2.0 * M * M)
    return term1 - term2


_trapz = getattr(np, "trapezoid", np.trapz)           # numpy>=2 renames trapz -> trapezoid


def crps_from_quantiles(y: np.ndarray, quantiles: np.ndarray, taus: np.ndarray) -> np.ndarray:
    r"""Approximate CRPS as twice the integral of the pinball loss over tau (CRPS decomposition).

    CRPS = 2 * \int_0^1 pinball_tau(y, q(tau)) dtau, approximated by the trapezoid rule
    over the supplied quantile levels.
    quantiles: (N, Q) ascending in tau; taus: (Q,). Returns (N,).
    """
    y = np.asarray(y, float).reshape(-1, 1)
    q = np.asarray(quantiles, float)
    taus = np.asarray(taus, float)
    diff = y - q                                      # (N, Q)
    pinball = np.where(diff >= 0, taus * diff, (taus - 1) * diff)   # (N, Q)
    return 2.0 * _trapz(pinball, taus, axis=1)


def reliability_curve(y: np.ndarray, mean: np.ndarray, std: np.ndarray, levels: np.ndarray):
    """Empirical vs nominal central coverage for a Gaussian forecast.

    For each nominal level p, builds the central p-interval and measures the fraction of
    observations inside. Returns (nominal, empirical), both (len(levels),). A calibrated
    model lies on the diagonal; empirical < nominal => overconfident (under-dispersed).
    """
    levels = np.asarray(levels, float)
    empirical = np.empty_like(levels)
    for i, p in enumerate(levels):
        lo, hi = gaussian_interval(mean, std, coverage=float(p))
        empirical[i] = picp(y, lo, hi)
    return levels, empirical
