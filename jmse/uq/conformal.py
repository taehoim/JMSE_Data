"""Distribution-free interval calibration: split conformal + CQR (P5, C1 rigor).

The parametric Gaussian/quantile intervals of Task 18 are only as calibrated as their
distributional assumptions. Split conformal prediction (Vovk; Lei et al. 2018) and
conformalized quantile regression (CQR; Romano, Patterson & Candes 2019) wrap an existing
point or quantile forecaster with a finite-sample, distribution-free coverage guarantee:
calibrate a nonconformity quantile on a held-out calibration set, then widen (or tighten)
the test intervals by that amount. Coverage is guaranteed to be at least 1-alpha under
exchangeability of the calibration and test scores.

All functions operate column-wise so a per-horizon forecast (N, H) is calibrated
independently at each horizon (residual scale grows with horizon).
"""
import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample (1-alpha) conformal quantile of 1-D nonconformity scores.

    Uses the ceil((n+1)(1-alpha))/n empirical order statistic (Lei et al. 2018); if that
    level exceeds 1 (n too small for the requested alpha) the maximum score is returned,
    which conservatively guarantees coverage.
    """
    s = np.asarray(scores, float).ravel()
    n = s.size
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level >= 1.0:
        return float(np.max(s))
    return float(np.quantile(s, level, method="higher"))


def _percol(fn, *cols):
    """Apply a scalar-returning fn to each column, returning a (H,) vector."""
    H = cols[0].shape[1]
    return np.array([fn(*[c[:, k] for c in cols]) for k in range(H)])


def split_conformal(pred_cal: np.ndarray, y_cal: np.ndarray, pred_test: np.ndarray,
                    alpha: float = 0.10):
    """Symmetric split-conformal interval around a point forecast, per horizon column.

    Nonconformity = absolute residual |y - pred| on the calibration set; the interval is
    pred_test +/- q_k with q_k the per-horizon conformal quantile. Returns (lo, hi), both
    shaped like pred_test.
    """
    pred_cal = np.asarray(pred_cal, float)
    y_cal = np.asarray(y_cal, float)
    pred_test = np.asarray(pred_test, float)
    resid = np.abs(y_cal - pred_cal)
    q = _percol(lambda r: conformal_quantile(r, alpha), resid)        # (H,)
    return pred_test - q[None, :], pred_test + q[None, :]


def cqr(qlo_cal: np.ndarray, qhi_cal: np.ndarray, y_cal: np.ndarray,
        qlo_test: np.ndarray, qhi_test: np.ndarray, alpha: float = 0.10):
    """Conformalized quantile regression offset, per horizon column (Romano et al. 2019).

    Conformity score E = max(qlo - y, y - qhi) on the calibration set (positive when y is
    outside the band, negative when comfortably inside). The conformal quantile Q_k of E
    expands (Q>0) or contracts (Q<0) the test band to [qlo - Q, qhi + Q]. Returns (lo, hi).
    """
    qlo_cal = np.asarray(qlo_cal, float)
    qhi_cal = np.asarray(qhi_cal, float)
    y_cal = np.asarray(y_cal, float)
    if qlo_cal.ndim == 1:                                            # allow 1-D convenience
        qlo_cal, qhi_cal, y_cal = qlo_cal[:, None], qhi_cal[:, None], y_cal[:, None]
        qlo_test, qhi_test = np.asarray(qlo_test)[:, None], np.asarray(qhi_test)[:, None]
        squeeze = True
    else:
        qlo_test, qhi_test = np.asarray(qlo_test, float), np.asarray(qhi_test, float)
        squeeze = False
    E = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)                 # (Ncal, H)
    Q = _percol(lambda e: conformal_quantile(e, alpha), E)          # (H,)
    lo, hi = qlo_test - Q[None, :], qhi_test + Q[None, :]
    if squeeze:
        return lo.ravel(), hi.ravel()
    return lo, hi


def first_half_mask(groups: np.ndarray) -> np.ndarray:
    """Boolean mask selecting the first half (by appearance order) of each group's rows.

    The test windows are stored group-contiguous and time-ordered, so the earlier half of
    every group forms an exchangeable temporal calibration set and the later half the
    evaluation set. Odd-length groups put the extra row in the evaluation half.
    """
    groups = np.asarray(groups)
    mask = np.zeros(groups.shape[0], dtype=bool)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        half = len(idx) // 2
        mask[idx[:half]] = True
    return mask
