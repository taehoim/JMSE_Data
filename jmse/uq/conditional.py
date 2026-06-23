"""Conditional coverage of prediction intervals (P5, C1 rigor).

Marginal PICP can mask conditional miscoverage: an interval can hit 90% on average while
under-covering the rare large-inclination cells that matter for early warning, or one sea
state, or the far horizons. These helpers slice empirical coverage and mean interval width
by horizon, by a per-window key (sea state Hs or tonnage), and by inclination regime
(below vs. above a danger threshold). All take (N, H) arrays of targets and interval bounds.
"""
import numpy as np
import pandas as pd


def coverage_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """(coverage, mean_width) over all cells of the given arrays."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    return cov, width


def coverage_by_horizon(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> pd.DataFrame:
    """Per-horizon coverage and mean width; one row per column k=1..H."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    rows = []
    for k in range(y.shape[1]):
        cov, w = coverage_width(y[:, [k]], lo[:, [k]], hi[:, [k]])
        rows.append({"horizon_s": k + 1, "coverage": cov, "mpiw_deg": np.degrees(w), "n": y.shape[0]})
    return pd.DataFrame(rows)


def coverage_by_key(y, lo, hi, keys, key_name: str = "key") -> pd.DataFrame:
    """Coverage and mean width grouped by a per-window key (e.g. Hs or tonnage).

    keys is a length-N array; all H horizons of a window share its key. One row per unique
    key value, sorted ascending.
    """
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    keys = np.asarray(keys)
    rows = []
    for v in np.unique(keys):
        m = keys == v
        cov, w = coverage_width(y[m], lo[m], hi[m])
        rows.append({key_name: v, "coverage": cov, "mpiw_deg": np.degrees(w), "n": int(m.sum())})
    return pd.DataFrame(rows)


def coverage_by_regime(y, lo, hi, theta: float) -> pd.DataFrame:
    """Coverage split by inclination regime: cells with y <= theta ('below') vs y > theta ('above').

    The 'above' regime is the rare large-inclination tail the alarm cares about; conditional
    coverage there is the honest test of the interval. theta is in target units (radians).
    """
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    inside = (y >= lo) & (y <= hi)
    above = y > theta
    rows = []
    for name, mask in (("below", ~above), ("above", above)):
        n = int(mask.sum())
        cov = float(inside[mask].mean()) if n else float("nan")
        rows.append({"regime": name, "threshold_deg": round(np.degrees(theta), 1),
                     "coverage": cov, "n": n})
    return pd.DataFrame(rows)
