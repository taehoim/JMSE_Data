"""Leakage-safe sliding windows, chronological splits, and train-only scaling.

Windows never straddle a (tonnage, Hs) file boundary. Scaler statistics are fit on
the training split only; validation/test reuse those statistics.
"""
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from jmse import config
from jmse.data.curate import load_curated


def make_windows(values: np.ndarray, target: np.ndarray, lookback: int, horizon: int):
    """Return (X, y, idx).

    X[i]   = values[i : i+lookback]                 -> (lookback, n_features)
    y[i]   = target[i+lookback : i+lookback+horizon] -> (horizon,)
    idx[i] = last input index t = i + lookback - 1
    Number of windows = N - lookback - horizon + 1.
    """
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    N, F = values.shape
    n_win = N - lookback - horizon + 1
    if n_win <= 0:
        return (np.empty((0, lookback, F)), np.empty((0, horizon)), np.empty((0,), int))

    Xwin = sliding_window_view(values, lookback, axis=0).transpose(0, 2, 1)  # (N-L+1, L, F)
    X = Xwin[:n_win]
    twin = sliding_window_view(target, horizon)                              # (N-H+1, H)
    y = twin[lookback: lookback + n_win]
    idx = np.arange(lookback - 1, lookback - 1 + n_win)
    return X, y, idx


def chrono_split(n: int, train: float = 0.7, val: float = 0.15, embargo: int = 0):
    """Chronological split with an optional embargo (windows dropped at the start of
    val and test) so windows in different splits share no timestep. Use
    embargo = lookback + horizon - 1.
    """
    n_tr = int(n * train)
    n_va = int(n * val)
    va_start = min(n_tr + embargo, n_tr + n_va)
    te_start = min(n_tr + n_va + embargo, n)
    return slice(0, n_tr), slice(va_start, n_tr + n_va), slice(te_start, n)


def fit_scaler(X: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(X.reshape(-1, X.shape[-1]))
    return sc


def apply_scaler(X: np.ndarray, sc: StandardScaler) -> np.ndarray:
    shp = X.shape
    return sc.transform(X.reshape(-1, shp[-1])).reshape(shp)


def _group_windows(df, lookback, horizon, features=None):
    """Yield (tonnage, Hs, realization, X, y, last_obs, yhist, idx) per series; no straddle.

    Windows are built within a single (tonnage, Hs, realization) series, so a window
    never mixes two wave-phase realizations (same leakage-safety rule that kept windows
    inside a (tonnage, Hs) file before the realization axis was added).

    last_obs[i] = target value at the window's last input step t (= Xacc(t)), used
    by the persistence baseline.
    yhist[i]    = target trajectory over the input window, Xacc[t-L+1 .. t] -> (L,);
                  its last column equals last_obs. Used by the classical AR/Kalman
                  baselines, which forecast recursively from the target's own history.
    idx[i]      = absolute last-input timestep t within the series' (time-sorted) order;
                  consecutive within a chrono split (stride-1 windows), so it orders the
                  windows in time for the early-warning lead-time analysis (P3).
    `features` selects the input columns (default config.FEATURES) for the C5 ablations.
    """
    features = features or config.FEATURES
    for (ton, hs, real), g in df.sort_values("time").groupby(["tonnage", "Hs", "realization"]):
        vals = g[features].to_numpy(float)
        tgt = g[config.TARGET].to_numpy(float)
        X, y, idx = make_windows(vals, tgt, lookback, horizon)
        if len(X):
            yhist = sliding_window_view(tgt, lookback)[: len(X)]   # (n_win, L), aligned with X
            yield ton, hs, real, X, y, tgt[idx], yhist, idx


def _assemble_split(groups, embargo: int = 0):
    """Chrono-split each group 70/15/15 (with embargo) and concatenate across groups.

    Each group is a single (tonnage, Hs, realization) series, so the chrono split is
    applied WITHIN a realization and windows never straddle a realization boundary.
    Also records per-window group identity (`group_*`, an int id) and last-input time
    (`tidx_*`); `group_keys` maps id -> (tonnage, Hs, realization). These carry no signal
    and are used only to reconstruct each series' contiguous test timeline for lead-time
    analysis.
    """
    buckets = {k: [] for k in ("train", "val", "test")}
    group_keys = []
    for gid, (ton, hs, real, X, y, lo, yh, idx) in enumerate(groups):
        group_keys.append((ton, hs, real))
        tr, va, te = chrono_split(len(X), embargo=embargo)
        gid_arr = np.full(len(X), gid, dtype=int)
        for name, sl in (("train", tr), ("val", va), ("test", te)):
            buckets[name].append((X[sl], y[sl], lo[sl], yh[sl], gid_arr[sl], idx[sl]))
    out = {"group_keys": group_keys}
    # Every current series yields >= ~619 windows (shortest is 10t/Hs7 r3 at 643 rows)
    # >> embargo 24, so all three splits are non-empty; guard protects future short series.
    for name, parts in buckets.items():
        if not parts or all(len(p[0]) == 0 for p in parts):
            raise ValueError(f"empty '{name}' split — series too short for lookback+horizon+embargo")
        out[f"X_{name}"] = np.concatenate([p[0] for p in parts])
        out[f"y_{name}"] = np.concatenate([p[1] for p in parts])
        out[f"last_obs_{name}"] = np.concatenate([p[2] for p in parts])
        out[f"yhist_{name}"] = np.concatenate([p[3] for p in parts])
        out[f"group_{name}"] = np.concatenate([p[4] for p in parts])
        out[f"tidx_{name}"] = np.concatenate([p[5] for p in parts])
    return out


def build_combined_holdout(hold_hs: float, hold_ton: int,
                           lookback: int = None, horizon: int = None) -> dict:
    """Joint Hs x vessel holdout: test is the single (hold_ton, hold_hs) cell, and training sees
    NEITHER that sea state NOR that vessel (the whole row and column are removed from train/val).

    This is a stricter test than LOHO/LOVO: the held cell is novel on both axes simultaneously, so
    the model must extrapolate jointly. The L-shaped cells that share exactly one axis with the held
    cell are dropped (neither trained on nor tested) so the test reflects pure joint extrapolation.
    Train/val come from the 8 remaining cells (chrono 85/15 per cell, embargoed). Scaler fit on train.
    """
    lookback = lookback or config.LOOKBACK
    horizon = horizon or config.HORIZON
    df = load_curated()
    train_parts, val_parts, test_parts, group_keys = [], [], [], []
    for gid, (ton, hs, real, X, y, lo, yh, idx) in enumerate(_group_windows(df, lookback, horizon)):
        group_keys.append((ton, hs, real))
        gid_arr = np.full(len(X), gid, dtype=int)
        # Hold out by (ton, hs) regardless of realization: the held cell spans all 6.
        is_held_cell = (hs == hold_hs and ton == hold_ton)
        shares_one_axis = (hs == hold_hs) ^ (ton == hold_ton)
        if is_held_cell:
            test_parts.append((X, y, lo, yh, gid_arr, idx))
        elif shares_one_axis:
            continue                                            # drop L-shaped cells
        else:
            tr, va, _ = chrono_split(len(X), train=0.85, val=0.15, embargo=lookback + horizon - 1)
            train_parts.append((X[tr], y[tr], lo[tr], yh[tr], gid_arr[tr], idx[tr]))
            val_parts.append((X[va], y[va], lo[va], yh[va], gid_arr[va], idx[va]))
    if not test_parts:
        raise ValueError(f"No held cell for Hs={hold_hs}, tonnage={hold_ton}.")

    def cat(parts, j):
        return np.concatenate([p[j] for p in parts])

    d = {"group_keys": group_keys}
    for split, parts in (("train", train_parts), ("val", val_parts), ("test", test_parts)):
        d[f"X_{split}"] = cat(parts, 0)
        d[f"y_{split}"] = cat(parts, 1)
        d[f"last_obs_{split}"] = cat(parts, 2)
        d[f"yhist_{split}"] = cat(parts, 3)
        d[f"group_{split}"] = cat(parts, 4)
        d[f"tidx_{split}"] = cat(parts, 5)
    return _scale_dict(d)


def _scale_dict(d: dict) -> dict:
    """Fit scaler on X_train, apply to all X_*; attach 'scaler'."""
    sc = fit_scaler(d["X_train"])
    for name in ("train", "val", "test"):
        key = f"X_{name}"
        if key in d:
            d[key] = apply_scaler(d[key], sc)
    d["scaler"] = sc
    return d


def build_id_arrays(lookback: int = None, horizon: int = None, features=None) -> dict:
    """In-distribution arrays: chrono 70/15/15 within each file, train-only scaling.

    `features` selects input columns (default config.FEATURES) for the C5 feature ablation.
    """
    lookback = lookback or config.LOOKBACK
    horizon = horizon or config.HORIZON
    df = load_curated()
    groups = list(_group_windows(df, lookback, horizon, features=features))
    d = _assemble_split(groups, embargo=lookback + horizon - 1)
    return _scale_dict(d)


def _build_holdout(is_held, lookback, horizon, empty_msg, df=None) -> dict:
    """Generic leave-out builder: groups matching `is_held(ton, hs, real)` are the test
    set in full; the rest are chrono-split (85/15, embargoed) into train/val. Scaler fit
    on train only. Shared by the OOD (Hs/tonnage) and LORO (realization) studies.

    `df` overrides the curated source frame (e.g. the unclamped GM-floor variant); it
    defaults to the canonical curated parquet.
    """
    df = load_curated() if df is None else df
    train_parts, val_parts, test_parts = [], [], []
    group_keys = []
    for gid, (ton, hs, real, X, y, lo, yh, idx) in enumerate(_group_windows(df, lookback, horizon)):
        group_keys.append((ton, hs, real))
        gid_arr = np.full(len(X), gid, dtype=int)
        if is_held(ton, hs, real):
            test_parts.append((X, y, lo, yh, gid_arr, idx))
        else:
            tr, va, _ = chrono_split(len(X), train=0.85, val=0.15, embargo=lookback + horizon - 1)
            train_parts.append((X[tr], y[tr], lo[tr], yh[tr], gid_arr[tr], idx[tr]))
            val_parts.append((X[va], y[va], lo[va], yh[va], gid_arr[va], idx[va]))

    if not test_parts:
        raise ValueError(empty_msg)
    # Every non-held series yields >= ~619 windows >> embargo 24, so train/val are always
    # populated here; guard protects external users / future short series post-P5.
    for split, parts in (("train", train_parts), ("val", val_parts)):
        if not parts or all(len(p[0]) == 0 for p in parts):
            raise ValueError(f"empty '{split}' split — series too short for lookback+horizon+embargo")

    def cat(parts, j):
        return np.concatenate([p[j] for p in parts])

    d = {"group_keys": group_keys}
    for split, parts in (("train", train_parts), ("val", val_parts), ("test", test_parts)):
        d[f"X_{split}"] = cat(parts, 0)
        d[f"y_{split}"] = cat(parts, 1)
        d[f"last_obs_{split}"] = cat(parts, 2)
        d[f"yhist_{split}"] = cat(parts, 3)
        d[f"group_{split}"] = cat(parts, 4)
        d[f"tidx_{split}"] = cat(parts, 5)
    return _scale_dict(d)


def build_ood_arrays(
    hold_hs: Optional[float] = None,
    hold_ton: Optional[int] = None,
    lookback: int = None,
    horizon: int = None,
    df=None,
) -> dict:
    """Out-of-distribution arrays: held-out Hs and/or tonnage form the test set.

    Train/val come from the remaining groups (chrono val carved per remaining group);
    the held-out group(s) are the test set in full. Holding out a vessel/sea state holds
    ALL of its realizations (the predicate ignores the realization axis). Scaler fit on
    train only. `df` overrides the curated source (e.g. the unclamped GM-floor variant).
    """
    lookback = lookback or config.LOOKBACK
    horizon = horizon or config.HORIZON

    def is_held(ton, hs, real):
        return (hold_hs is not None and hs == hold_hs) or (hold_ton is not None and ton == hold_ton)

    return _build_holdout(is_held, lookback, horizon,
                          "No held-out group selected (set hold_hs and/or hold_ton).", df=df)


def build_loro_arrays(hold_real: int, lookback: int = None, horizon: int = None) -> dict:
    """Leave-One-Realization-Out arrays: all windows of wave-phase realization `hold_real`
    form the test set; train/val come from the other realizations of EVERY (ton, Hs) cell.

    Unlike LOHO/LOVO, every (tonnage, Hs) appears in both train and test (only the wave
    phase differs), so the fold isolates generalization to an unseen phase realization
    rather than to an unseen vessel or sea state. Scaler fit on train only.
    """
    lookback = lookback or config.LOOKBACK
    horizon = horizon or config.HORIZON

    def is_held(ton, hs, real):
        return real == hold_real

    return _build_holdout(is_held, lookback, horizon,
                          f"No windows for held realization {hold_real}.")
