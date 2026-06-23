"""Benchmark aggregation and significance testing (C3, Task 13).

Two layers:
1. `aggregate_seeds` — mean +/- std across `config.SEEDS` per (model, horizon),
   producing the benchmark table T4.
2. `paired_wilcoxon_vs` — paired Wilcoxon signed-rank over per-window absolute
   errors between each model and a reference model. The test is run at the
   *window* level (large N), not the seed level (n=5 has no usable power), so the
   p-values are meaningful. A Holm correction guards the family of comparisons.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Display order / friendly labels for the benchmark set.
MODEL_ORDER = ["persistence", "ar", "kalman", "ridge", "gbm", "tcn", "transformer", "gru", "lstm"]
MODEL_LABELS = {
    "persistence": "Persistence", "ar": "AR(8)", "kalman": "Kalman",
    "ridge": "Ridge (window)", "gbm": "GBM (window)",
    "tcn": "TCN", "transformer": "Transformer", "gru": "GRU", "lstm": "LSTM",
}


def _order_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def aggregate_seeds(raw: pd.DataFrame) -> pd.DataFrame:
    """raw columns: model, seed, horizon_s, rmse_deg, mae_deg, r2 (horizon_s incl. 'overall').

    Returns one row per (model, horizon_s) with mean/std/n over seeds for each metric,
    ordered by MODEL_ORDER then horizon. Deterministic models (single seed) get std=0.
    """
    metrics = ["rmse_deg", "mae_deg", "r2"]
    g = raw.groupby(["model", "horizon_s"], sort=False)
    agg = g[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["model", "horizon_s"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    agg["n_seeds"] = g.size().values
    for m in metrics:                                   # single-seed std -> 0, not NaN
        agg[f"{m}_std"] = agg[f"{m}_std"].fillna(0.0)
    agg["_mk"] = agg["model"].map(_order_key)
    # horizon sort: numeric first (1..H), 'overall' last
    agg["_hk"] = agg["horizon_s"].map(lambda h: 999 if h == "overall" else int(h))
    agg = agg.sort_values(["_mk", "_hk"]).drop(columns=["_mk", "_hk"]).reset_index(drop=True)
    return agg


def mean_std_over_seeds(raw: pd.DataFrame, group_cols, metric_cols,
                        passthrough_cols=(), seed_col: str = "seed") -> pd.DataFrame:
    """Generic per-group mean/std over seeds for the UQ / early-warning / representation tables (S1).

    Unlike `aggregate_seeds` (hard-wired to the benchmark's model x horizon schema), this groups by
    arbitrary `group_cols` and reports mean+/-std (ddof=1) and a seed count for each metric in
    `metric_cols`. Single-seed (or seed-invariant) groups get std=0 rather than NaN, matching the
    deterministic-model convention in T4. `passthrough_cols` are constants across seeds (e.g. the
    label prevalence and event count of an early-warning cell) and are carried through by first value.

    `raw` has one row per (group, seed); returns one row per group with columns
    group_cols + {metric}_mean + {metric}_std + passthrough_cols + n_seeds.
    """
    metric_cols, passthrough_cols = list(metric_cols), list(passthrough_cols)
    g = raw.groupby(group_cols, sort=False)
    agg = g[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m in metric_cols for s in ("mean", "std")]
    for m in metric_cols:                                # seed-invariant / single seed -> std 0
        agg[f"{m}_std"] = agg[f"{m}_std"].fillna(0.0)
    if passthrough_cols:
        agg = agg.join(g[passthrough_cols].first())
    agg["n_seeds"] = g.size()
    return agg.reset_index()


def mean_only(meanstd: pd.DataFrame, metric_cols) -> pd.DataFrame:
    """Collapse a `mean_std_over_seeds` frame back to the single-seed table schema (means only).

    Lets the multi-seed runners emit the standard `T*.csv` (mean values, original column names)
    for the existing plotters and the paper's point estimates, alongside the `*_meanstd.csv` that
    carries the +/-std for the manuscript tables. Keeps figures/back-compat working unchanged.
    """
    metric_cols = list(metric_cols)
    out = meanstd.copy()
    for m in metric_cols:
        out[m] = out[f"{m}_mean"]
    drop = [f"{m}_{s}" for m in metric_cols for s in ("mean", "std")] + ["n_seeds"]
    return out.drop(columns=[c for c in drop if c in out.columns])


def write_t4(agg: pd.DataFrame, out_dir: Path) -> Path:
    """Write the overall-horizon slice of the aggregate as the headline table T4."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = agg[agg["horizon_s"] == "overall"].copy()
    overall.insert(0, "label", overall["model"].map(lambda m: MODEL_LABELS.get(m, m)))
    path = out_dir / "T4_benchmark.csv"
    overall.to_csv(path, index=False)
    agg.to_csv(out_dir / "T4_benchmark_per_horizon.csv", index=False)
    return path


def _abs_errors(npz_path: Path) -> np.ndarray:
    """Flatten per-window absolute errors (radians) from a saved preds npz."""
    z = np.load(npz_path)
    return np.abs(z["y_pred"] - z["y_true"]).ravel()


def paired_wilcoxon_vs(preds_dir: Path, reference: str = "lstm") -> pd.DataFrame:
    """Paired Wilcoxon (window-level abs error) of every model vs `reference`.

    Lower median error than the reference => the model is better. Returns one row per
    competing model with the statistic, raw p, Holm-adjusted p, and median error delta
    (model - reference; negative means the model beats the reference).
    """
    preds_dir = Path(preds_dir)
    ref_path = preds_dir / f"{reference}.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"reference preds not found: {ref_path}")
    ref_err = _abs_errors(ref_path)

    rows = []
    for npz in sorted(preds_dir.glob("*.npz"), key=lambda p: _order_key(p.stem)):
        model = npz.stem
        if model == reference:
            continue
        err = _abs_errors(npz)
        if err.shape != ref_err.shape:
            raise ValueError(f"{model}: error shape {err.shape} != reference {ref_err.shape}")
        try:
            stat, p = wilcoxon(err, ref_err, zero_method="wilcox", alternative="two-sided")
        except ValueError:                              # all-zero differences
            stat, p = float("nan"), 1.0
        rows.append({
            "model": model, "label": MODEL_LABELS.get(model, model),
            "reference": reference, "statistic": float(stat), "p_value": float(p),
            "median_abs_err_delta_rad": float(np.median(err) - np.median(ref_err)),
        })
    df = pd.DataFrame(rows)
    df = _holm(df, "p_value")
    return df


def _holm(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    """Holm-Bonferroni step-down adjusted p-values over the comparison family."""
    df = df.copy()
    order = df[pcol].fillna(1.0).sort_values().index.tolist()
    m = len(order)
    adj, running = {}, 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * df.loc[idx, pcol])
        running = max(running, val)                     # enforce monotonicity
        adj[idx] = running
    df["p_holm"] = df.index.map(adj)
    return df
