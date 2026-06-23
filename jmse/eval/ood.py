"""Generalization study (Task 23-24, C4): Leave-One-Hs-Out and Leave-One-Vessel-Out.

Each fold holds out one sea state (LOHO) or one vessel tonnage (LOVO): the held-out
group is the test set in full and is entirely absent from train/val (build_ood_arrays).
Comparing fold performance to the in-distribution (ID) baseline quantifies how well the
forecaster extrapolates to unseen conditions.

Usage:
    python -m jmse.eval.ood [--config jmse/run_configs/lstm_id.yaml] [--seeds 0 1 2] [--smoke]
Outputs under results/generalization[_smoke]/: raw_metrics.csv, per_fold.csv,
T7_generalization.csv (+ F9 via plots.generalization).
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from jmse import config
from jmse.eval import stats
from jmse.train import load_config, train


N_REALIZATIONS = 6


def generalization_folds():
    """The 3 LOHO (sea state) + 5 LOVO (tonnage) + 6 LORO (wave-phase) leave-one-out folds.

    LORO holds out one wave-phase realization at a time: it isolates generalization to an
    unseen phase realization (every vessel/sea state is in both train and test), separating
    it from the unseen-vessel (LOVO) and unseen-sea-state (LOHO) extrapolation regimes.
    """
    folds = []
    for hs in config.HS_VALUES:
        folds.append({"regime": "loho", "hold_hs": float(hs), "hold_ton": None,
                      "hold_real": None, "label": f"Hs={hs:g}", "scope": "LOHO"})
    for ton in config.VESSELS:
        folds.append({"regime": "lovo", "hold_hs": None, "hold_ton": int(ton),
                      "hold_real": None, "label": f"{ton}t", "scope": "LOVO"})
    for r in range(N_REALIZATIONS):
        folds.append({"regime": "loro", "hold_hs": None, "hold_ton": None,
                      "hold_real": int(r), "label": f"r{r}", "scope": "LORO"})
    return folds


def combined_holdout_folds():
    """Joint Hs x vessel folds: the four grid corners plus the center cell.

    Corners test joint extrapolation (both axes at an extreme); the center tests joint
    interpolation (both axes interior). Training sees neither the held sea state nor the held
    vessel (build_combined_holdout).
    """
    return [
        {"hold_hs": 7.0, "hold_ton": 10, "label": "Hs7x10t", "kind": "corner"},
        {"hold_hs": 3.0, "hold_ton": 10, "label": "Hs3x10t", "kind": "corner"},
        {"hold_hs": 7.0, "hold_ton": 50, "label": "Hs7x50t", "kind": "corner"},
        {"hold_hs": 3.0, "hold_ton": 50, "label": "Hs3x50t", "kind": "corner"},
        {"hold_hs": 5.0, "hold_ton": 30, "label": "Hs5x30t", "kind": "center"},
    ]


def run_combined_holdout(base_config="jmse/run_configs/lstm_id.yaml", seeds=None, smoke=False):
    """Train the backbone on each joint Hs x vessel fold and tabulate ID-relative skill -> T15."""
    out_dir = config.RESULTS_DIR / ("combined_smoke" if smoke else "combined")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(base_config)
    seeds = [0] if smoke else (seeds or [0, 1, 2])

    records = []
    for fold in combined_holdout_folds():
        for seed in seeds:
            cfg = {**base, "regime": "combined", "hold_hs": fold["hold_hs"],
                   "hold_ton": fold["hold_ton"], "seed": seed, "name": f"comb_{fold['label']}"}
            deg = train(cfg, smoke=smoke)
            print(f"{fold['label']:9s} ({fold['kind']:6s}) seed={seed} "
                  f"overall R2={deg['overall']['r2']:.4f} RMSE={deg['overall']['rmse']:.2f}deg")
            for r in _rows(fold["label"], fold["kind"], seed, deg):
                records.append(r)
    raw = pd.DataFrame.from_records(records)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)

    overall = raw[raw["horizon_s"] == "overall"]
    g = overall.groupby(["model", "scope"])
    t15 = g.agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                rmse_deg_mean=("rmse_deg", "mean"), rmse_deg_std=("rmse_deg", "std"),
                n_seeds=("seed", "nunique")).reset_index()
    t15[["r2_std", "rmse_deg_std"]] = t15[["r2_std", "rmse_deg_std"]].fillna(0.0)
    t15 = t15.rename(columns={"model": "fold", "scope": "kind"})
    t15.to_csv(out_dir / "T15_combined_holdout.csv", index=False)
    return out_dir, raw, t15


def _rows(label, scope, seed, deg):
    rows = []
    for k in range(len(deg["r2"])):
        rows.append({"model": label, "scope": scope, "seed": seed, "horizon_s": k + 1,
                     "rmse_deg": deg["rmse"][k], "mae_deg": deg["mae"][k], "r2": deg["r2"][k]})
    o = deg["overall"]
    rows.append({"model": label, "scope": scope, "seed": seed, "horizon_s": "overall",
                 "rmse_deg": o["rmse"], "mae_deg": o["mae"], "r2": o["r2"]})
    return rows


def run_generalization(base_config="jmse/run_configs/lstm_id.yaml", seeds=None, smoke=False):
    out_dir = config.RESULTS_DIR / ("generalization_smoke" if smoke else "generalization")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(base_config)
    seeds = [0] if smoke else (seeds or list(config.SEEDS))

    runs = [{"regime": "id", "hold_hs": None, "hold_ton": None, "hold_real": None,
             "label": "ID", "scope": "ID"}]
    runs += generalization_folds()

    records = []
    for run in runs:
        for seed in seeds:
            cfg = {**base, "regime": run["regime"], "hold_hs": run["hold_hs"],
                   "hold_ton": run["hold_ton"], "hold_real": run.get("hold_real"), "seed": seed,
                   "name": f"gen_{run['label'].replace('=', '').replace(' ', '')}"}
            deg = train(cfg, smoke=smoke)
            print(f"{run['scope']:5s} {run['label']:7s} seed={seed} "
                  f"overall R2={deg['overall']['r2']:.4f} RMSE={deg['overall']['rmse']:.2f}deg")
            records.extend(_rows(run["label"], run["scope"], seed, deg))

    raw = pd.DataFrame.from_records(records)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)

    per_fold = _aggregate_folds(raw)
    per_fold.to_csv(out_dir / "per_fold.csv", index=False)
    t7 = _build_t7(raw)
    t7.to_csv(out_dir / "T7_generalization.csv", index=False)
    return out_dir, raw, per_fold, t7


def _aggregate_folds(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-(fold, horizon) mean/std over seeds; carries the scope label."""
    agg = stats.aggregate_seeds(raw.drop(columns=["scope"]))
    scope = raw.drop_duplicates("model").set_index("model")["scope"]
    agg.insert(1, "scope", agg["model"].map(scope))
    return agg


def _build_t7(raw: pd.DataFrame) -> pd.DataFrame:
    """Scope-level table (ID vs LOHO vs LOVO vs LORO): overall RMSE/R2 mean±std + R2 drop vs ID."""
    overall = raw[raw["horizon_s"] == "overall"]
    g = overall.groupby("scope")
    t7 = g.agg(rmse_deg_mean=("rmse_deg", "mean"), rmse_deg_std=("rmse_deg", "std"),
               r2_mean=("r2", "mean"), r2_std=("r2", "std"),
               n_folds=("model", "nunique"), n_runs=("r2", "size")).reset_index()
    t7[["rmse_deg_std", "r2_std"]] = t7[["rmse_deg_std", "r2_std"]].fillna(0.0)
    # delta_r2_vs_id is the degradation relative to the ID baseline. If the ID row is
    # missing (caller passed only OOD folds) the delta is NaN by definition -- warn rather
    # than silently emitting NaN, since this table is normally built with ID present.
    if (t7["scope"] == "ID").any():
        id_r2 = float(t7.loc[t7["scope"] == "ID", "r2_mean"].iloc[0])
    else:
        id_r2 = np.nan
        warnings.warn("_build_t7: no ID baseline row present; delta_r2_vs_id will be NaN.",
                      stacklevel=2)
    t7["delta_r2_vs_id"] = t7["r2_mean"] - id_r2
    order = {"ID": 0, "LOHO": 1, "LOVO": 2, "LORO": 3}
    return t7.sort_values("scope", key=lambda s: s.map(order).fillna(99)).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="jmse/run_configs/lstm_id.yaml")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out_dir, raw, per_fold, t7 = run_generalization(args.config, args.seeds, smoke=args.smoke)
    print("\n=== T7: generalization (ID vs LOHO vs LOVO) ===")
    print(t7.to_string(index=False))
    print(f"wrote {out_dir / 'T7_generalization.csv'}")
    if not args.no_plots:
        from jmse.plots.generalization import plot_F9
        print("figure:", plot_F9(out_dir))


if __name__ == "__main__":
    main()
