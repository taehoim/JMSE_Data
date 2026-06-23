"""Ablation study (Task 27, C5 appendix): lookback, horizon, feature set, model size.

Each axis varies one design choice around the reference LSTM and reports the resulting
Xacc skill (degrees). Lookback/feature/size ablations keep horizon fixed so overall R^2 is
comparable; the horizon ablation reports t+1 and t+5 R^2 (overall is not comparable across
different horizon counts). Results feed the appendix tables and the F10b ablation figure.

Usage:
    python -m jmse.eval.ablation [--config jmse/run_configs/lstm_id.yaml] [--seeds 0 1] [--smoke]
Outputs under results/ablation[_smoke]/: ablation_<axis>.csv + ablation_summary.csv (+ figure).
"""
import argparse

import numpy as np
import pandas as pd

from jmse import config
from jmse.data.windowing import build_id_arrays
from jmse.eval.metrics import per_horizon_metrics, to_degrees
from jmse.train import DEVICE, fit_neural_on, load_config

FEATURE_SETS = {
    "all": config.FEATURES,                                   # u v w p q r phi theta
    "kinematic": ["u", "v", "w", "p", "q", "r"],              # velocities/rates, no angles
    "rates+angles": ["p", "q", "r", "phi", "theta"],
    "angles": ["phi", "theta"],                               # Xacc = sqrt(phi^2+theta^2)
}
LOOKBACKS = [10, 20, 30]
HORIZONS = [5, 10]
HIDDENS = [64, 128, 256]


def _eval(cfg, d, smoke):
    m, ysc = fit_neural_on(cfg, d["X_train"], d["y_train"], d["X_val"], d["y_val"], smoke=smoke)
    pred = m.predict(d["X_test"], device=DEVICE, target_scaler=ysc)
    return to_degrees(per_horizon_metrics(d["y_test"], pred))


def _row(axis, setting, seed, deg, h_index=None):
    """Overall row, or a specific horizon row when h_index is given."""
    if h_index is None:
        o = deg["overall"]
        return {"axis": axis, "setting": setting, "seed": seed, "metric": "overall",
                "rmse_deg": o["rmse"], "r2": o["r2"]}
    return {"axis": axis, "setting": setting, "seed": seed, "metric": f"t+{h_index+1}",
            "rmse_deg": deg["rmse"][h_index], "r2": deg["r2"][h_index]}


def run_ablations(base_config="jmse/run_configs/lstm_id.yaml", seeds=None, smoke=False):
    out_dir = config.RESULTS_DIR / ("ablation_smoke" if smoke else "ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(base_config)
    seeds = [0] if smoke else (seeds or [0, 1, 2])
    rows = []

    for seed in seeds:
        # lookback (horizon fixed) — overall comparable
        for L in LOOKBACKS:
            d = build_id_arrays(lookback=L)
            deg = _eval({**base, "seed": seed}, d, smoke)
            rows.append(_row("lookback", L, seed, deg))
        # feature subsets (horizon fixed)
        for name, feats in FEATURE_SETS.items():
            d = build_id_arrays(features=feats)
            deg = _eval({**base, "seed": seed}, d, smoke)
            rows.append(_row("features", name, seed, deg))
        # model size (hidden width)
        for hidden in HIDDENS:
            d = build_id_arrays()
            deg = _eval({**base, "seed": seed, "hidden": hidden}, d, smoke)
            rows.append(_row("hidden", hidden, seed, deg))
        # horizon — overall not comparable, so report t+1 and t+5
        for H in HORIZONS:
            d = build_id_arrays(horizon=H)
            deg = _eval({**base, "seed": seed}, d, smoke)
            rows.append(_row("horizon", H, seed, deg, h_index=0))       # t+1
            rows.append(_row("horizon", H, seed, deg, h_index=4))       # t+5

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "ablation_raw.csv", index=False)
    summary = (raw.groupby(["axis", "setting", "metric"])
               .agg(rmse_deg_mean=("rmse_deg", "mean"), rmse_deg_std=("rmse_deg", "std"),
                    r2_mean=("r2", "mean"), r2_std=("r2", "std"), n=("seed", "nunique"))
               .reset_index())
    summary[["rmse_deg_std", "r2_std"]] = summary[["rmse_deg_std", "r2_std"]].fillna(0.0)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    return out_dir, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="jmse/run_configs/lstm_id.yaml")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out_dir, summary = run_ablations(args.config, args.seeds, smoke=args.smoke)
    print(summary.to_string(index=False))
    print(f"wrote {out_dir / 'ablation_summary.csv'}")
    if not args.no_plots:
        from jmse.plots.ablation import plot_ablation
        print("figure:", plot_ablation(out_dir))


if __name__ == "__main__":
    main()
