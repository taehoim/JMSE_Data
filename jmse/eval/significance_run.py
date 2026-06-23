"""Significance orchestrator (rigor follow-up) -> T21.

Formalizes the two headline comparisons the manuscript previously supported only with seed
mean +/- std:
  (1) probabilistic vs point alarm ROC-AUC, per exceedance threshold, from the saved alarm scores;
  (2) LSTM vs GRU and LSTM vs the strongest non-neural baseline (GBM) overall R^2, from the saved
      per-window benchmark predictions.
Each uses the record-level cluster bootstrap (primary, dependence-aware) and, for the AUC case,
the DeLong test (standard cross-check). The alarm comparison is run per seed and summarized so the
result is reported as both an effect size and a consistency-across-seeds count.

Usage:  python -m jmse.eval.significance_run [--seeds 0 1 2 3 4] [--n-boot 2000]
"""
import argparse

import numpy as np
import pandas as pd

from jmse import config
from jmse.data.windowing import build_id_arrays
from jmse.eval import significance as sig
from jmse.eval.bootstrap import overall_r2

BENCH_PREDS = config.RESULTS_DIR / "benchmark" / "preds"


def alarm_significance(seeds=(0, 1, 2, 3, 4), n_boot=2000):
    base = config.RESULTS_DIR / "earlywarning"
    rows = []
    for td in config.DANGER_THRESHOLDS_DEG:
        th = float(np.radians(td))
        per_seed = []
        for s in seeds:
            z = np.load(base / f"seed{s}" / "scores.npz")
            H = z["y_true"].shape[1]
            labels = (z["y_true"] > th).astype(int).ravel()
            point, prob = z["point"].ravel(), z[f"prob_{td}"].ravel()
            groups = np.repeat(z["groups"], H)
            cb = sig.auc_diff_cluster_bootstrap(labels, prob, point, groups, n_boot=n_boot, seed=s)
            _, _, _, dp = sig.delong_test(labels, prob, point)
            per_seed.append({"diff": cb["diff"], "lo": cb["lo"], "hi": cb["hi"],
                             "p_boot": cb["p"], "p_delong": dp})
        df = pd.DataFrame(per_seed)
        rows.append({
            "threshold_deg": td,
            "delta_auc_mean": df["diff"].mean(), "delta_auc_std": df["diff"].std(ddof=1),
            "ci_lo_seed0": df["lo"].iloc[0], "ci_hi_seed0": df["hi"].iloc[0],
            "p_boot_max": df["p_boot"].max(), "p_delong_max": df["p_delong"].max(),
            "n_sig_seeds": int((df["lo"] > 0).sum()), "n_seeds": len(seeds),
        })
    return pd.DataFrame(rows)


def model_significance(n_boot=5000):
    d = build_id_arrays()
    groups = d["group_test"]
    zl = np.load(BENCH_PREDS / "lstm.npz")
    y = zl["y_true"]
    rows = []
    for other in ("gru", "gbm"):
        zo = np.load(BENCH_PREDS / f"{other}.npz")
        out = sig.metric_diff_cluster_bootstrap(y, zl["y_pred"], zo["y_pred"], groups,
                                                overall_r2, n_boot=n_boot, seed=0)
        rows.append({"comparison": f"LSTM - {other.upper()}", "delta_r2": out["diff"],
                     "ci_lo": out["lo"], "ci_hi": out["hi"], "p_boot": out["p"]})
    return pd.DataFrame(rows)


def run(seeds=(0, 1, 2, 3, 4), n_boot=2000):
    out_dir = config.RESULTS_DIR / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)
    alarm = alarm_significance(seeds, n_boot)
    model = model_significance(max(n_boot, 5000))
    alarm.to_csv(out_dir / "T21_alarm_significance.csv", index=False)
    model.to_csv(out_dir / "T21_model_significance.csv", index=False)
    return out_dir, alarm, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out_dir, alarm, model = run(seeds=args.seeds, n_boot=args.n_boot)
    pd.set_option("display.width", 200)
    print("== alarm (prob - point) AUC significance =="); print(alarm.to_string(index=False))
    print("== model overall-R2 significance =="); print(model.to_string(index=False))
    print(f"wrote T21_* under {out_dir}")


if __name__ == "__main__":
    main()
