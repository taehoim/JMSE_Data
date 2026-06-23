"""Statistical-rigor orchestrator (P7) -> T13: record-level bootstrap CIs and per-condition skill.

Consumes the saved per-window benchmark predictions (results/benchmark/preds/<model>.npz, in
build_id_arrays test order) and the record (group) identity, and reports for the headline models:
  - overall R^2 and RMSE with a 95% record (cluster) bootstrap CI, alongside the pooled (micro)
    vs record-balanced (macro) averages -- the honest sampling uncertainty over the 15 records;
  - per-sea-state and per-tonnage R^2 with record-bootstrap CIs (condition-resolved skill).

No model training -- everything is computed from the committed predictions.
Usage:  python -m jmse.eval.statrigor [--models lstm gru] [--n-boot 2000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jmse import config
from jmse.data.windowing import build_id_arrays
from jmse.eval import bootstrap as bs

PREDS = config.RESULTS_DIR / "benchmark" / "preds"


def _conditions(groups, group_keys):
    ton = np.array([group_keys[g][0] for g in groups], float)
    hs = np.array([group_keys[g][1] for g in groups], float)
    return ton, hs


def run_statrigor(models=("lstm", "gru"), n_boot=2000, alpha=0.05, seed=0, out_dir=None):
    out_dir = Path(out_dir) if out_dir else config.RESULTS_DIR / "benchmark"
    d = build_id_arrays()
    groups, gk = d["group_test"], d["group_keys"]
    ton, hs = _conditions(groups, gk)

    overall_rows, cond_rows = [], []
    for model in models:
        z = np.load(PREDS / f"{model}.npz")
        y, yhat = z["y_true"], z["y_pred"]
        assert len(y) == len(groups), f"{model}: preds/group misalignment"
        mm_r2 = bs.macro_micro_ci(y, yhat, groups, bs.overall_r2, n_boot, alpha, seed)
        r2_ci = mm_r2["micro"]
        rmse_ci = bs.group_bootstrap_ci(y, yhat, groups, bs.overall_rmse_deg, n_boot, alpha, seed)
        overall_rows.append({
            "model": model,
            "r2_micro": r2_ci["point"], "r2_micro_lo": r2_ci["lo"], "r2_micro_hi": r2_ci["hi"],
            "r2_macro": mm_r2["macro"]["point"], "r2_macro_lo": mm_r2["macro"]["lo"],
            "r2_macro_hi": mm_r2["macro"]["hi"],
            "rmse_deg": rmse_ci["point"], "rmse_lo": rmse_ci["lo"], "rmse_hi": rmse_ci["hi"],
        })
        for cond, name in ((hs, "Hs"), (ton, "tonnage")):
            t = bs.per_condition_ci(y, yhat, cond, groups, bs.overall_r2, name, n_boot, alpha, seed)
            t.insert(0, "model", model)
            t = t.rename(columns={name: "level"}).assign(condition=name)
            cond_rows.append(t)

    overall = pd.DataFrame(overall_rows)
    cond = pd.concat(cond_rows, ignore_index=True)[
        ["model", "condition", "level", "point", "lo", "hi", "n_groups", "n"]]
    overall.to_csv(out_dir / "T13_bootstrap_overall.csv", index=False)
    cond.to_csv(out_dir / "T13_bootstrap_by_condition.csv", index=False)
    return out_dir, overall, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "gru"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out_dir, overall, cond = run_statrigor(models=tuple(args.models), n_boot=args.n_boot)
    pd.set_option("display.width", 200)
    print("== overall (record bootstrap CIs) =="); print(overall.to_string(index=False))
    print("== by condition =="); print(cond.to_string(index=False))
    print(f"wrote T13_* under {out_dir}")


if __name__ == "__main__":
    main()
