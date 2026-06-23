"""Operating-point sensitivity of the event-based alarm (rigor follow-up) -> T22 + F14.

The event-based table fixes one false-alarm budget (FPR<=0.10) and one refractory (5 s). Reviewers
rightly ask how the operator-facing burden moves with those knobs. This sweep varies the FPR budget
and the refractory period for the probabilistic alarm at the 15-degree threshold (the most frequent
events) and reports event detection rate, false-alert episodes per hour, and event-level precision,
each pooled across records and averaged over seeds. It quantifies the detection-vs-burden trade-off a
deployment would tune, using only the saved alarm scores (no retraining).

Usage:  python -m jmse.earlywarning.sensitivity [--seeds 0 1 2 3 4] [--threshold 15]
"""
import argparse

import numpy as np
import pandas as pd

from jmse import config
from jmse.earlywarning.events_run import score_seed

FPRS = [0.05, 0.10, 0.20]
REFRACTORIES = [0, 3, 5, 10]


COST_FPRS = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
COST_ALARMS = ("prob", "point", "naive")


def cost_curve(seeds=(0, 1, 2, 3, 4), threshold_deg=15, fprs=COST_FPRS, refractory=0,
               base=None, out_dir=None):
    """Operating curve: event detection / precision vs false-alert episodes per hour.

    Sweeps a fine false-alarm budget for each alarm at no suppression and reports, pooled across
    records and averaged over seeds, the false-alert rate against detection and precision. This is
    the safety-system operating curve the FPR-fixed table cannot show.
    """
    base = base if base is not None else config.RESULTS_DIR / "earlywarning"
    out_dir = out_dir if out_dir is not None else config.RESULTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    th = float(np.radians(threshold_deg))
    H = config.HORIZON
    rows = []
    for fpr in fprs:
        per_seed = {a: [] for a in COST_ALARMS}
        for s in seeds:
            z = np.load(base / f"seed{s}" / "scores.npz")
            df = score_seed(z, threshold_deg, th, fpr, refractory, H).set_index("alarm")
            for a in COST_ALARMS:
                per_seed[a].append(df.loc[a])
        for a in COST_ALARMS:
            sp = pd.DataFrame(per_seed[a])
            rows.append({"alarm": a, "fpr_budget": fpr,
                         "false_episodes_per_hour": sp["false_episodes_per_hour"].mean(),
                         "detection_rate": sp["detection_rate"].mean(),
                         "precision": sp["precision"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "cost_curve.csv", index=False)
    return out_dir, out


def run_sensitivity(seeds=(0, 1, 2, 3, 4), threshold_deg=15, fprs=FPRS, refractories=REFRACTORIES,
                    base=None, out_dir=None):
    base = base if base is not None else config.RESULTS_DIR / "earlywarning"
    out_dir = out_dir if out_dir is not None else config.RESULTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    th = float(np.radians(threshold_deg))
    H = config.HORIZON

    rows = []
    for fpr in fprs:
        for refr in refractories:
            per_seed = []
            for s in seeds:
                z = np.load(base / f"seed{s}" / "scores.npz")
                df = score_seed(z, threshold_deg, th, fpr, refr, H)
                per_seed.append(df[df["alarm"] == "prob"].iloc[0])
            sp = pd.DataFrame(per_seed)
            rows.append({
                "fpr_budget": fpr, "refractory_s": refr,
                "detection_rate": sp["detection_rate"].mean(),
                "detection_rate_std": sp["detection_rate"].std(ddof=1),
                "false_episodes_per_hour": sp["false_episodes_per_hour"].mean(),
                "false_episodes_per_hour_std": sp["false_episodes_per_hour"].std(ddof=1),
                "precision": sp["precision"].mean(), "precision_std": sp["precision"].std(ddof=1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "T22_sensitivity.csv", index=False)
    return out_dir, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--threshold", type=int, default=15)
    args = ap.parse_args()
    out_dir, out = run_sensitivity(seeds=args.seeds, threshold_deg=args.threshold)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"wrote {out_dir / 'T22_sensitivity.csv'}")
    _, cc = cost_curve(seeds=args.seeds, threshold_deg=args.threshold)
    print(cc.to_string(index=False))
    print(f"wrote {out_dir / 'cost_curve.csv'}")


if __name__ == "__main__":
    main()
