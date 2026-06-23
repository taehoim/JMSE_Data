"""Event-based early-warning orchestrator (P6, C2 rigor) -> T12.

Re-scores the saved alarm streams (results/earlywarning[/seed<s>]/scores.npz) at the
operational *event* level rather than the per-cell level of Table 6, and---critically---fixes
the optimistic operating point of the original analysis. Here the alarm threshold alpha is
selected on a held-out *calibration* partition (the earlier temporal half of every test group)
to meet the FPR<=0.10 budget, frozen, and only then applied to the disjoint *evaluation* half on
which all event metrics are computed. Selecting and evaluating alpha on disjoint data removes the
threshold-on-the-test-set leakage.

For each danger threshold and each alarm (point / probabilistic / non-learned trend baseline) it
reports, on the evaluation half: detection (recall) and missed-event rates, false-alert episodes
per hour, event-level precision, and lead-time quantiles (p10/p50/p90), pooled across groups and
aggregated mean +/- std over seeds.

Usage:  python -m jmse.earlywarning.events_run [--seeds 0 1 2 3 4] [--refractory 5] [--smoke]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jmse import config
from jmse.earlywarning import events as ev
from jmse.earlywarning import roc
from jmse.eval import stats
from jmse.uq import conformal as cf

ALARMS = ("point", "prob", "naive")
T12_METRICS = ["detection_rate", "missed_event_rate", "false_episodes_per_hour",
               "precision", "lead_mean_s", "lead_p10_s", "lead_p50_s", "lead_p90_s"]


def _alarm_score(z, name, theta_deg):
    if name == "prob":
        return z[f"prob_{int(theta_deg)}"]
    return z[name]


def score_seed(z, theta_deg, theta_rad, fpr, refractory, horizon, fs_hz=1.0):
    """Event metrics for one seed's scores.npz, alpha selected on the calibration half."""
    y_true, groups = z["y_true"], z["group_test"] if "group_test" in z else z["groups"]
    cal_mask = cf.first_half_mask(groups)
    eval_mask = ~cal_mask
    labels_cells = (y_true > theta_rad).astype(int)
    rows = []
    for name in ALARMS:
        score = _alarm_score(z, name, theta_deg)
        # alpha selected on the calibration cells to meet the false-alarm budget, then frozen
        lc, sc = labels_cells[cal_mask], score[cal_mask]
        if lc.any() and (~lc.astype(bool)).any():
            alpha = roc.operating_point_at_fpr(lc.ravel(), sc.ravel(), fpr)["threshold"]
        else:
            alpha = np.inf
        # event metrics on the disjoint evaluation half, pooled across groups
        raws = []
        for g in np.unique(groups[eval_mask]):
            gm = eval_mask & (groups == g)
            lab_ts = (y_true[gm][:, 0] > theta_rad).astype(int)        # actual heel timeline (t+1)
            raws.append(ev.event_raw(lab_ts, score[gm], alpha, horizon, refractory))
        m = ev.pool_event_metrics(raws, fs_hz)
        m.update({"threshold_deg": theta_deg, "alarm": name, "alpha": float(alpha)})
        rows.append(m)
    return pd.DataFrame(rows)


def run_events(seeds=(0, 1, 2, 3, 4), smoke=False, refractory=0, out_dir=None):
    base = config.RESULTS_DIR / ("earlywarning_smoke" if smoke else "earlywarning")
    out_dir = Path(out_dir) if out_dir else base
    fpr = config.EARLY_WARNING_FPR
    horizon = config.HORIZON
    per_seed = []
    for s in seeds:
        z = np.load(base / f"seed{s}" / "scores.npz")
        for td, tr in zip(config.DANGER_THRESHOLDS_DEG, config.DANGER_THRESHOLDS_RAD):
            per_seed.append(score_seed(z, td, tr, fpr, refractory, horizon).assign(seed=s))
    raw = pd.concat(per_seed, ignore_index=True)
    raw.to_csv(out_dir / "T12_event_by_seed.csv", index=False)
    meanstd = stats.mean_std_over_seeds(raw, ["threshold_deg", "alarm"], T12_METRICS,
                                        passthrough_cols=["n_events", "n_episodes"])
    meanstd.to_csv(out_dir / "T12_event_metrics.csv", index=False)
    return out_dir, meanstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--refractory", type=int, default=0,
                    help="alert-suppression period (s); 0 = one episode per maximal fire run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir, ms = run_events(seeds=args.seeds, smoke=args.smoke, refractory=args.refractory)
    cols = ["threshold_deg", "alarm", "detection_rate_mean", "missed_event_rate_mean",
            "false_episodes_per_hour_mean", "precision_mean", "lead_p50_s_mean"]
    print(ms[cols].to_string(index=False))
    print(f"wrote {out_dir / 'T12_event_metrics.csv'}")


if __name__ == "__main__":
    main()
