"""k-of-n hysteresis / evidence-accumulation alarm debounce (R3.7).

Reviewer R3.7 asks, concretely, how an onboard system would curb the false-alert burden:
"discuss what false-alert rate is operationally acceptable and how alert suppression, temporal
hysteresis, or evidence accumulation would be implemented in a real onboard system." This module
implements one such mechanism and quantifies its trade-off on the saved early-warning streams.

The mechanism is a k-of-n debounce, the standard evidence accumulator a watch-keeping alarm uses:
the alarm RAISES at time t only when at least k of the last n raw per-step decisions are above the
operating threshold (window = steps t-n+1..t inclusive, t itself always counted). This suppresses
isolated single-step spikes (the dominant false-alert source) while a sustained exceedance still
raises -- only delayed by the latch, exactly k-1 steps for a fresh sustained run. k=1,n=1 is the
identity (the raw alarm); n>=k>=1.

The sweep driver re-scores the saved alarm streams (results/earlywarning/seed<s>/scores.npz) at the
EVENT level for a grid of (k,n), reusing the calibration->evaluation alpha selection and the
event-counting / lead-time logic of jmse.earlywarning.events(_run); it never retrains a model. It
reports, pooled across records and aggregated over seeds, the false-alert episode rate and warning
lead time at each (k,n), giving the false-alert-vs-lead-time curve the manuscript needs.

Usage:  python -m jmse.earlywarning.hysteresis [--seeds 0 1 2 3 4] [--threshold 15]
Outputs under results/sensitivity/: T_hysteresis.csv, F_hysteresis.png.
"""
import argparse

import numpy as np
import pandas as pd

from jmse import config
from jmse.earlywarning import events as ev
from jmse.earlywarning import roc
from jmse.eval import stats
from jmse.uq import conformal as cf

# (k, n) grid for the sweep: n in {1,3,5,7,10}; k a sensible subset of 1..n (n=1 is the raw alarm).
GRID = [(1, 1),
        (1, 3), (2, 3), (3, 3),
        (1, 5), (2, 5), (3, 5), (5, 5),
        (1, 7), (3, 7), (5, 7), (7, 7),
        (1, 10), (3, 10), (5, 10), (7, 10), (10, 10)]

H_METRICS = ["detection_rate", "missed_event_rate", "false_alert_rate",
             "false_episodes_per_hour", "precision", "lead_time_s",
             "lead_p10_s", "lead_p50_s", "lead_p90_s"]


def k_of_n(series, k: int, n: int, alpha: float = None) -> np.ndarray:
    """Apply a k-of-n debounce to a per-timestep alarm series -> debounced boolean series.

    The alarm latches at step t when at least k of the last n raw decisions (steps t-n+1..t,
    t included) are True; the leading n-1 steps use the (shorter) available window. With a score
    series, pass `alpha` to threshold it (raw decision = series >= alpha) first; otherwise the
    series is cast to boolean. k=1,n=1 reproduces the raw alarm exactly (identity). Requires
    n >= k >= 1.
    """
    if k < 1 or n < 1 or k > n:
        raise ValueError(f"need n >= k >= 1; got k={k}, n={n}")
    s = np.asarray(series)
    raw = (s >= alpha) if alpha is not None else s.astype(bool)
    if raw.size == 0:
        return raw.astype(bool)
    # sliding count of True over the trailing window of length n via a cumulative sum
    csum = np.concatenate([[0], np.cumsum(raw.astype(int))])     # csum[i] = sum raw[:i]
    idx = np.arange(1, raw.size + 1)
    lo = np.maximum(0, idx - n)
    count = csum[idx] - csum[lo]                                  # True count in (t-n, t]
    return count >= k


def _fires_kofn(score, alpha, k, n):
    """Raw any-horizon decision (max over horizons >= alpha) passed through the k-of-n debounce."""
    raw = np.asarray(score, float).max(axis=1) >= alpha
    return k_of_n(raw, k, n)


def score_seed(z, theta_deg, theta_rad, fpr, horizon, grid=GRID, alarm="prob", fs_hz=1.0):
    """Event metrics for one seed across the (k,n) grid; alpha selected on the calibration half.

    Mirrors jmse.earlywarning.events_run.score_seed: alpha is chosen on the earlier temporal half
    (cells) to meet the FPR budget, frozen, then the debounced decision is evaluated on the disjoint
    later half, pooled across records. Only the probabilistic alarm is debounced by default (the
    deployed channel); pass `alarm` to sweep another.
    """
    y_true = z["y_true"]
    groups = z["group_test"] if "group_test" in z else z["groups"]
    score = z[f"prob_{int(theta_deg)}"] if alarm == "prob" else z[alarm]
    cal_mask = cf.first_half_mask(groups)
    eval_mask = ~cal_mask

    labels_cells = (y_true > theta_rad).astype(int)
    lc, sc = labels_cells[cal_mask], score[cal_mask]
    if lc.any() and (~lc.astype(bool)).any():
        alpha = roc.operating_point_at_fpr(lc.ravel(), sc.ravel(), fpr)["threshold"]
    else:
        alpha = np.inf

    rows = []
    for (k, n) in grid:
        raws = []
        for g in np.unique(groups[eval_mask]):
            gm = eval_mask & (groups == g)
            lab_ts = (y_true[gm][:, 0] > theta_rad).astype(int)       # actual heel timeline (t+1)
            fires = _fires_kofn(score[gm], alpha, k, n)
            raws.append(ev.event_raw_from_fires(lab_ts, fires, horizon))
        m = ev.pool_event_metrics(raws, fs_hz)
        # episode-level false-alert fraction = n_false_episodes / n_total_episodes = 1 - precision
        # (the operator-facing burden is also reported in absolute terms as false_episodes_per_hour)
        m["false_alert_rate"] = (1.0 - m["precision"]) if not np.isnan(m["precision"]) else float("nan")
        m["lead_time_s"] = m["lead_mean_s"]
        m.update({"threshold_deg": theta_deg, "alarm": alarm,
                  "k": k, "n": n, "alpha": float(alpha)})
        rows.append(m)
    return pd.DataFrame(rows)


def run_hysteresis(seeds=(0, 1, 2, 3, 4), threshold_deg=15, grid=GRID, alarm="prob",
                   base=None, out_dir=None):
    """Sweep the (k,n) hysteresis grid over the saved early-warning streams, aggregate over seeds.

    Consumes results/earlywarning/seed<s>/scores.npz (written by jmse.earlywarning.run); it does
    not retrain. Writes results/sensitivity/T_hysteresis.csv with the per-(k,n) false-alert rate
    and lead time aggregated mean+/-std over seeds. Raises a clear error if the saved streams are
    absent (the full sweep runs in Phase 3 after the early-warning batch).
    """
    base = base if base is not None else config.RESULTS_DIR / "earlywarning"
    out_dir = out_dir if out_dir is not None else config.RESULTS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    th = float(np.radians(threshold_deg))
    H = config.HORIZON
    fpr = config.EARLY_WARNING_FPR

    per_seed = []
    for s in seeds:
        path = base / f"seed{s}" / "scores.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"saved early-warning scores not found: {path}. Run the early-warning batch first, "
                f"e.g. `python3 -m jmse.earlywarning.run --method quantile --seeds {' '.join(map(str, seeds))}`."
            )
        z = np.load(path)
        per_seed.append(score_seed(z, threshold_deg, th, fpr, H, grid, alarm).assign(seed=s))
    raw = pd.concat(per_seed, ignore_index=True)
    raw.to_csv(out_dir / "T_hysteresis_by_seed.csv", index=False)

    meanstd = stats.mean_std_over_seeds(raw, ["k", "n"], H_METRICS,
                                        passthrough_cols=["threshold_deg", "alarm", "alpha"])
    # collapse to the tidy single-value table the manuscript/plot consume (means + std + n_seeds)
    out = stats.mean_only(meanstd, H_METRICS)
    out["n_seeds"] = meanstd["n_seeds"]
    for m in ("false_alert_rate", "false_episodes_per_hour", "lead_time_s", "detection_rate"):
        out[f"{m}_std"] = meanstd[f"{m}_std"]
    out = out.sort_values(["n", "k"]).reset_index(drop=True)
    out.to_csv(out_dir / "T_hysteresis.csv", index=False)
    return out_dir, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--threshold", type=int, default=15)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    out_dir, out = run_hysteresis(seeds=args.seeds, threshold_deg=args.threshold)
    cols = ["k", "n", "detection_rate", "false_episodes_per_hour", "false_alert_rate",
            "lead_time_s", "precision"]
    pd.set_option("display.width", 200)
    print(out[cols].to_string(index=False))
    print(f"wrote {out_dir / 'T_hysteresis.csv'}")
    if not args.no_plot:
        from jmse.plots.hysteresis import plot_F_hysteresis
        print("figure:", plot_F_hysteresis(out_dir))


if __name__ == "__main__":
    main()
