"""Early-warning orchestrator (Task 22, C2): probabilistic vs point alarm -> T6 + F7 + F8.

Consumes the P2 UQ predictive distributions (results/uq[_smoke]/preds/<method>.npz) and,
for each danger threshold theta in config.DANGER_THRESHOLDS, scores two alarms against the
strict-'>' exceedance labels on the ID test set:
  - point alarm : score = point/median forecast  (what a deterministic model raises)
  - prob alarm  : score = P(Xacc(t+k) > theta)    (uncertainty-aware)
Reports ROC-AUC / PR-AUC / best-F1 / effective horizon for both, and the mean warning
lead time at a *matched false-alarm rate* (the C2 punchline). Aligns to the test windows
via build_id_arrays (same order as the saved preds), using group ids for lead time.

Usage:
    python -m jmse.earlywarning.run [--method quantile|ensemble|mc_dropout] [--smoke]
Outputs under results/earlywarning[_smoke]/: T6_earlywarning.csv, per_horizon_auc.csv, F7, F8.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jmse import config
from jmse.data.windowing import build_id_arrays
from jmse.earlywarning import alarm, leadtime, roc
from jmse.eval import stats

# metrics aggregated as mean +/- std across seeds (S1); identifiers carried through unchanged.
T6_METRICS = ["roc_auc", "pr_auc", "best_f1", "effective_horizon_s",
              "achieved_fpr", "detection_rate", "mean_lead_s"]
T6_PASSTHROUGH = ["prevalence", "budget_fpr", "n_events"]


def predictive(method: str, z):
    """Return (point_score (N,H), prob_fn(theta)->(N,H)) for a saved UQ preds bundle."""
    if method == "quantile":
        q, taus = z["quantiles"], z["taus"]
        median = q[:, :, int(np.argmin(np.abs(taus - 0.5)))]
        return median, (lambda theta: alarm.prob_exceed_quantile(q, taus, theta))
    mean, std = z["mean"], z["std"]                      # ensemble / mc_dropout (Gaussian)
    return mean, (lambda theta: alarm.prob_exceed_gaussian(mean, std, theta))


def _per_horizon_auc(label, score):
    return np.array([roc.roc_auc(label[:, k], score[:, k]) for k in range(label.shape[1])])


def run_earlywarning(method="quantile", smoke=False, uq_dir=None, out_dir=None):
    uq_dir = Path(uq_dir) if uq_dir else config.RESULTS_DIR / ("uq_smoke" if smoke else "uq")
    out_dir = Path(out_dir) if out_dir else config.RESULTS_DIR / ("earlywarning_smoke" if smoke else "earlywarning")
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(uq_dir / "preds" / f"{method}.npz")
    d = build_id_arrays()
    y_true, groups = z["y_true"], d["group_test"]
    assert len(y_true) == len(groups) == len(d["y_test"]), "preds/test-window misalignment"
    point_score, prob_fn = predictive(method, z)
    # S2: learning-free domain-rule alarm -- linear-trend extrapolation of the observed angle,
    # scored against the same labels/budget as the learned alarms (no model, seed-invariant).
    naive_score = alarm.trend_forecast(d["yhist_test"], y_true.shape[1])

    rows, per_h_rows = [], []
    for theta_deg, theta in zip(config.DANGER_THRESHOLDS_DEG, config.DANGER_THRESHOLDS_RAD):
        label = alarm.exceedance_labels(y_true, theta)
        prob = prob_fn(theta)
        prevalence = float(label.mean())

        # all alarms thresholded to the SAME false-alarm budget -> lead time at equal cost
        budget = config.EARLY_WARNING_FPR
        for name, score in (("point", point_score), ("prob", prob), ("naive", naive_score)):
            auc_h = _per_horizon_auc(label, score)
            eff_h = leadtime.effective_horizon(auc_h)
            f1, thr = roc.best_f1(label.ravel(), score.ravel())

            if label.any() and (~label).any():
                alpha = roc.operating_point_at_fpr(label.ravel(), score.ravel(), budget)["threshold"]
            else:
                alpha = np.inf                               # no positives/negatives -> never fire
            lt = leadtime.lead_times(y_true, score, groups, theta, alpha)

            rows.append({
                "threshold_deg": theta_deg, "alarm": name, "prevalence": round(prevalence, 4),
                "roc_auc": roc.roc_auc(label.ravel(), score.ravel()),
                "pr_auc": roc.pr_auc(label.ravel(), score.ravel()),
                "best_f1": f1, "effective_horizon_s": eff_h,
                "budget_fpr": budget, "achieved_fpr": round(lt["fpr"], 4),
                "detection_rate": lt["detection_rate"],
                "mean_lead_s": lt["mean_lead_s"], "n_events": lt["n_events"],
            })
            for k, a in enumerate(auc_h):
                per_h_rows.append({"threshold_deg": theta_deg, "alarm": name,
                                   "horizon_s": k + 1, "roc_auc": a})

    t6 = pd.DataFrame(rows)
    t6.to_csv(out_dir / "T6_earlywarning.csv", index=False)
    pd.DataFrame(per_h_rows).to_csv(out_dir / "per_horizon_auc.csv", index=False)
    np.savez(out_dir / "scores.npz", y_true=y_true, groups=groups, point=point_score,
             naive=naive_score,
             **{f"prob_{int(td)}": prob_fn(th)
                for td, th in zip(config.DANGER_THRESHOLDS_DEG, config.DANGER_THRESHOLDS_RAD)})
    return out_dir, t6


def run_earlywarning_multiseed(method="quantile", seeds=(0, 1, 2), smoke=False):
    """Score the early-warning table over per-seed UQ artifacts and report mean +/- std (S1).

    Reads each seed's predictive distribution from <uq>/seed<s>/preds (written by
    `jmse.uq.run --seeds ...`), scores the table per seed, then aggregates over seeds. The
    point/probabilistic alarms vary with the seed-specific model; the naive domain-rule alarm is
    seed-invariant (std 0). Writes T6_earlywarning_by_seed.csv (raw), T6_earlywarning_meanstd.csv
    (mean +/- std for the manuscript), and T6_earlywarning.csv (means, for the existing plots)."""
    base_uq = config.RESULTS_DIR / ("uq_smoke" if smoke else "uq")
    base_out = config.RESULTS_DIR / ("earlywarning_smoke" if smoke else "earlywarning")
    base_out.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for s in seeds:
        _, t6 = run_earlywarning(method, smoke=smoke, uq_dir=base_uq / f"seed{s}",
                                 out_dir=base_out / f"seed{s}")
        per_seed.append(t6.assign(seed=s))
    raw = pd.concat(per_seed, ignore_index=True)
    raw.to_csv(base_out / "T6_earlywarning_by_seed.csv", index=False)

    meanstd = stats.mean_std_over_seeds(raw, ["threshold_deg", "alarm"], T6_METRICS,
                                        passthrough_cols=T6_PASSTHROUGH)
    meanstd.to_csv(base_out / "T6_earlywarning_meanstd.csv", index=False)
    stats.mean_only(meanstd, T6_METRICS).to_csv(base_out / "T6_earlywarning.csv", index=False)
    return base_out, meanstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="quantile", choices=["quantile", "ensemble", "mc_dropout"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="per-seed UQ dirs (uq/seed<s>) to aggregate mean+/-std; omit for single-seed")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    print(f"early-warning  method={args.method}  smoke={args.smoke}  seeds={args.seeds}")
    if args.seeds and len(args.seeds) > 1:
        out_dir, ms = run_earlywarning_multiseed(args.method, seeds=args.seeds, smoke=args.smoke)
        cols = ["threshold_deg", "alarm"] + [f"{m}_mean" for m in ("roc_auc", "pr_auc", "best_f1", "mean_lead_s")] \
            + [f"{m}_std" for m in ("roc_auc", "pr_auc", "best_f1", "mean_lead_s")]
        print(ms[cols].to_string(index=False))
        print(f"wrote {out_dir / 'T6_earlywarning_meanstd.csv'}")
        return
    out_dir, t6 = run_earlywarning(args.method, smoke=args.smoke)
    cols = ["threshold_deg", "alarm", "roc_auc", "pr_auc", "best_f1",
            "effective_horizon_s", "achieved_fpr", "detection_rate", "mean_lead_s"]
    print(t6[cols].to_string(index=False))
    print(f"wrote {out_dir / 'T6_earlywarning.csv'}")
    if not args.no_plots:
        from jmse.plots.earlywarning import plot_F7, plot_F8
        print("figures:", plot_F7(out_dir, method=args.method), plot_F8(out_dir))


if __name__ == "__main__":
    main()
