"""Out-of-distribution UQ calibration (P7, C1xC4) -> T16.

A predictive interval that is calibrated in-distribution is not guaranteed to stay calibrated when
the model extrapolates. This runner trains the three UQ methods on the training conditions of an
OOD fold (their aleatoric term estimated, as in deployment, from in-distribution validation
residuals) and scores PICP@90 / MPIW / CRPS / point-RMSE on the held-out condition's test set. The
gap from the ID values in Table 5 quantifies how much the intervals degrade out of distribution.

By default the fold is Leave-One-Hs-Out at the most energetic sea state (Hs=7 m), the headline OOD
case in the generalization study. Heavy (trains 1+M+1 models per seed) -- GPU job.

Usage:  python -m jmse.uq.ood_uq [--hold-hs 7] [--hold-ton N] [--seeds 0 1 2] [--smoke]
"""
import argparse

import pandas as pd

from jmse import config as cfgmod
from jmse.data.windowing import build_combined_holdout, build_ood_arrays
from jmse.eval import stats
from jmse.train import load_config
from jmse.uq.run import T5_METRICS, run_uq


def run_ood_uq(hold_hs=7.0, hold_ton=None, seeds=(0, 1, 2), smoke=False, combined=False,
               base_config="jmse/run_configs/uq_id.yaml"):
    cfg = load_config(base_config)
    label = (f"Hs{hold_hs:g}x{hold_ton}t" if combined
             else (f"Hs{hold_hs:g}" if hold_hs else f"{hold_ton}t"))
    out_base = cfgmod.RESULTS_DIR / ("ood_uq_smoke" if smoke else "ood_uq") / label
    out_base.mkdir(parents=True, exist_ok=True)
    if combined:
        arrays = build_combined_holdout(hold_hs=hold_hs, hold_ton=hold_ton)
    else:
        arrays = build_ood_arrays(hold_hs=hold_hs, hold_ton=hold_ton)

    per_seed = []
    n_members = 2 if smoke else len(cfg.get("ensemble_seeds", list(cfgmod.SEEDS)))
    for s in seeds:
        ens = [s * 100 + i for i in range(n_members)]
        _, t5 = run_uq(cfg, smoke=smoke, seed=s, ensemble_seeds=ens,
                       out_dir=out_base / f"seed{s}", arrays=arrays)
        per_seed.append(t5.assign(seed=s))
    raw = pd.concat(per_seed, ignore_index=True)
    raw.to_csv(out_base / "T16_ood_uq_by_seed.csv", index=False)
    meanstd = stats.mean_std_over_seeds(raw, ["method"], T5_METRICS)
    meanstd.insert(0, "fold", label)
    meanstd.to_csv(out_base / "T16_ood_uq_meanstd.csv", index=False)
    return out_base, meanstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-hs", type=float, default=7.0)
    ap.add_argument("--hold-ton", type=int, default=None)
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_base, ms = run_ood_uq(hold_hs=args.hold_hs, hold_ton=args.hold_ton,
                              seeds=args.seeds, smoke=args.smoke, combined=args.combined)
    cols = ["fold", "method", "picp_mean", "picp_std", "mpiw_deg_mean", "crps_deg_mean",
            "point_rmse_deg_mean"]
    print(ms[cols].to_string(index=False))
    print(f"wrote {out_base / 'T16_ood_uq_meanstd.csv'}")


if __name__ == "__main__":
    main()
