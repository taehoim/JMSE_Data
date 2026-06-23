"""Benchmark sweep: run every model across seeds and assemble the C3 results.

For each model config and each seed in `config.SEEDS`, this trains in-process and
records per-(model, seed, horizon) metrics into a tidy long-form CSV. Classical and
persistence baselines are deterministic, so they run once (seed 0). Per-window test
predictions for seed 0 of each model are saved for the paired significance test.

Usage:
    python -m jmse.sweep                         # full sweep (GPU recommended)
    python -m jmse.sweep --smoke                 # fast CPU infra check (2 epochs)
    python -m jmse.sweep --seeds 0 1 --models lstm gru
Outputs under results/benchmark[_smoke]/: raw_metrics.csv, preds/<model>.npz,
T4_benchmark*.csv, significance.csv (+ figures F4, F5 via plots.benchmark).
"""
import argparse

import pandas as pd

from jmse import config
from jmse.eval import stats
from jmse.eval.run import evaluate_id
from jmse.models.classical import is_classical
from jmse.train import build_arrays, load_config, train

# Benchmark set: persistence floor + 2 classical + 4 neural backbones.
BENCH = {
    "persistence": None,
    "ar": "jmse/run_configs/ar_id.yaml",
    "kalman": "jmse/run_configs/kalman_id.yaml",
    "tcn": "jmse/run_configs/tcn_id.yaml",
    "transformer": "jmse/run_configs/transformer_id.yaml",
    "gru": "jmse/run_configs/gru_id.yaml",
    "lstm": "jmse/run_configs/lstm_id.yaml",
}


def _rows_from_deg(model, seed, deg):
    rows = []
    for k in range(len(deg["r2"])):
        rows.append({"model": model, "seed": seed, "horizon_s": k + 1,
                     "rmse_deg": deg["rmse"][k], "mae_deg": deg["mae"][k], "r2": deg["r2"][k]})
    o = deg["overall"]
    rows.append({"model": model, "seed": seed, "horizon_s": "overall",
                 "rmse_deg": o["rmse"], "mae_deg": o["mae"], "r2": o["r2"]})
    return rows


def run_sweep(models, seeds, smoke=False):
    out_dir = config.RESULTS_DIR / ("benchmark_smoke" if smoke else "benchmark")
    preds_dir = out_dir / "preds"
    records = []

    for model in models:
        cfg_path = BENCH[model]
        deterministic = (model == "persistence") or is_classical(model)
        model_seeds = [seeds[0]] if deterministic else seeds

        for si, seed in enumerate(model_seeds):
            save_preds = preds_dir / f"{model}.npz" if si == 0 else None

            if model == "persistence":
                d = build_arrays({"regime": "id"})
                deg = evaluate_id("persistence", arrays=d, split="id", save_preds_path=save_preds)
            else:
                cfg = load_config(cfg_path)
                cfg["seed"] = seed
                deg = train(cfg, smoke=smoke, save_preds_path=save_preds)  # preds from this exact model
            print(f"{model:12s} seed={seed} overall R2={deg['overall']['r2']:.4f} "
                  f"RMSE={deg['overall']['rmse']:.2f}deg")
            records.extend(_rows_from_deg(model, seed, deg))

    raw = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    return out_dir, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(BENCH))
    ap.add_argument("--seeds", nargs="*", type=int, default=config.SEEDS)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_dir, raw = run_sweep(args.models, args.seeds, smoke=args.smoke)
    agg = stats.aggregate_seeds(raw)
    t4 = stats.write_t4(agg, out_dir)
    print(f"wrote {t4}")
    try:
        sig = stats.paired_wilcoxon_vs(out_dir / "preds", reference="lstm")
        sig.to_csv(out_dir / "significance.csv", index=False)
        print(f"wrote {out_dir / 'significance.csv'}")
    except FileNotFoundError as e:
        print(f"skip significance: {e}")

    if not args.no_plots:
        from jmse.plots.benchmark import plot_benchmark
        figs = plot_benchmark(out_dir)
        print("figures:", *[str(f) for f in figs])


if __name__ == "__main__":
    main()
