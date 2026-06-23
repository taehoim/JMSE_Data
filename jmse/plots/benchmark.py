"""Benchmark figures F4 (per-horizon skill) and F5 (overall ranking) for C3.

Reads the sweep's aggregate (mean +/- std over seeds) and renders:
  F4: RMSE(deg) and R^2 vs forecast horizon, one line per model with +/-1 std band.
  F5: overall RMSE(deg) and R^2 bar charts with std error bars, model-ranked.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from jmse.eval import stats
from jmse.plots.style import add_grid


def _load_agg(out_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(out_dir / "raw_metrics.csv")
    return stats.aggregate_seeds(raw)


def _models_in(agg: pd.DataFrame):
    present = [m for m in stats.MODEL_ORDER if m in set(agg["model"])]
    present += [m for m in agg["model"].unique() if m not in present]
    return present


def plot_F4(agg: pd.DataFrame, out_dir: Path) -> Path:
    per_h = agg[agg["horizon_s"] != "overall"].copy()
    per_h["h"] = per_h["horizon_s"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in _models_in(agg):
        sub = per_h[per_h["model"] == model].sort_values("h")
        if sub.empty:
            continue
        label = stats.MODEL_LABELS.get(model, model)
        for ax, mean, std, ylab in (
            (axes[0], "rmse_deg_mean", "rmse_deg_std", "RMSE (deg)"),
            (axes[1], "r2_mean", "r2_std", "R$^2$"),
        ):
            ax.plot(sub["h"], sub[mean], marker="o", label=label)
            ax.fill_between(sub["h"], sub[mean] - sub[std], sub[mean] + sub[std], alpha=0.15)
            ax.set_xlabel("Forecast horizon (s)")
            ax.set_ylabel(ylab)
    axes[0].set_title("(a) Error growth with horizon")
    axes[1].set_title("(b) Skill decay with horizon")
    axes[1].axhline(0, ls="--", lw=0.8, color="grey")
    for ax in axes:
        add_grid(ax)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "F4_per_horizon.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def plot_F5(agg: pd.DataFrame, out_dir: Path) -> Path:
    overall = agg[agg["horizon_s"] == "overall"].copy()
    overall["_mk"] = overall["model"].map(stats._order_key)
    overall = overall.sort_values("_mk")
    labels = [stats.MODEL_LABELS.get(m, m) for m in overall["model"]]
    x = range(len(overall))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, overall["rmse_deg_mean"], yerr=overall["rmse_deg_std"], capsize=3, color="#4C72B0")
    axes[0].set_ylabel("Overall RMSE (deg)")
    axes[0].set_title("(a) Overall error (lower is better)")
    axes[1].bar(x, overall["r2_mean"], yerr=overall["r2_std"], capsize=3, color="#55A868")
    axes[1].axhline(0, ls="--", lw=0.8, color="grey")
    axes[1].set_ylabel("Overall R$^2$")
    axes[1].set_title("(b) Overall skill (higher is better)")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        add_grid(ax, axis="y")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "F5_overall_ranking.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def plot_benchmark(out_dir: Path):
    out_dir = Path(out_dir)
    agg = _load_agg(out_dir)
    return [plot_F4(agg, out_dir), plot_F5(agg, out_dir)]
