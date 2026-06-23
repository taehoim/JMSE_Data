"""F13: event-based early-warning comparison (P6, C2 rigor).

Grouped bars over the three exceedance thresholds for the point / probabilistic / naive alarms:
(a) event detection rate, (b) false-alert episodes per hour, (c) event-level precision, each with
seed std error bars. The probabilistic alarm's lower false-alert rate and higher precision at equal
detection is the operational form of the C2 result. Reads T12_event_metrics.csv from
jmse.earlywarning.events_run.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jmse.plots.style import add_grid

ALARMS = ["point", "prob", "naive"]
LABELS = {"point": "Point", "prob": "Probabilistic", "naive": "Naive (trend)"}
COLORS = {"point": "#DD8452", "prob": "#4C72B0", "naive": "#937860"}


def _grouped(ax, df, metric, title, ylabel):
    thr = sorted(df["threshold_deg"].unique())
    x = np.arange(len(thr))
    w = 0.26
    for j, a in enumerate(ALARMS):
        sub = df[df["alarm"] == a].set_index("threshold_deg").reindex(thr)
        ax.bar(x + (j - 1) * w, sub[f"{metric}_mean"], w, yerr=sub[f"{metric}_std"],
               capsize=2, label=LABELS[a], color=COLORS[a])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t)}$^\\circ$" for t in thr])
    ax.set_xlabel("Exceedance threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    add_grid(ax, axis="y")


def plot_cost_curve(sens_dir: Path, out_path: Path = None) -> Path:
    """Operating curve: event detection (a) and precision (b) vs false-alert episodes per hour,
    one line per alarm, swept over the false-alarm budget. Reads cost_curve.csv from
    jmse.earlywarning.sensitivity.cost_curve.
    """
    sens_dir = Path(sens_dir)
    df = pd.read_csv(sens_dir / "cost_curve.csv")
    out_path = Path(out_path) if out_path else sens_dir / "F14_cost_curve.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for a in ALARMS:
        sub = df[df["alarm"] == a].sort_values("false_episodes_per_hour")
        axes[0].plot(sub["false_episodes_per_hour"], sub["detection_rate"], marker="o",
                     label=LABELS[a], color=COLORS[a])
        axes[1].plot(sub["false_episodes_per_hour"], sub["precision"], marker="o",
                     label=LABELS[a], color=COLORS[a])
    axes[0].set_xlabel("False-alert episodes / hour")
    axes[0].set_ylabel("Event detection rate")
    axes[0].set_title("(a) Detection vs alert burden")
    axes[0].set_ylim(0, 1.02)
    add_grid(axes[0])
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xlabel("False-alert episodes / hour")
    axes[1].set_ylabel("Event-level precision")
    axes[1].set_title("(b) Precision vs alert burden")
    axes[1].set_ylim(0, 0.8)
    add_grid(axes[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def plot_F13(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "T12_event_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _grouped(axes[0], df, "detection_rate", "(a) Event detection rate", "Detection rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=8, loc="lower right")
    _grouped(axes[1], df, "false_episodes_per_hour", "(b) False-alert episodes / hour",
             "Episodes / hour")
    _grouped(axes[2], df, "precision", "(c) Event-level precision", "Precision")
    axes[2].set_ylim(0, 0.7)
    fig.tight_layout()
    path = out_dir / "F13_event.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
