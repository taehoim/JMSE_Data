"""F_hysteresis: false-alert-vs-lead-time trade-off of the k-of-n alarm debounce (R3.7).

The k-of-n evidence accumulator trades alert burden against warning lead time: requiring more
of the last n steps to exceed (larger k) suppresses isolated false alerts but delays the latch,
shortening lead time. This figure draws that frontier from T_hysteresis.csv (one point per (k,n)
grid cell), with the operating curves grouped by window length n so the cost of each debounce
setting is read directly.

(a) warning lead time vs false-alert episodes per hour, one line per window length n;
(b) event detection rate vs false-alert episodes per hour (the burden-vs-recall view).
The raw alarm (k=1, n=1) is marked as the no-debounce reference.

Reads results/sensitivity/T_hysteresis.csv from jmse.earlywarning.hysteresis.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from jmse.plots.style import add_grid

# one colour per window length n; the raw alarm (1,1) is highlighted separately
N_COLORS = {1: "#937860", 3: "#4C72B0", 5: "#55A868", 7: "#DD8452", 10: "#C44E52"}


def plot_F_hysteresis(sens_dir: Path, out_path: Path = None) -> Path:
    sens_dir = Path(sens_dir)
    df = pd.read_csv(sens_dir / "T_hysteresis.csv")
    out_path = Path(out_path) if out_path else sens_dir / "F_hysteresis.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    burden = "false_episodes_per_hour"
    for n in sorted(df["n"].unique()):
        sub = df[df["n"] == n].sort_values("k")
        color = N_COLORS.get(int(n), "#666666")
        axes[0].plot(sub[burden], sub["lead_time_s"], marker="o", color=color, label=f"n={int(n)}")
        axes[1].plot(sub[burden], sub["detection_rate"], marker="o", color=color, label=f"n={int(n)}")

    raw = df[(df["k"] == 1) & (df["n"] == 1)]
    if len(raw):
        r = raw.iloc[0]
        axes[0].scatter([r[burden]], [r["lead_time_s"]], s=90, facecolors="none",
                        edgecolors="black", zorder=5, label="raw (k=1,n=1)")
        axes[1].scatter([r[burden]], [r["detection_rate"]], s=90, facecolors="none",
                        edgecolors="black", zorder=5, label="raw (k=1,n=1)")

    axes[0].set_xlabel("False-alert episodes / hour")
    axes[0].set_ylabel("Warning lead time (s)")
    axes[0].set_title("(a) Lead time vs alert burden")
    add_grid(axes[0])
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_xlabel("False-alert episodes / hour")
    axes[1].set_ylabel("Event detection rate")
    axes[1].set_title("(b) Detection vs alert burden")
    axes[1].set_ylim(0, 1.02)
    add_grid(axes[1])
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path
