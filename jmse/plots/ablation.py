"""Ablation figure (C5 appendix): overall R^2 across the four design axes.

Lookback, feature-set and model-size panels show overall R^2 (one bar per setting); the
horizon panel shows t+1 and t+5 R^2 side by side (overall is not comparable across horizon
counts). Error bars are +/-1 std over seeds.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bar(ax, sub, title, xlabel):
    sub = sub.copy()
    x = np.arange(len(sub))
    ax.bar(x, sub["r2_mean"], yerr=sub["r2_std"], capsize=3, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"], rotation=20, ha="right")
    ax.set_ylabel("R$^2$")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axhline(0, lw=0.8, color="grey")


def plot_ablation(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    s = pd.read_csv(out_dir / "ablation_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    look = s[(s.axis == "lookback") & (s.metric == "overall")].copy()
    look["label"] = look["setting"].astype(str) + " s"
    _bar(axes[0, 0], look.sort_values("setting", key=lambda c: c.astype(int)),
         "(a) Lookback window", "Lookback (s)")

    feat = s[(s.axis == "features") & (s.metric == "overall")].copy()
    feat["label"] = feat["setting"]
    order = {"all": 0, "rates+angles": 1, "kinematic": 2, "angles": 3}
    _bar(axes[0, 1], feat.sort_values("setting", key=lambda c: c.map(order)),
         "(b) Feature set", "Input features")

    hid = s[(s.axis == "hidden") & (s.metric == "overall")].copy()
    hid["label"] = hid["setting"].astype(str)
    _bar(axes[1, 0], hid.sort_values("setting", key=lambda c: c.astype(int)),
         "(c) Model size", "Hidden units")

    hor = s[s.axis == "horizon"].copy()
    hor["label"] = hor["setting"].astype(str) + "s/" + hor["metric"]
    _bar(axes[1, 1], hor.sort_values(["setting", "metric"]),
         "(d) Horizon (t+1, t+5 R$^2$)", "Max horizon / step")

    fig.suptitle("F10b  Ablations (overall R$^2$, mean $\\pm$ std over seeds)")
    fig.tight_layout()
    path = out_dir / "F10b_ablation.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
