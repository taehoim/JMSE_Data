"""F2: Xacc (total inclination angle) distribution across sea states and vessels."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jmse import config
from jmse.data.curate import load_curated
from jmse.plots.style import add_grid

FIG_DIR = config.RESULTS_DIR / "figures"


def plot_xacc_distribution() -> Path:
    df = load_curated()
    df = df.assign(deg=np.degrees(df["Xacc"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) distribution by sea state
    data = [df.loc[df.Hs == hs, "deg"].values for hs in config.HS_VALUES]
    axes[0].violinplot(data, showmedians=True)
    axes[0].set_xticks(range(1, len(config.HS_VALUES) + 1))
    axes[0].set_xticklabels([f"Hs={hs} m" for hs in config.HS_VALUES])
    for d in config.DANGER_THRESHOLDS_DEG:
        axes[0].axhline(d, ls="--", lw=0.8, color="grey")
    axes[0].set_ylabel("Total inclination angle (deg)")
    axes[0].set_title("(a) Distribution by sea state")
    add_grid(axes[0], axis="y")

    # (b) >15deg exceedance rate by tonnage x sea state
    tons = sorted(df.tonnage.unique())
    width = 0.25
    for i, hs in enumerate(config.HS_VALUES):
        rates = [(df[(df.tonnage == t) & (df.Hs == hs)]["deg"] > 15).mean() * 100 for t in tons]
        axes[1].bar(np.arange(len(tons)) + i * width, rates, width, label=f"Hs={hs} m")
    axes[1].set_xticks(np.arange(len(tons)) + width)
    axes[1].set_xticklabels([f"{t} t" for t in tons])
    axes[1].set_ylabel(r"Exceedance rate, $\theta_{\mathrm{TIA}}>15^\circ$ (%)")
    axes[1].set_title("(b) Operational-threshold exceedance")
    axes[1].legend()
    add_grid(axes[1], axis="y")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "F2_xacc_distribution.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return out
