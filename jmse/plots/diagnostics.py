"""F11: inclination-signal diagnostics (reviewer 2.4) — ACF, PACF, PSD, and a raw-series excerpt.

Visualizes why a 1 Hz persistence baseline is negative-R^2: short-lag autocorrelation is weak and
the spectrum is broadband rather than a single sharp roll line, so the inclination is not trivially
predictable from its last value. Read from results/diagnostics/acf.csv plus the curated signal.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from jmse import config


def plot_F11(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    acf = pd.read_csv(out_dir / "acf.csv")
    from jmse.data.curate import load_curated
    df = load_curated()
    tgt = config.TARGET
    hs_list = sorted(df["Hs"].unique())
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, len(hs_list)))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for col, metric, title in [(0, "acf", "(a) Autocorrelation"), (1, "pacf", "(b) Partial autocorrelation")]:
        a = ax[0, col]
        for hs, c in zip(hs_list, colors):
            sub = acf[acf["group"] == f"Hs{int(hs)}"]
            a.plot(sub["lag"], sub[metric], color=c, marker="o", ms=2.5, lw=1, label=f"$H_s$={int(hs)} m")
        a.axhline(0.0, color="0.6", lw=0.8)
        a.set_xlabel("lag (s)"); a.set_ylabel(metric.upper()); a.set_title(title)
    ax[0, 0].legend(fontsize=8, loc="upper right")

    aP = ax[1, 0]
    for hs, c in zip(hs_list, colors):
        g = df[df["Hs"] == hs]
        psds, freqs = [], None
        for _, rec in g.groupby(["tonnage", "Hs"]):
            freqs, pxx = welch(rec.sort_values("time")[tgt].to_numpy(float), fs=1.0, nperseg=512)
            psds.append(pxx)
        aP.semilogy(freqs, np.mean(psds, axis=0), color=c, lw=1.3, label=f"$H_s$={int(hs)} m")
    aP.set_xlabel("frequency (Hz)"); aP.set_ylabel("PSD (deg$^2$/Hz)")
    aP.set_title("(c) Power spectral density")

    aR = ax[1, 1]
    for hs, c in zip(hs_list, colors):
        rec = df[(df["Hs"] == hs) & (df["tonnage"] == 30)].sort_values("time")
        seg = rec.iloc[1000:1300]
        aR.plot(seg["time"].to_numpy() - seg["time"].iloc[0],
                np.degrees(seg[tgt].to_numpy()), color=c, lw=0.9, label=f"$H_s$={int(hs)} m")
    aR.set_xlabel("time (s)"); aR.set_ylabel(r"$\theta_{\mathrm{TIA}}$ (deg)")
    aR.set_title("(d) Raw inclination excerpt (30 t)")

    fig.tight_layout()
    path = out_dir / "F11_diagnostics.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path
