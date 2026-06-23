"""Signal diagnostics for the inclination time series (reviewer 2.4).

Quantifies the temporal structure that makes a 1 Hz persistence baseline attain a negative R^2:
autocorrelation (ACF), partial autocorrelation (PACF via Levinson--Durbin), and the power
spectral density (PSD, Welch) with the dominant period. Run over the curated dataset, pooled and
per sea state, to show the low short-lag correlation is a property of the signal/sampling, not a
target-construction artefact, and to compare the 1 Hz rate against the motion's dominant period.

Usage:
    python -m jmse.eval.diagnostics      # -> results/diagnostics/{acf,psd,summary}.csv + F11
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

from jmse import config


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Biased sample autocorrelation for lags 0..nlags (acf[0]=1)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.concatenate([[1.0], np.zeros(nlags)])
    return np.array([np.dot(x[: n - k], x[k:]) / denom for k in range(nlags + 1)])


def pacf_levinson(acf_vals: np.ndarray) -> np.ndarray:
    """Partial autocorrelation (pacf[0]=1) from an ACF via the Levinson--Durbin recursion."""
    r = np.asarray(acf_vals, float)
    p = len(r) - 1
    pacf = np.zeros(p + 1)
    pacf[0] = 1.0
    if p < 1:
        return pacf
    phi = np.zeros((p + 1, p + 1))
    phi[1, 1] = r[1]
    pacf[1] = r[1]
    for k in range(2, p + 1):
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        kk = num / den if den != 0 else 0.0
        phi[k, k] = kk
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - kk * phi[k - 1, k - j]
        pacf[k] = kk
    return pacf


def dominant_period(x: np.ndarray, fs: float = 1.0) -> float:
    """Period (s) of the largest Welch PSD peak, ignoring the zero-frequency (DC) bin."""
    f, pxx = welch(np.asarray(x, float), fs=fs, nperseg=min(512, len(x)))
    mask = f > 0
    if not mask.any():
        return float("inf")
    fpeak = f[mask][np.argmax(pxx[mask])]
    return float(1.0 / fpeak) if fpeak > 0 else float("inf")


def run_diagnostics(nlags: int = 30, fs: float = 1.0):
    """Per-(Hs) and pooled ACF/PACF/PSD over the curated inclination signal -> tidy CSVs."""
    from jmse.data.curate import load_curated

    out_dir = config.RESULTS_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_curated()
    tgt = config.TARGET

    acf_rows, summ_rows = [], []
    groups = [("all", df)] + [(f"Hs{int(hs)}", g) for hs, g in df.groupby("Hs")]
    for label, g in groups:
        # concatenate per-record series (each record contiguous) for ACF; pool for PSD peak
        per_record_acf = []
        for _, rec in g.groupby(["tonnage", "Hs"]):
            s = rec.sort_values("time")[tgt].to_numpy(float)
            if len(s) > nlags + 1:
                per_record_acf.append(acf(s, nlags))
        a = np.mean(per_record_acf, axis=0)
        p = pacf_levinson(a)
        for k in range(nlags + 1):
            acf_rows.append({"group": label, "lag": k, "acf": a[k], "pacf": p[k]})
        # dominant period (median over records): the rectified TIA vs its raw roll/pitch carriers
        def med_period(var):
            return float(np.median([dominant_period(rec.sort_values("time")[var].to_numpy(float), fs)
                                    for _, rec in g.groupby(["tonnage", "Hs"])]))
        summ_rows.append({"group": label, "lag1_acf": float(a[1]),
                          "tia_dominant_period_s": med_period(tgt),
                          "roll_dominant_period_s": med_period("phi"),
                          "pitch_dominant_period_s": med_period("theta"),
                          "n_records": g.groupby(["tonnage", "Hs"]).ngroups})
    pd.DataFrame(acf_rows).to_csv(out_dir / "acf.csv", index=False)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(out_dir / "summary.csv", index=False)
    return out_dir, summ


def main():
    out_dir, summ = run_diagnostics()
    print(summ.to_string(index=False))
    print(f"wrote {out_dir / 'summary.csv'}")
    from jmse.plots.diagnostics import plot_F11
    print("figure:", plot_F11(out_dir))


if __name__ == "__main__":
    main()
