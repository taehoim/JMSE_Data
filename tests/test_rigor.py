"""Tests for the UQ-rigor compute pipeline (P5): shapes, schema, and sanity."""
import numpy as np

from jmse.uq import rigor


def _synthetic(n_per_group=200, H=5, seed=0):
    r = np.random.default_rng(seed)
    group_keys = [(10, 3.0), (20, 5.0), (30, 7.0)]
    groups = np.repeat(np.arange(len(group_keys)), n_per_group)
    N = groups.size
    mean = np.abs(r.standard_normal((N, H)) * 0.1) + 0.2       # radians, positive heel signal
    std = np.full((N, H), 0.05)
    y = mean + std * r.standard_normal((N, H))                 # drawn from the encoded Gaussian
    z = 1.645
    quant = np.stack([mean - z * std, mean, mean + z * std], axis=-1)   # (N,H,3)
    taus = np.array([0.05, 0.5, 0.95])
    return quant, taus, mean, std, y, groups, group_keys


def test_compute_rigor_schema_and_ranges():
    quant, taus, mean, std, y, groups, gk = _synthetic()
    thr_rad = [np.radians(15), np.radians(20), np.radians(25)]
    thr_deg = [15, 20, 25]
    out = rigor.compute_rigor(quant, taus, mean, std, y, groups, gk, thr_rad, thr_deg)

    cond = out["conditional"]
    assert set(cond["dim"]) == {"horizon", "Hs", "tonnage", "marginal", "regime"}
    assert (cond["coverage"].dropna().between(0, 1)).all()
    # 5 horizons + 3 Hs + 3 tonnage + 1 marginal + 2 regime rows
    assert len(cond) == 5 + 3 + 3 + 1 + 2

    conf = out["conformal"]
    assert list(conf["method"]) == ["Quantile band", "Quantile + CQR",
                                    "Ensemble Gaussian", "Ensemble + split-conformal"]
    assert (conf["coverage"].between(0, 1)).all()

    prob = out["probcalib"]
    assert len(prob) == 3 * 2                                   # 3 thresholds x 2 sources
    assert (prob["brier"] >= 0).all() and (prob["ece"] >= 0).all()


def test_conformal_eval_coverage_near_nominal_on_calibrated_synthetic():
    # Data are generated from the same Gaussian the band encodes -> all methods ~0.90.
    quant, taus, mean, std, y, groups, gk = _synthetic(n_per_group=2000)
    conf = rigor.compute_conformal(quant, taus, mean, std, y, groups)
    for _, row in conf.iterrows():
        assert abs(row["coverage"] - 0.90) < 0.05
