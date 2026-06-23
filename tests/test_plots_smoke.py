def test_f2_distribution_figure_is_created():
    from jmse.plots.distribution import plot_xacc_distribution
    out = plot_xacc_distribution()
    assert out.exists() and out.stat().st_size > 0


def test_f12_uq_rigor_figure_is_created(tmp_path):
    import pandas as pd
    from jmse.plots.uq_rigor import plot_F12
    pd.DataFrame([
        {"dim": "horizon", "value": k, "coverage_mean": 0.95 - 0.01 * k,
         "coverage_std": 0.005, "mpiw_deg_mean": 6.0 + k, "mpiw_deg_std": 0.1} for k in range(1, 6)
    ] + [
        {"dim": "Hs", "value": v, "coverage_mean": c, "coverage_std": 0.005,
         "mpiw_deg_mean": 10.0, "mpiw_deg_std": 0.1}
        for v, c in [(3.0, 0.94), (5.0, 0.92), (7.0, 0.89)]
    ] + [
        {"dim": "regime", "value": r, "coverage_mean": c, "coverage_std": 0.005,
         "mpiw_deg_mean": float("nan"), "mpiw_deg_std": 0.0}
        for r, c in [("below", 0.93), ("above", 0.82)]
    ]).to_csv(tmp_path / "T9_conditional_coverage.csv", index=False)
    pd.DataFrame([
        {"method": m, "coverage_mean": c, "coverage_std": 0.002,
         "mpiw_deg_mean": 10.7, "mpiw_deg_std": 0.1, "n_eval": 5430}
        for m, c in [("Quantile band", 0.915), ("Quantile + CQR", 0.896),
                     ("Ensemble Gaussian", 0.908), ("Ensemble + split-conformal", 0.890)]
    ]).to_csv(tmp_path / "T10_conformal.csv", index=False)
    out = plot_F12(tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_f_hysteresis_figure_is_created(tmp_path):
    import pandas as pd
    from jmse.plots.hysteresis import plot_F_hysteresis
    rows = []
    for n in (1, 3, 5):
        for k in range(1, n + 1):
            rows.append({"k": k, "n": n,
                         "false_episodes_per_hour": 120.0 / (k + 1),
                         "lead_time_s": 4.0 - 0.3 * (k - 1),
                         "detection_rate": min(1.0, 0.95 - 0.05 * (k - 1)),
                         "precision": 0.4 + 0.05 * (k - 1),
                         "false_alert_rate": 0.6 - 0.05 * (k - 1),
                         "n_seeds": 5})
    pd.DataFrame(rows).to_csv(tmp_path / "T_hysteresis.csv", index=False)
    out = plot_F_hysteresis(tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_f13_event_figure_is_created(tmp_path):
    import pandas as pd
    from jmse.plots.events import plot_F13
    rows = []
    for td in (15, 20, 25):
        for a, det, fe, pr in [("point", 0.86, 200, 0.4), ("prob", 0.88, 160, 0.45),
                               ("naive", 0.6, 100, 0.4)]:
            rows.append({"threshold_deg": td, "alarm": a,
                         "detection_rate_mean": det, "detection_rate_std": 0.01,
                         "false_episodes_per_hour_mean": fe, "false_episodes_per_hour_std": 5.0,
                         "precision_mean": pr, "precision_std": 0.01})
    pd.DataFrame(rows).to_csv(tmp_path / "T12_event_metrics.csv", index=False)
    out = plot_F13(tmp_path)
    assert out.exists() and out.stat().st_size > 0
