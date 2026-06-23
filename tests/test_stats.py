import numpy as np
import pandas as pd

from jmse.eval import stats


def _raw():
    # two models, lstm over 2 seeds, ar deterministic (1 seed), horizons 1..2 + overall
    rows = []
    for seed, r2 in [(0, 0.60), (1, 0.64)]:
        rows += [
            {"model": "lstm", "seed": seed, "horizon_s": 1, "rmse_deg": 3.0, "mae_deg": 2.0, "r2": r2 + 0.1},
            {"model": "lstm", "seed": seed, "horizon_s": 2, "rmse_deg": 4.0, "mae_deg": 3.0, "r2": r2},
            {"model": "lstm", "seed": seed, "horizon_s": "overall", "rmse_deg": 3.5, "mae_deg": 2.5, "r2": r2},
        ]
    rows += [
        {"model": "ar", "seed": 0, "horizon_s": 1, "rmse_deg": 5.0, "mae_deg": 4.0, "r2": 0.30},
        {"model": "ar", "seed": 0, "horizon_s": 2, "rmse_deg": 5.5, "mae_deg": 4.5, "r2": 0.25},
        {"model": "ar", "seed": 0, "horizon_s": "overall", "rmse_deg": 5.25, "mae_deg": 4.25, "r2": 0.27},
    ]
    return pd.DataFrame(rows)


def test_aggregate_seeds_mean_std_and_order():
    agg = stats.aggregate_seeds(_raw())
    lstm_overall = agg[(agg.model == "lstm") & (agg.horizon_s == "overall")].iloc[0]
    assert abs(lstm_overall["r2_mean"] - 0.62) < 1e-9
    assert abs(lstm_overall["r2_std"] - np.std([0.60, 0.64], ddof=1)) < 1e-9
    ar_overall = agg[(agg.model == "ar") & (agg.horizon_s == "overall")].iloc[0]
    assert ar_overall["r2_std"] == 0.0 and ar_overall["n_seeds"] == 1
    # ar (classical) ordered before lstm; 'overall' is the last horizon row per model
    models_in_order = agg["model"].tolist()
    assert models_in_order.index("ar") < models_in_order.index("lstm")


def test_write_t4(tmp_path):
    agg = stats.aggregate_seeds(_raw())
    path = stats.write_t4(agg, tmp_path)
    t4 = pd.read_csv(path)
    assert set(t4["model"]) == {"lstm", "ar"}
    assert (t4["horizon_s"] == "overall").all()
    assert "label" in t4.columns


def test_mean_std_over_seeds_generic_grouping():
    # generic mean+/-std aggregator used for the UQ / early-warning / representation tables (S1)
    rows = []
    for seed, picp in [(0, 0.70), (1, 0.74), (2, 0.72)]:
        rows.append({"method": "Quantile", "seed": seed, "picp": picp, "mpiw_deg": 10.0})
        rows.append({"method": "MC-Dropout", "seed": seed, "picp": 0.90, "mpiw_deg": 12.0})
    agg = stats.mean_std_over_seeds(pd.DataFrame(rows), ["method"], ["picp", "mpiw_deg"])
    q = agg[agg.method == "Quantile"].iloc[0]
    assert abs(q["picp_mean"] - 0.72) < 1e-9
    assert abs(q["picp_std"] - np.std([0.70, 0.74, 0.72], ddof=1)) < 1e-9
    assert q["n_seeds"] == 3
    assert q["mpiw_deg_std"] == 0.0                      # constant metric -> std 0, not NaN


def test_mean_std_over_seeds_passthrough_columns():
    rows = [{"threshold_deg": 15, "alarm": "naive", "seed": s, "roc_auc": 0.9,
             "prevalence": 0.12, "n_events": 1361} for s in (0, 1, 2)]
    agg = stats.mean_std_over_seeds(pd.DataFrame(rows), ["threshold_deg", "alarm"],
                                    ["roc_auc"], passthrough_cols=["prevalence", "n_events"])
    r = agg.iloc[0]
    assert r["roc_auc_std"] == 0.0 and r["prevalence"] == 0.12 and r["n_events"] == 1361


def test_paired_wilcoxon_vs(tmp_path):
    preds = tmp_path / "preds"
    preds.mkdir()
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal((400, 5))
    # lstm: small error; ar: clearly larger error -> ar significantly worse
    np.savez(preds / "lstm.npz", y_true=y_true, y_pred=y_true + 0.1 * rng.standard_normal((400, 5)))
    np.savez(preds / "ar.npz", y_true=y_true, y_pred=y_true + 0.5 * rng.standard_normal((400, 5)))
    df = stats.paired_wilcoxon_vs(preds, reference="lstm")
    assert list(df["model"]) == ["ar"]
    row = df.iloc[0]
    assert row["median_abs_err_delta_rad"] > 0          # ar has larger error than lstm
    assert row["p_value"] < 0.001 and row["p_holm"] <= 1.0
