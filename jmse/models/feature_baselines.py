"""Non-neural baselines that use the SAME windowed 8-DOF inputs as the neural models (reviewer 2.5).

The classical AR/Kalman baselines forecast from the target's own history only, which is weaker
information than the neural models receive. To make the benchmark's inputs fair, these two baselines
consume the full flattened input window $\\{u,v,w,p,q,r,\\phi,\\theta\\}\\times L$ and predict all $H$
horizons directly: a ridge linear regressor (closed form) and a gradient-boosting regressor
(per-horizon). They are deterministic given a fixed seed, so they enter the benchmark once.

Interface mirrors the neural models: fit(X, y) with X (N, L, F), y (N, H); predict(X) -> (N, H).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


def _flat(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return X.reshape(len(X), -1)


class RidgeWindowForecaster:
    """Ridge direct multi-step regressor on the flattened input window (multi-output)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = Ridge(alpha=self.alpha).fit(_flat(X), np.asarray(y, float))
        return self

    def predict(self, X: np.ndarray, horizon: int = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RidgeWindowForecaster.predict called before fit")
        return self.model.predict(_flat(X))


class GBMWindowForecaster:
    """Gradient-boosting (histogram) regressor per horizon on the flattened input window."""

    def __init__(self, seed: int = 0, max_iter: int = 200):
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        base = HistGradientBoostingRegressor(max_iter=self.max_iter, random_state=self.seed)
        self.model = MultiOutputRegressor(base).fit(_flat(X), np.asarray(y, float))
        return self

    def predict(self, X: np.ndarray, horizon: int = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GBMWindowForecaster.predict called before fit")
        return self.model.predict(_flat(X))


FEATURE_BASELINES = {"ridge": RidgeWindowForecaster, "gbm": GBMWindowForecaster}


def run_feature_baselines(smoke: bool = False):
    """Fit the fair-input baselines on the ID split and merge them into the benchmark artifacts.

    Reuses the exact ID windows of the neural sweep, appends ridge/gbm rows to the benchmark
    raw_metrics.csv (idempotently), saves per-window preds for the significance test, and rebuilds T4.
    """
    from jmse import config
    from jmse.data.windowing import build_id_arrays
    from jmse.eval import stats
    from jmse.eval.metrics import per_horizon_metrics, to_degrees
    from jmse.models.base import set_seed

    bench = config.RESULTS_DIR / ("benchmark_smoke" if smoke else "benchmark")
    (bench / "preds").mkdir(parents=True, exist_ok=True)
    d = build_id_arrays()
    models = {"ridge": RidgeWindowForecaster(alpha=1.0),
              "gbm": GBMWindowForecaster(seed=0, max_iter=(20 if smoke else 200))}
    rows = []
    for name, model in models.items():
        set_seed(0)
        model.fit(d["X_train"], d["y_train"])
        pred = model.predict(d["X_test"])
        np.savez(bench / "preds" / f"{name}.npz", y_true=d["y_test"], y_pred=pred)
        deg = to_degrees(per_horizon_metrics(d["y_test"], pred))
        for k in range(len(deg["r2"])):
            rows.append({"model": name, "seed": 0, "horizon_s": k + 1, "rmse_deg": deg["rmse"][k],
                         "mae_deg": deg["mae"][k], "r2": deg["r2"][k]})
        rows.append({"model": name, "seed": 0, "horizon_s": "overall", "rmse_deg": deg["overall"]["rmse"],
                     "mae_deg": deg["overall"]["mae"], "r2": deg["overall"]["r2"]})
        print(f"{name:12s} overall R2={deg['overall']['r2']:.4f} RMSE={deg['overall']['rmse']:.2f}deg")

    new = pd.DataFrame(rows)
    raw_path = bench / "raw_metrics.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        raw = raw[~raw["model"].isin(models)]                # idempotent re-run
        raw = pd.concat([raw, new], ignore_index=True)
    else:
        raw = new
    raw.to_csv(raw_path, index=False)
    stats.write_t4(stats.aggregate_seeds(raw), bench)
    return bench, new


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    bench, new = run_feature_baselines(smoke=args.smoke)
    print(f"merged into {bench / 'raw_metrics.csv'} and rebuilt T4")
    if not args.no_plots:
        from jmse.plots.benchmark import plot_benchmark
        print("figures:", *[str(f) for f in plot_benchmark(bench)])


if __name__ == "__main__":
    main()
