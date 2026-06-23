"""Classical time-series baselines: AR(p) and a local-level Kalman filter.

Both operate directly on the target's own history (Xacc, radians) and forecast the
horizon *recursively*. They do not use the exogenous feature window or gradient
training; they fit in closed form on CPU. Their role in the benchmark is to show
how much the learned sequence models gain over standard statistical forecasters on
this low-autocorrelation signal (lag-1 acf approximately 0.21).

Interface (both classes):
    model.fit(yhist_train)            # yhist_train: (N, L) radians
    model.predict(yhist, horizon)     # yhist: (M, L) -> (M, horizon) radians
`yhist[i]` is the Xacc trajectory over the input window, newest value last.
"""
import numpy as np


class ARForecaster:
    """Autoregressive model x_t = c + sum_{k=1..p} a_k x_{t-k}, recursive multi-step.

    Fitted by ordinary least squares on lag/target pairs drawn from the training
    windows (overlapping rows are fine; OLS stays consistent).
    """

    def __init__(self, order: int = 8):
        self.order = int(order)
        self.coef_ = None        # (p,) lag coefficients a_1..a_p
        self.intercept_ = 0.0

    def fit(self, yhist_train: np.ndarray):
        p = self.order
        rows = np.asarray(yhist_train, float)
        if rows.ndim != 2 or rows.shape[1] <= p:
            raise ValueError(f"yhist width {rows.shape} must exceed order {p}")
        L = rows.shape[1]
        preds, targs = [], []
        for j in range(p, L):                       # target column j, lags j-1..j-p
            preds.append(rows[:, j - p:j][:, ::-1])  # cols [x_{j-1}, x_{j-2}, ..., x_{j-p}]
            targs.append(rows[:, j])
        Phi = np.vstack(preds)                       # (M, p), col 0 = most recent lag
        y = np.concatenate(targs)                    # (M,)
        A = np.hstack([np.ones((len(Phi), 1)), Phi]) # intercept + lags
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        return self

    def predict(self, yhist: np.ndarray, horizon: int) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("ARForecaster.predict called before fit")
        p = self.order
        rows = np.asarray(yhist, float)
        buf = rows[:, -p:][:, ::-1].copy()           # (N, p), buf[:,0]=most recent x_t
        out = np.empty((len(rows), horizon))
        for h in range(horizon):
            nxt = self.intercept_ + buf @ self.coef_  # (N,)
            out[:, h] = nxt
            buf = np.roll(buf, 1, axis=1)
            buf[:, 0] = nxt
        return out


class KalmanForecaster:
    """Local-level (random-walk-plus-noise) Kalman filter; forecast = filtered level.

    State noise q and observation noise R are estimated from the training first
    differences via the moment relations of an IMA(1,1)/local-level model:
        var(Δx) = q + 2R ,   acov(Δx, lag1) = -R .
    Under low autocorrelation this filter denoises and holds the level (a smoothed
    persistence), making it a fair "classical filter" point in the benchmark.
    """

    def __init__(self):
        self.q = None
        self.R = None

    def fit(self, yhist_train: np.ndarray):
        rows = np.asarray(yhist_train, float)
        dx = np.diff(rows, axis=1).ravel()           # first differences within windows
        dx = dx - dx.mean()
        g0 = float(np.mean(dx * dx))
        g1 = float(np.mean(dx[:-1] * dx[1:])) if dx.size > 1 else 0.0
        self.R = max(-g1, 1e-9)
        self.q = max(g0 + 2.0 * g1, 1e-9)
        return self

    def predict(self, yhist: np.ndarray, horizon: int) -> np.ndarray:
        if self.R is None:
            raise RuntimeError("KalmanForecaster.predict called before fit")
        rows = np.asarray(yhist, float)
        n, L = rows.shape
        level = rows[:, 0].copy()                    # init at first observation
        P = np.full(n, self.R)                        # initial state variance
        for t in range(1, L):                         # filter forward over the window
            P = P + self.q                            # predict step (level random walk)
            K = P / (P + self.R)                      # Kalman gain
            level = level + K * (rows[:, t] - level)  # update with observation
            P = (1.0 - K) * P
        # local-level forecast is flat: best estimate of all future obs is current level
        return np.repeat(level.reshape(-1, 1), horizon, axis=1)


_CLASSICAL = {
    "ar": ARForecaster,
    "kalman": KalmanForecaster,
}


def is_classical(name: str) -> bool:
    return name in _CLASSICAL


def build_classical(name: str, **kwargs):
    if name not in _CLASSICAL:
        raise KeyError(f"Unknown classical model '{name}'. Available: {sorted(_CLASSICAL)}")
    ctor = _CLASSICAL[name]
    if name == "ar":
        return ctor(order=kwargs.get("order", 8))
    return ctor()
