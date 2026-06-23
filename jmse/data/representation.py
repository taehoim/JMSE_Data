"""Aligned auxiliary targets for the C5 target-representation study.

Adds, to the standard ID arrays, the per-window future trajectories of phi, theta and GZ
(same windows/splits as build_id_arrays), plus the per-window GM, so a model can be
trained to predict (phi, theta) and reconstruct Xacc = sqrt(phi^2+theta^2), or to predict
GZ and invert Xacc = arcsin(GZ/GM). Alignment is guaranteed by indexing the curated series
with each window's own (group, last-input-time) identity; a reconstruction check asserts
the auxiliary phi/theta reproduce the primary Xacc target exactly.
"""
import numpy as np

from jmse import config
from jmse.data.curate import load_curated
from jmse.data.windowing import build_id_arrays

_AUX = ("phi", "theta", "GZ")


def gm_by_tonnage() -> dict:
    """GM per tonnage from the data relation GZ = GM*sin(Xacc) (GM constant per vessel)."""
    df = load_curated()
    gm = {}
    for ton, g in df.groupby("tonnage"):
        xa, gz = g["Xacc"].to_numpy(), g["GZ"].to_numpy()
        m = np.abs(np.sin(xa)) > 1e-3
        gm[int(ton)] = float(np.median(gz[m] / np.sin(xa[m])))
    return gm


def build_representation_arrays(lookback: int = None, horizon: int = None) -> dict:
    """ID arrays + aligned phi_/theta_/gz_/gm_ per split (keys mirror y_*)."""
    horizon = horizon or config.HORIZON
    d = build_id_arrays(lookback=lookback, horizon=horizon)
    df = load_curated()

    # per-(tonnage, Hs, realization) time-sorted target series, indexed the same way as
    # windowing: tidx is relative to each realization series' own time order.
    series = {}
    for (ton, hs, real), g in df.sort_values("time").groupby(["tonnage", "Hs", "realization"]):
        g = g.sort_values("time")
        series[(ton, hs, real)] = {c: g[c].to_numpy(float) for c in _AUX}
    gm = gm_by_tonnage()
    keys = d["group_keys"]

    for split in ("train", "val", "test"):
        gids, tidx = d[f"group_{split}"], d[f"tidx_{split}"]
        n = len(tidx)
        aux = {c: np.empty((n, horizon)) for c in _AUX}
        gm_arr = np.empty(n)
        for i, (gid, t) in enumerate(zip(gids, tidx)):
            key = tuple(keys[gid])
            sl = slice(t + 1, t + 1 + horizon)           # targets at t+1 .. t+H
            for c in _AUX:
                aux[c][i] = series[key][c][sl]
            gm_arr[i] = gm[int(key[0])]
        d[f"phi_{split}"] = aux["phi"]
        d[f"theta_{split}"] = aux["theta"]
        d[f"gz_{split}"] = aux["GZ"]
        d[f"gm_{split}"] = gm_arr
    return d
