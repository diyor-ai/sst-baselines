"""Descriptive warm-shift analysis: bias / variance split of the MSE and the area-mean anomaly series."""
import numpy as np

from sst.metrics import lat_weights


def _wmean(x, lat):
    w = np.broadcast_to(lat_weights(lat)[None, :, None], x.shape)
    return float((x * w).sum() / w.sum())


def bias_decomposition(f, o):
    """Mean bias (forecast - observed) and MSE = bias^2 + variance part, over all dates and
    grid points with cos(lat) weights. The variance part is MSE - bias^2 (spread of the errors
    around their mean). Shares of the MSE are given in percent."""
    err = f.values - o.values
    bias = _wmean(err, f["latitude"].values)
    mse = _wmean(err ** 2, f["latitude"].values)
    return {"bias": bias, "mse": mse, "bias_sq": bias ** 2, "variance_part": mse - bias ** 2,
            "bias_sq_pct_of_mse": 100 * bias ** 2 / mse, "variance_part_pct_of_mse": 100 * (1 - bias ** 2 / mse)}


def area_mean_series(anom):
    """cos(lat)-weighted area-mean anomaly for every day: a (time,) numpy array."""
    w = lat_weights(anom["latitude"].values)[None, :, None]
    x = anom.values
    wg = np.broadcast_to(w, x.shape)
    return (x * wg).sum(axis=(1, 2)) / wg.sum(axis=(1, 2))
