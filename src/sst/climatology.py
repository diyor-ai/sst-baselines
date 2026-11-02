"""Day-of-year climatology from TRAINING years only, with two smoothing options."""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d

CAL365 = pd.date_range("2001-01-01", "2001-12-31")  # a non-leap year defines the 365 calendar days
_LOOKUP = {m * 100 + d: i for i, (m, d) in enumerate(zip(CAL365.month, CAL365.day))}
SMOOTHINGS = ("harmonics3", "window31")


def doy365(times):
    """Index 0..364 of each date in a 365-day calendar; Feb 29 is mapped to Feb 28."""
    t = pd.DatetimeIndex(times)
    key = t.month * 100 + t.day
    key = np.where(key == 229, 228, key)
    return np.array([_LOOKUP[k] for k in key])


def raw_doy_mean(sst, train):
    """Mean SST for each of the 365 calendar days, using dates inside `train` only."""
    tr = sst.sel(time=slice(*train))
    idx = doy365(tr["time"].values)
    missing = sorted(set(range(365)) - set(idx))
    if missing:
        raise ValueError(f"training period has no data for calendar days {missing[:5]}...")
    out = np.stack([np.nanmean(tr.values[idx == i], axis=0) for i in range(365)])
    return xr.DataArray(out, dims=("doy", "latitude", "longitude"),
                        coords={"latitude": sst["latitude"], "longitude": sst["longitude"]})


def smooth_harmonics(clim, n_harm=3):
    """Replace the 365-day cycle by its least-squares fit of a mean plus n_harm annual harmonics."""
    d = np.arange(365)
    cols = [np.ones(365)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * d / 365), np.sin(2 * np.pi * k * d / 365)]
    X = np.stack(cols, axis=1)
    y = clim.values.reshape(365, -1)
    fit = np.full_like(y, np.nan, dtype=float)
    ok = ~np.isnan(y).any(axis=0)  # land columns stay NaN
    coef, *_ = np.linalg.lstsq(X, y[:, ok], rcond=None)
    fit[:, ok] = X @ coef
    return clim.copy(data=fit.reshape(clim.shape))


def smooth_window(clim, window=31):
    """Circular moving average over `window` calendar days (Dec 31 wraps to Jan 1)."""
    return clim.copy(data=uniform_filter1d(clim.values, size=window, axis=0, mode="wrap"))


def build_climatology(sst, cfg):
    """Smoothed climatology; the smoothing is chosen by cfg['climatology']['smoothing']."""
    c = cfg["climatology"]
    if c["years"] != "train":
        raise ValueError("climatology.years must be 'train' (anything else leaks validation/test data)")
    raw = raw_doy_mean(sst, cfg["split"]["train"])
    if c["smoothing"] == "harmonics3":
        return smooth_harmonics(raw, 3)
    if c["smoothing"] == "window31":
        return smooth_window(raw, 31)
    raise ValueError(f"set climatology.smoothing in config.yaml to one of {SMOOTHINGS}, got {c['smoothing']!r}")


def clim_at(clim, times):
    """Climatology laid out on the given dates -> (time, latitude, longitude)."""
    times = pd.DatetimeIndex(times)
    return xr.DataArray(clim.values[doy365(times)], dims=("time", "latitude", "longitude"),
                        coords={"time": times, "latitude": clim["latitude"], "longitude": clim["longitude"]})


def anomaly(sst, clim):
    return sst - clim_at(clim, sst["time"].values)
