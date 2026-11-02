"""Forecast baselines. Each returns a (time=target dates, latitude, longitude) DataArray.

`issue` and `target` are equal-length DatetimeIndex objects (target = issue + lead).
`sst` may have missing days; use sst.metrics.valid_pairs first so every pair is usable.
"""
import numpy as np
import pandas as pd
import xarray as xr

from sst.climatology import anomaly, clim_at


def _wrap(values, sst, target):
    return xr.DataArray(values, dims=("time", "latitude", "longitude"),
                        coords={"time": pd.DatetimeIndex(target), "latitude": sst["latitude"],
                                "longitude": sst["longitude"]})


def _obs(sst, dates):
    return sst.reindex(time=pd.DatetimeIndex(dates)).values  # NaN where the day is missing


def persistence(sst, issue, target):
    """Tomorrow looks like today: forecast(t+h) = SST(t)."""
    return _wrap(_obs(sst, issue), sst, target)


def climatology_forecast(sst, clim, issue, target):
    """Forecast(t+h) = climatology of the target calendar day (ignores the input day)."""
    return _wrap(clim_at(clim, target).values, sst, target)


def anomaly_persistence(sst, clim, issue, target):
    """Climatology of the target day plus today's anomaly, kept unchanged."""
    a = _obs(sst, issue) - clim_at(clim, issue).values
    return _wrap(clim_at(clim, target).values + a, sst, target)


def lag_corr(sst, clim, train, h):
    """Per grid point correlation r(h) between anomaly(t) and anomaly(t+h), training dates only."""
    a = anomaly(sst.sel(time=slice(*train)), clim)
    t0 = a.time.to_index()
    t1 = t0 + pd.Timedelta(days=h)
    keep = t1.isin(t0)
    x = a.sel(time=t0[keep]).values
    y = a.sel(time=t1[keep]).values
    num = np.nansum(x * y, axis=0)
    den = np.sqrt(np.nansum(x * x, axis=0) * np.nansum(y * y, axis=0))
    return xr.DataArray(num / den, dims=("latitude", "longitude"),
                        coords={"latitude": sst["latitude"], "longitude": sst["longitude"]})


def damped_persistence(sst, clim, issue, target, r):
    """Climatology of the target day plus r(h) times today's anomaly (r from training years)."""
    a = _obs(sst, issue) - clim_at(clim, issue).values
    return _wrap(clim_at(clim, target).values + r.values[None] * a, sst, target)


def ocn(sst, issue, target, n_years):
    """Optimal-climate-normals style: mean of the same calendar day over the last n_years years.

    For target date T the values come from T minus 1..n_years years (Feb 29 uses Feb 28).
    Every date used must be strictly before the issue date, otherwise a ValueError is raised.
    """
    issue, target = pd.DatetimeIndex(issue), pd.DatetimeIndex(target)
    base = target.where(~((target.month == 2) & (target.day == 29)), target - pd.Timedelta(days=1))
    stack = []
    for k in range(1, n_years + 1):
        dates = base - pd.DateOffset(years=k)
        if not (dates < issue).all():
            raise ValueError("OCN would use data from on/after the issue date")
        stack.append(_obs(sst, dates))
    with np.errstate(all="ignore"):
        mean = np.nanmean(np.stack(stack), axis=0)
    return _wrap(mean, sst, target)
