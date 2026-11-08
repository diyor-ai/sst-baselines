"""B1 spatial context: data for patch models. Samples = (inner grid point p, issue date t), inputs from a (2*halo+inner)^2 grid.

The grid holds the inner box plus a halo of `halo` points per side. Targets/evaluation are on the INNER points only.
Patch of an inner point = the (size x size) block of grid points centred on it, row-major (d_lat, d_lon) from -r..r, so the
centre is index (size*size)//2. Edge points of the inner box take their outer neighbours from the halo.

Anomalies: climatology from TRAINING years (existing code, per point). Standardised by ONE number per region: the std of the
TRAINING anomalies of the INNER points (identical to the existing LSTM's number); halo points are divided by that same number.
Nothing here reads a day after the end of the evaluation period or a later day than the issue date t.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from sst.climatology import anomaly, build_climatology
from sst.download import ROOT
from sst.lstm_data import SEQ_LEN, day_features, valid_mask

PROCESSED = ROOT / "data" / "processed"


def load_b1(path=None):
    with open(path or ROOT / "config_b1.yaml") as f:
        return yaml.safe_load(f)


def patch_index(n_inner, halo, size):
    """(n_inner[0]*n_inner[1], size*size) flat indices into the (n_inner+2*halo) grid; row p = inner point p (row-major)."""
    r = size // 2
    assert size % 2 == 1 and r <= halo
    nlat, nlon = n_inner
    nlon_g = nlon + 2 * halo
    ii, jj = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing="ij")
    ii, jj = ii.reshape(-1) + halo, jj.reshape(-1) + halo          # position of the inner point on the full grid
    offs = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]
    return np.stack([(ii + di) * nlon_g + (jj + dj) for di, dj in offs], axis=1).astype(np.int64)


def inner_block(arr, halo, n_inner=None):
    """Select the inner block of an array whose last two dims are the full grid."""
    return arr[..., halo:arr.shape[-2] - halo, halo:arr.shape[-1] - halo]


@dataclass
class B1Data:
    dates: pd.DatetimeIndex
    a: np.ndarray               # (T, G) float32 standardised anomalies on the FULL grid (inner + halo), NaN = missing
    dayf: np.ndarray            # (T, 2) sin/cos day of year
    std: float                  # training anomaly std of the inner points (deg C)
    leads: tuple
    train_pairs: np.ndarray     # flat t*P+p, p over inner points (P = n_inner points)
    val_pairs: np.ndarray
    pidx: np.ndarray            # (P, size*size) full-grid flat index of every patch point
    n_inner: tuple
    halo: int
    size: int
    clim: object                # climatology of the full grid (doy, lat, lon), TRAINING years only

    @property
    def n_points(self):         # inner points = P
        return self.pidx.shape[0]

    @property
    def n_in(self):             # channels per input day
        return self.size * self.size + 2

    @property
    def centre(self):           # full-grid flat index of each inner point
        return self.pidx[:, self.pidx.shape[1] // 2]


def make_data_b1(sst, cfg, halo, size, clim=None, period="val"):
    """`sst`: DataArray (time, latitude, longitude) on the FULL grid, ALREADY cut at the end of `period` (checked here)."""
    split, leads = cfg["split"], tuple(cfg["leads_days"])
    tr0, tr1 = map(pd.Timestamp, split["train"])
    va0, va1 = map(pd.Timestamp, split[period])
    if pd.Timestamp(sst.time.max().values) > va1:
        raise ValueError("sst contains days after the end of the evaluation period: cut the data first")
    if clim is None:
        clim = build_climatology(sst, cfg)                # training years only
    full = pd.date_range(sst.time.min().values, sst.time.max().values)
    anom = anomaly(sst, clim).reindex(time=full)
    nlat_g, nlon_g = anom.sizes["latitude"], anom.sizes["longitude"]
    n_inner = (nlat_g - 2 * halo, nlon_g - 2 * halo)
    a = anom.values.reshape(len(full), nlat_g * nlon_g)
    pidx = patch_index(n_inner, halo, size)
    centre = pidx[:, pidx.shape[1] // 2]
    in_train = (full >= tr0) & (full <= tr1)
    a_inner = a[:, centre]                                # (T, P) in the same order as the existing inner-only data
    std = float(np.nanstd(a_inner[in_train]))
    a32 = (a / std).astype(np.float32)
    # a sample needs: complete 30-day window at the centre and ALL patch points, and the centre's targets
    ok = valid_mask(a32[:, centre], leads)
    win = valid_mask(a32, ())                             # window-only validity for every grid point
    for k in range(pidx.shape[1]):
        ok &= win[:, pidx[:, k]]
    last = max(leads)
    t_ok = (full - pd.Timedelta(days=SEQ_LEN - 1) >= tr0) & (full + pd.Timedelta(days=last) <= tr1)
    v_ok = (full >= va0) & (full + pd.Timedelta(days=last) <= va1)
    return B1Data(dates=full, a=a32, dayf=day_features(full), std=std, leads=leads,
                  train_pairs=np.flatnonzero(ok & t_ok[:, None]), val_pairs=np.flatnonzero(ok & v_ok[:, None]),
                  pidx=pidx, n_inner=n_inner, halo=halo, size=size, clim=clim)


def gather_inputs_b1(data, flat):
    """(B, 30, size*size + 2) float32 for flat sample indices t*P+p: patch anomalies of days t-29..t, then sin/cos doy of those days."""
    P = data.n_points
    t, p = flat // P, flat % P
    idx = t[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]                        # days t-29..t (never later than t)
    pa = data.a[idx[:, :, None], data.pidx[p][:, None, :]]                        # (B, 30, size*size)
    return np.concatenate([pa, data.dayf[idx]], axis=2)


def gather_b1(data, flat):
    P = data.n_points
    t, p = flat // P, flat % P
    x = gather_inputs_b1(data, flat)
    y = np.stack([data.a[t + h, data.centre[p]] for h in data.leads], axis=1)
    return x, y


def load_halo_sst(region_halo, cfg):
    """Halo-grid SST cut at the validation end BEFORE anything else (the test period is never loaded into memory)."""
    v1 = cfg["split"]["val"][1]
    sst = xr.open_dataset(PROCESSED / f"{region_halo}.nc")["sst"].sel(time=slice(None, v1)).load()
    assert str(sst.time.max().values)[:10] <= v1
    return sst
