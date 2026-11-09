"""B2 whole-region models: data. One sample = one issue date t = one whole 48x48 field.

Anomalies/standardisation come from sst.b1_data.make_data_b1 (halo grid, training-year climatology, ONE regional std = training std of
the inner points), so they are identical to B1. Input = days t-29..t (never later) x (anomaly, sin doy, cos doy); target = inner 40x40
anomaly at t+h. The model outputs 48x48; `crop` cuts [halo:-halo] = the inner 40x40.
Training issue dates: all 30 input days AND all targets in the training years. Evaluation issue dates: issue in the validation years,
t+max(lead) in the validation years, 30-day history complete (may reach into 2015) = the pairs of Ridge-patch / LSTM-patch.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sst.b1_data import load_b1, load_halo_sst, make_data_b1
from sst.lstm_data import SEQ_LEN, day_features

HALO = 4
GRID = 48
INNER = 40


@dataclass
class B2Data:
    dates: pd.DatetimeIndex
    a: np.ndarray            # (T, 48*48) float32 standardised anomalies, no NaN
    dayf: np.ndarray         # (T, 2) sin/cos of day of year
    std: float
    leads: tuple
    train_issue: np.ndarray  # indices t into `dates`
    val_issue: np.ndarray
    clim: object
    halo: int


def crop(x, halo=HALO):
    """(..., 48, 48) -> (..., 40, 40): exactly the inner box."""
    return x[..., halo:x.shape[-2] - halo, halo:x.shape[-1] - halo]


def make_data_b2(sst, cfg, halo=HALO, period="val"):
    """`sst`: halo-grid DataArray (time, lat, lon) ALREADY cut at the end of `period`."""
    d1 = make_data_b1(sst, cfg, halo, size=5, period=period)       # anomalies, std, climatology (checks the cut)
    ny, nx = sst.sizes["latitude"], sst.sizes["longitude"]
    assert (ny, nx) == (GRID, GRID), (ny, nx)
    a = d1.a
    assert not np.isnan(a).any(), "B2 expects a complete halo grid (no NaN / land)"
    tr0, tr1 = map(pd.Timestamp, cfg["split"]["train"])
    va0, va1 = map(pd.Timestamp, cfg["split"][period])
    full, last = d1.dates, max(d1.leads)
    t_ok = (full - pd.Timedelta(days=SEQ_LEN - 1) >= tr0) & (full + pd.Timedelta(days=last) <= tr1)
    v_ok = (full - pd.Timedelta(days=SEQ_LEN - 1) >= full[0]) & (full >= va0) & (full + pd.Timedelta(days=last) <= va1)
    return B2Data(dates=full, a=a, dayf=d1.dayf, std=d1.std, leads=d1.leads, train_issue=np.flatnonzero(t_ok),
                  val_issue=np.flatnonzero(v_ok), clim=d1.clim, halo=halo)


def gather_x(data, t):
    """(B, 30, 3, 48, 48) float32 for issue indices t: [anomaly, sin, cos] of days t-29..t. Reads nothing later than t."""
    t = np.asarray(t)
    idx = t[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
    an = data.a[idx].reshape(len(t), SEQ_LEN, 1, GRID, GRID)
    dy = np.broadcast_to(data.dayf[idx][..., None, None], (len(t), SEQ_LEN, 2, GRID, GRID))
    return np.concatenate([an, dy], axis=2)


def gather_y(data, t):
    """(B, 3, 40, 40) float32 targets: inner anomaly at t+h."""
    t = np.asarray(t)
    y = np.stack([data.a[t + h].reshape(len(t), GRID, GRID) for h in data.leads], axis=1)
    return np.ascontiguousarray(crop(y, data.halo))


def load_data_b2(region, cfg, period="val"):
    return make_data_b2(load_halo_sst(f"{region}_halo", cfg), cfg, load_b1()["halo"]["width_points"])
