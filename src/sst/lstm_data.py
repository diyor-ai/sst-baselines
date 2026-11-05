"""Data for the LSTM experiment: samples = (grid point, issue date t).

Input  : 30 x 3 sequence = standardised anomaly a(t-29..t) plus sin/cos of the day of year of each of those days.
Target : standardised anomaly at t+1, t+7, t+30.
Anomalies are relative to the existing TRAINING climatology; the standardisation is ONE number per region, the
standard deviation of the TRAINING anomalies. Nothing here ever reads a day after the split's last day.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sst.climatology import anomaly, build_climatology, doy365

SEQ_LEN = 30


def day_features(dates):
    """(n_days, 2) array: sin and cos of the day of year (365-day calendar, Feb 29 -> Feb 28)."""
    ang = 2.0 * np.pi * doy365(dates) / 365.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def valid_mask(a, leads, seq_len=SEQ_LEN):
    """(T, P) bool: True where the window a[t-seq_len+1..t] and the targets a[t+h] all exist and are not NaN.
    Row t = issue date. Rows without enough history or without all targets are False."""
    ok = ~np.isnan(a)
    T = a.shape[0]
    csum = np.concatenate([np.zeros((1, a.shape[1]), dtype=np.int64), np.cumsum(ok, axis=0, dtype=np.int64)])
    # number of valid days in t-seq_len+1..t, for t >= seq_len-1
    win = np.zeros_like(ok)
    win[seq_len - 1:] = (csum[seq_len:] - csum[:T - seq_len + 1]) == seq_len
    out = win.copy()
    for h in leads:
        tgt = np.zeros_like(ok)
        tgt[:T - h] = ok[h:]
        out &= tgt
    return out


@dataclass
class LSTMData:
    dates: pd.DatetimeIndex     # daily axis (all days up to the validation end)
    a: np.ndarray               # (T, P) float32 standardised anomalies (NaN = missing)
    dayf: np.ndarray            # (T, 2) float32 sin/cos of day of year
    std: float                  # training anomaly std of the region (units: deg C)
    leads: tuple
    train_pairs: np.ndarray     # flat indices t*P+p of training samples (int64)
    val_pairs: np.ndarray       # flat indices of validation samples (int64, sorted by date then point)
    shape: tuple                # (nlat, nlon)
    clim: object                # climatology DataArray (doy, lat, lon)

    @property
    def n_points(self):
        return self.a.shape[1]


def make_data(sst, cfg, clim=None, period="val"):
    """`sst`: DataArray (time, latitude, longitude), daily, ALREADY cut at the end of `period` (checked here).
    period="val" (default, used for training/validation) or "test" (only the guarded test runner, src/sst/lstm_test.py).
    The `val_pairs` field holds the pairs of that period."""
    split, leads = cfg["split"], tuple(cfg["leads_days"])
    tr0, tr1 = map(pd.Timestamp, split["train"])
    va0, va1 = map(pd.Timestamp, split[period])
    if pd.Timestamp(sst.time.max().values) > va1:
        raise ValueError("sst contains days after the end of the evaluation period: cut the data first")
    if clim is None:
        clim = build_climatology(sst, cfg)  # uses training years only
    full = pd.date_range(sst.time.min().values, sst.time.max().values)
    anom = anomaly(sst, clim).reindex(time=full)
    P = anom.sizes["latitude"] * anom.sizes["longitude"]
    a = anom.values.reshape(len(full), P)
    in_train = (full >= tr0) & (full <= tr1)
    std = float(np.nanstd(a[in_train]))
    a32 = (a / std).astype(np.float32)
    ok = valid_mask(a32, leads)
    last = max(leads)
    # training issue dates: window and ALL targets inside the training years
    t_ok = (full - pd.Timedelta(days=SEQ_LEN - 1) >= tr0) & (full + pd.Timedelta(days=last) <= tr1)
    # validation issue dates: issue in the validation years, all targets inside them (history may reach into 2015)
    v_ok = (full >= va0) & (full + pd.Timedelta(days=last) <= va1)
    tr_flat = np.flatnonzero(ok & t_ok[:, None])
    va_flat = np.flatnonzero(ok & v_ok[:, None])
    return LSTMData(dates=full, a=a32, dayf=day_features(full), std=std, leads=leads, train_pairs=tr_flat,
                    val_pairs=va_flat, shape=(anom.sizes["latitude"], anom.sizes["longitude"]), clim=clim)


def gather_inputs(data, flat):
    """Inputs only, (B, 30, 3) float32, for flat sample indices t*P+p. Reads days t-29..t and nothing later."""
    P = data.n_points
    t, p = flat // P, flat % P
    idx = t[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]            # days t-29..t (never later than t)
    return np.concatenate([data.a[idx, p[:, None]][..., None], data.dayf[idx]], axis=2)


def gather(data, flat):
    """Inputs (B, 30, 3) and targets (B, 3) float32 numpy arrays for flat sample indices t*P+p."""
    P = data.n_points
    t, p = flat // P, flat % P
    x = gather_inputs(data, flat)
    y = np.stack([data.a[t + h, p] for h in data.leads], axis=1)
    return x, y
