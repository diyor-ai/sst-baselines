"""Ridge baseline: one linear model per grid point and per lead, predicting the anomaly at t+h.

Features at issue date t (8 per grid point): anomalies at lags 0..6 (t, t-1, ..., t-6) and the mean
anomaly over t-29..t. Nothing after t is ever used. Anomalies are relative to the TRAINING climatology.

Penalty convention: the model minimises  mean((y - b - Z w)^2) + alpha * |w|^2  with Z the standardised
features (training mean/std), i.e. alpha is on the scale of the normal equations divided by n, so one
grid of alphas (1e-3 ... 1e3) means the same thing for every region and lead.
"""
import numpy as np
import pandas as pd

N_LAGS = 7          # lags 0..6
MEAN_WIN = 30       # mean over t-29..t
N_FEAT = N_LAGS + 1
ALPHA_GRID = tuple(10.0 ** np.arange(-3, 3.01, 0.5))  # 13 values, 1e-3 ... 1e3


def make_features(a, t0, issue):
    """Feature array (n_issue, n_points, 8) for the given issue dates.

    `a` is (time, points) of anomalies on a complete daily axis starting at `t0` (NaN = missing day).
    Row i only reads days <= issue[i]. A row is NaN where any needed day is missing or lies before t0
    (not enough history), so callers can drop it and count it.
    """
    a = np.asarray(a, dtype=float)
    T, P = a.shape
    t0, issue = pd.Timestamp(t0), pd.DatetimeIndex(issue)
    pad = MEAN_WIN - 1
    ap = np.concatenate([np.full((pad, P), np.nan), a])  # padded: day d sits at row d + pad
    day = np.asarray((issue - t0).days)                   # position of each issue date on the axis
    inside = (day >= 0) & (day < T)
    pos = np.where(inside, day + pad, pad)                # dummy position for rows outside, set to NaN below
    roll = pd.DataFrame(ap).rolling(MEAN_WIN, min_periods=MEAN_WIN).mean().to_numpy()  # row i: days i-29..i
    feats = [ap[pos - k] for k in range(N_LAGS)] + [roll[pos]]
    X = np.stack(feats, axis=-1)
    X[~inside] = np.nan
    return X


def usable_rows(X, y):
    """Rows where every feature of every point and the target are present."""
    return ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(y).any(axis=1))


def fit_ridge(X, y, alphas=ALPHA_GRID):
    """Fit one ridge model per point for every alpha. X is (n, points, 8), y is (n, points).

    Features are standardised with THIS data's mean/std (call it with training rows only); the target
    is centred by its mean (the intercept). Returns a dict holding the standardisation statistics and
    `coef`, the weights on standardised features, shape (n_alpha, points, 8).
    """
    n = X.shape[0]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    ym = y.mean(axis=0)
    Z, yc = (X - mu) / sd, y - ym
    G = np.einsum("npi,npj->pij", Z, Z) / n
    b = np.einsum("npi,np->pi", Z, yc) / n
    eye = np.eye(X.shape[-1])
    coef = np.stack([np.linalg.solve(G + al * eye, b[..., None])[..., 0] for al in alphas])
    return {"mu": mu, "sd": sd, "ymean": ym, "coef": coef, "alphas": np.asarray(alphas, dtype=float)}


def predict_ridge(model, X):
    """Predicted target for every alpha: (n_alpha, n, points). Uses the stored training mean/std only."""
    Z = (X - model["mu"]) / model["sd"]
    return np.einsum("npi,api->anp", Z, model["coef"]) + model["ymean"][None, None, :]


def coef_original_units(model, ia):
    """Weights on the raw (unstandardised) features for alpha index `ia`: (points, 8)."""
    return model["coef"][ia] / model["sd"]


# ---------------------------------------------------------------------------------------------
# Orchestration: fit on training years, choose alpha on validation, produce forecasts.
# ---------------------------------------------------------------------------------------------
import xarray as xr  # noqa: E402

from sst.climatology import anomaly, clim_at  # noqa: E402
from sst.metrics import rmse, valid_pairs  # noqa: E402


def feature_ready_days(dates):
    """Days d whose whole history d-29..d is present: a forecast issued on d has all its inputs."""
    dates = pd.DatetimeIndex(dates)
    full = pd.date_range(dates.min(), dates.max())
    have = pd.Series(full.isin(dates).astype(float), index=full)
    ok = have.rolling(MEAN_WIN, min_periods=MEAN_WIN).sum() == MEAN_WIN
    return full[ok.to_numpy()]


def _pairs(sst_times, start, end, h):
    """Issue/target pairs for targets in [start, end]. A pair is skipped (and counted) if the target
    day is missing, or the issue day lacks its full 30-day history (so the features are incomplete)."""
    ready = feature_ready_days(sst_times)
    present = pd.DatetimeIndex(sst_times)
    target = pd.date_range(start, end)
    issue = target - pd.Timedelta(days=h)
    in_ok, tg_ok = issue.isin(ready), target.isin(present)
    ok = in_ok & tg_ok
    counts = {"lead_days": h, "expected": int(len(target)), "used": int(ok.sum()), "skipped": int((~ok).sum()),
              "skipped_input_incomplete": int((~in_ok).sum()), "skipped_target_missing": int((~tg_ok).sum())}
    return issue[ok], target[ok], counts


def ridge_lead(sst, clim, cfg, h, chunk=200, period="val", alpha=None):
    """Everything for one region and lead. For period="val" `sst` must already be cut at the validation end.

    period="test": the model is still fitted on TRAINING years only and uses the given `alpha` (chosen earlier
    on validation, stored in config.yaml); no alpha is chosen and nothing from the test targets enters the fit.

    1. fit on training years only (targets <= train end; features read only days <= issue),
    2. predict the validation targets for every alpha and pick the alpha with the lowest
       area-mean validation RMSE (one alpha for the whole region),
    3. return the forecast SST for that alpha: climatology(t+h) + predicted anomaly.
    """
    train, val = cfg["split"]["train"], cfg["split"][period]
    if (period == "test") != (alpha is not None):
        raise ValueError("alpha must be given for period='test' and must not be given for period='val'")
    sst_tr = sst.sel(time=slice(*train))
    anom_full = anomaly(sst, clim).reindex(time=pd.date_range(sst.time.min().values, sst.time.max().values))
    anom_tr = anomaly(sst_tr, clim).reindex(time=pd.date_range(sst_tr.time.min().values, sst_tr.time.max().values))
    nlat, nlon = sst.sizes["latitude"], sst.sizes["longitude"]
    a_full = anom_full.values.reshape(anom_full.sizes["time"], -1)
    a_tr = anom_tr.values.reshape(anom_tr.sizes["time"], -1)
    t0_full, t0_tr = anom_full.time.values[0], anom_tr.time.values[0]

    i_tr, t_tr, c_tr = _pairs(sst_tr.time.values, train[0], train[1], h)
    i_va, t_va, c_va = _pairs(sst.time.values, val[0], val[1], h)
    y_tr = anom_tr.sel(time=t_tr).values.reshape(len(t_tr), -1)
    obs_va = sst.sel(time=t_va)
    clim_va = clim_at(clim, t_va)
    alphas = np.asarray(ALPHA_GRID) if alpha is None else np.asarray([alpha], dtype=float)
    P = a_full.shape[1]
    pred = np.empty((len(alphas), len(t_va), P))
    for p0 in range(0, P, chunk):
        sl = slice(p0, min(P, p0 + chunk))
        Xtr = make_features(a_tr[:, sl], t0_tr, i_tr)
        ok = usable_rows(Xtr, y_tr[:, sl])
        assert ok.all(), "training rows with missing inputs should have been excluded by _pairs"
        model = fit_ridge(Xtr, y_tr[:, sl], alphas)
        Xva = make_features(a_full[:, sl], t0_full, i_va)
        assert not np.isnan(Xva).any()
        pred[:, :, sl] = predict_ridge(model, Xva)
    pred = pred.reshape(len(alphas), len(t_va), nlat, nlon)

    def as_sst(anom_pred):
        return xr.DataArray(clim_va.values + anom_pred, dims=("time", "latitude", "longitude"),
                            coords={"time": pd.DatetimeIndex(t_va), "latitude": sst["latitude"],
                                    "longitude": sst["longitude"]})

    if alpha is not None:  # fixed alpha (test): no selection, no validation curve
        info = {"alpha": float(alpha), "fit_years": train, "train_pairs": c_tr, "test_pairs": c_va,
                "n_train_rows": int(len(i_tr))}
        return as_sst(pred[0]), (i_va, t_va), info
    curve = [rmse(as_sst(pred[k]), obs_va) for k in range(len(alphas))]
    best = int(np.argmin(curve))
    info = {"alphas": [float(x) for x in alphas], "val_rmse": [float(x) for x in curve],
            "alpha": float(alphas[best]), "alpha_index": best,
            "alpha_at_grid_edge": best in (0, len(alphas) - 1),
            "train_pairs": c_tr, "val_pairs": c_va, "n_train_rows": int(len(i_tr))}
    return as_sst(pred[best]), (i_va, t_va), info
