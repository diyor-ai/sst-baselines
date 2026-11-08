"""B1: Ridge+doy and Ridge-patch (validation only).

One linear model per inner grid point and lead. Features at issue day t (all use days <= t only):
  per patch point: anomaly lags 0..6 and mean anomaly t-29..t (8, the existing Ridge features) ->  patch radius 0 = own point (8);
  patch radius 2 = 25 points x 8 = 200;  plus sin/cos of the day of year of t (2) when `doy`.
  ridge_doy  : radius 0, doy  -> 10 features;   ridge_patch: radius 2, doy -> 202 features;   radius 0 without doy = existing Ridge.
Penalty/standardisation exactly as ridge.py: min mean((y-b-Zw)^2) + alpha*|w|^2, Z standardised with TRAINING mean/std.
Training rows: the existing Ridge rule (targets in 1982-2015, complete 30-day history at the issue day).
Alpha: 13-value grid, one per region and lead, chosen on validation RMSE (SST, cos-lat weighted, inner points) over the LSTM validation pairs.

  python -m sst.b1_ridge        -> results/b1_ridge_val.json, data/derived/b1_ridge_<region>.npz
"""
import json
import subprocess
import time

import numpy as np
import pandas as pd
import xarray as xr

from sst.b1_data import inner_block, load_b1, load_halo_sst, make_data_b1
from sst.climatology import anomaly, clim_at
from sst.download import ROOT, load_config
from sst.lstm_data import day_features
from sst.metrics import rmse
from sst.ridge import ALPHA_GRID, _pairs, make_features

DERIVED = ROOT / "data" / "derived"


def point_block(F, i, j, halo, radius):
    """Features of the patch around inner point (i, j). F: (n, n_lat_grid, n_lon_grid, 8). Order: (d_lat, d_lon, feature)."""
    r0, c0 = i + halo, j + halo
    return F[:, r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1, :].reshape(F.shape[0], -1)


def fit_point(X, y, alphas):
    """Same maths as ridge.fit_ridge for ONE point, with BLAS matmuls (X (n, f), y (n,)). Returns mu, sd, ymean, coef (A, f)."""
    n = X.shape[0]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    ym = y.mean()
    Z = (X - mu) / sd
    G = Z.T @ Z / n
    b = Z.T @ (y - ym) / n
    eye = np.eye(X.shape[1])
    coef = np.stack([np.linalg.solve(G + al * eye, b) for al in alphas])
    return mu, sd, ym, coef


def predict_point(mu, sd, ym, coef, X):
    return ((X - mu) / sd) @ coef.T + ym        # (m, A)


def fit_predict(Ftr, ytr, Fva, dtr, dva, radius, halo, n_inner, alphas, doy=True, log=None):
    """Loop over inner points. Ftr (n, G_lat, G_lon, 8), ytr (n, P) anomalies, Fva (m, ...), dtr/dva (n|m, 2) doy features.
    Returns pred anomalies (A, m, P) in deg C."""
    nlat, nlon = n_inner
    P = nlat * nlon
    pred = np.empty((len(alphas), Fva.shape[0], P))
    for p in range(P):
        i, j = divmod(p, nlon)
        Xtr, Xva = point_block(Ftr, i, j, halo, radius), point_block(Fva, i, j, halo, radius)
        if doy:
            Xtr, Xva = np.concatenate([Xtr, dtr], axis=1), np.concatenate([Xva, dva], axis=1)
        assert not np.isnan(Xtr).any() and not np.isnan(ytr[:, p]).any() and not np.isnan(Xva).any()
        mu, sd, ym, coef = fit_point(Xtr, ytr[:, p], alphas)
        pred[:, :, p] = predict_point(mu, sd, ym, coef, Xva).T
        if log and p % 400 == 399:
            log(f"    point {p + 1}/{P}")
    return pred


def as_da(values, sst_inner, times):
    return xr.DataArray(values, dims=("time", "latitude", "longitude"),
                        coords={"time": pd.DatetimeIndex(times), "latitude": sst_inner["latitude"], "longitude": sst_inner["longitude"]})


def run_region(region, cfg, b1, models=(("ridge_doy", 0, True), ("ridge_patch", 2, True)), log=print, sst=None):
    """Returns (results dict, predictions dict). Reads only data up to the validation end."""
    halo = b1["halo"]["width_points"]
    sst = load_halo_sst(f"{region}_halo", cfg) if sst is None else sst   # `sst` given only by tests (synthetic data)
    data = make_data_b1(sst, cfg, halo, b1["patch"]["size"])
    P = data.n_points
    nlat, nlon = data.n_inner
    iss_idx = np.unique(data.val_pairs // P)
    assert len(data.val_pairs) == len(iss_idx) * P
    issue = data.dates[iss_idx]
    leads = list(cfg["leads_days"])
    clim = data.clim
    train = cfg["split"]["train"]
    sst_tr = sst.sel(time=slice(*train))
    anom_full = anomaly(sst, clim).reindex(time=pd.date_range(sst.time.min().values, sst.time.max().values))
    anom_tr = anomaly(sst_tr, clim).reindex(time=pd.date_range(sst_tr.time.min().values, sst_tr.time.max().values))
    a_full = anom_full.values.reshape(anom_full.sizes["time"], -1)
    a_tr = anom_tr.values.reshape(anom_tr.sizes["time"], -1)
    t0_full, t0_tr = anom_full.time.values[0], anom_tr.time.values[0]
    sst_inner = inner_block(sst, halo)
    G = a_full.shape[1]
    gshape = (anom_full.sizes["latitude"], anom_full.sizes["longitude"])
    # features at the validation issue dates (days <= issue only)
    Fva_all = make_features(a_full, t0_full, issue).reshape(len(issue), *gshape, -1)
    dva = day_features(issue).astype(float)
    results, preds = {m: {} for m, _, _ in models}, {m: {} for m, _, _ in models}
    for h in leads:
        i_tr, t_tr, c_tr = _pairs(sst_tr.time.values, train[0], train[1], h)
        assert pd.Timestamp(t_tr.max()) <= pd.Timestamp(train[1])
        y_tr = inner_block(anom_tr.sel(time=t_tr).values.reshape(len(t_tr), *gshape), halo).reshape(len(t_tr), P)
        Ftr = make_features(a_tr, t0_tr, i_tr).reshape(len(i_tr), *gshape, -1)
        dtr = day_features(i_tr).astype(float)
        tgt = issue + pd.Timedelta(days=h)
        obs = sst_inner.sel(time=tgt)
        ct = clim_at(clim, tgt).values.reshape(len(tgt), *gshape)
        ct_inner = inner_block(ct, halo)
        for m, radius, doy in models:
            t1 = time.time()
            pred = fit_predict(Ftr, y_tr, Fva_all, dtr, dva, radius, halo, data.n_inner, ALPHA_GRID, doy=doy, log=log)
            pred = pred.reshape(len(ALPHA_GRID), len(tgt), nlat, nlon)
            curve = [rmse(as_da(ct_inner + pred[k], sst_inner, tgt), obs) for k in range(len(ALPHA_GRID))]
            best = int(np.argmin(curve))
            results[m][str(h)] = {"alphas": [float(a) for a in ALPHA_GRID], "val_rmse": [float(c) for c in curve],
                                  "alpha": float(ALPHA_GRID[best]), "alpha_index": best, "alpha_at_grid_edge": best in (0, len(ALPHA_GRID) - 1),
                                  "n_train_rows": int(len(i_tr)), "n_features": int(point_block(Ftr[:1], 0, 0, halo, radius).shape[1] + (2 if doy else 0)),
                                  "train_pairs": c_tr, "seconds_not_a_result": time.time() - t1}
            preds[m][str(h)] = pred[best]
            log(f"[{region} {m} h={h}] alpha={ALPHA_GRID[best]:g} val RMSE {curve[best]:.4f}  (grid edge: {best in (0, len(ALPHA_GRID) - 1)})")
        del Ftr
    return results, preds, issue


def main():
    cfg, b1 = load_config(), load_b1()
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = {"definition": __doc__, "alpha_grid": [float(a) for a in ALPHA_GRID], "selected_on": cfg["split"]["val"], "regions": {}}
    for r in cfg["regions"]:
        res, preds, issue = run_region(r, cfg, b1)
        arrays = {f"{m}_h{h}": p for m, d in preds.items() for h, p in d.items()}
        np.savez(DERIVED / f"b1_ridge_{r}.npz", issue=issue.values.astype("datetime64[D]").astype(str), **arrays)
        out["regions"][r] = res
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out["git_commit"] = git("rev-parse", "HEAD")
    out["git_dirty"] = bool(git("status", "--porcelain", "--", "src/sst", "config.yaml", "config_b1.yaml"))
    (ROOT / "results" / "b1_ridge_val.json").write_text(json.dumps(out, indent=1, allow_nan=False))
    print("wrote results/b1_ridge_val.json")


if __name__ == "__main__":
    main()
