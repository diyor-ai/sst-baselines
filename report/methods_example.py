"""Worked example for report/methods_note.md: one Kuroshio grid point, one TEST issue date, lead 7 d.

Every number is computed by calling the project's own functions (sst.baselines, sst.climatology, sst.ridge)
and then re-derived by hand-written steps; the script asserts that both agree. Nothing is refitted for
evaluation: the climatology, r(h) and the Ridge model are the training-year (1982-2015) objects used in the
test run, with the alpha stored in config.yaml. Output: results/methods_example.json.

Run:  python report/methods_example.py
"""
import json

import numpy as np
import pandas as pd
import xarray as xr

from sst.baselines import (anomaly_persistence, climatology_forecast, damped_persistence, lag_corr, ocn,
                           persistence)
from sst.climatology import anomaly, build_climatology, clim_at, doy365, raw_doy_mean
from sst.download import ROOT, load_config
from sst.ridge import (ALPHA_GRID, MEAN_WIN, N_LAGS, _pairs, fit_ridge, make_features, predict_ridge, ridge_lead)

REGION, LAT, LON = "kuroshio", 35.125, 147.125
ISSUE, H = pd.Timestamp("2021-03-15"), 7


def main():
    cfg = load_config()
    train = cfg["split"]["train"]
    sst = xr.open_dataset(ROOT / "data" / "processed" / f"{REGION}.nc")["sst"].load()
    clim = build_climatology(sst, cfg)                       # training years only, 3 harmonics
    target = ISSUE + pd.Timedelta(days=H)
    ij = (int(np.argmin(abs(sst.latitude.values - LAT))), int(np.argmin(abs(sst.longitude.values - LON))))
    pt = lambda da: da.isel(latitude=ij[0], longitude=ij[1])  # noqa: E731
    iss, tgt = pd.DatetimeIndex([ISSUE]), pd.DatetimeIndex([target])
    out = {"region": REGION, "lat": float(sst.latitude[ij[0]]), "lon": float(sst.longitude[ij[1]]),
           "issue": str(ISSUE.date()), "lead": H, "target": str(target.date()), "train": train}

    # --- climatology: raw day-of-year mean, harmonic fit, coefficients ------------------------------------
    raw = pt(raw_doy_mean(sst, train)).values                      # (365,)
    d = np.arange(365)
    X = np.stack([np.ones(365)] + [f(2 * np.pi * k * d / 365) for k in (1, 2, 3) for f in (np.cos, np.sin)], axis=1)
    coef, *_ = np.linalg.lstsq(X, raw, rcond=None)
    fit = X @ coef
    assert np.allclose(fit, pt(clim).values, atol=1e-10)
    di, dt = int(doy365(iss)[0]), int(doy365(tgt)[0])
    c_i, c_t = float(fit[di]), float(fit[dt])
    assert np.isclose(c_i, float(pt(clim_at(clim, iss)).values[0])) and np.isclose(c_t, float(pt(clim_at(clim, tgt)).values[0]))
    out["clim"] = {"coef": coef.tolist(), "d_issue": di + 1, "d_target": dt + 1,
                   "raw_issue": float(raw[di]), "raw_target": float(raw[dt]), "c_issue": c_i, "c_target": c_t,
                   "rms_raw_minus_fit": float(np.sqrt(np.mean((raw - fit) ** 2))),
                   "n_train_years": 34}

    s_t = float(pt(sst).sel(time=ISSUE).values); s_obs = float(pt(sst).sel(time=target).values)
    a_t = s_t - c_i
    out.update(s_t=s_t, s_obs=s_obs, a_t=a_t)

    # --- damped persistence r(h): sums over training pairs (t, t+h), uncentred ----------------------------
    a_tr = pt(anomaly(sst.sel(time=slice(*train)), clim))
    t0 = a_tr.time.to_index(); t1 = t0 + pd.Timedelta(days=H); keep = t1.isin(t0)
    x, y = a_tr.sel(time=t0[keep]).values, a_tr.sel(time=t1[keep]).values
    num, sxx, syy = float((x * y).sum()), float((x * x).sum()), float((y * y).sum())
    r_h = num / np.sqrt(sxx * syy)
    assert np.isclose(r_h, float(pt(lag_corr(sst, clim, train, H)).values))
    out["r"] = {"r": r_h, "n_pairs": int(keep.sum()), "num": num, "sxx": sxx, "syy": syy,
                "mean_x": float(x.mean()), "mean_y": float(y.mean())}

    # --- OCN: same calendar day as the TARGET, 1..10 years before the target year -------------------------
    dates = [target - pd.DateOffset(years=k) for k in range(1, cfg["ocn_years"] + 1)]
    vals = [float(pt(sst).sel(time=dd).values) for dd in dates]
    assert all(dd < ISSUE for dd in dates)
    m_ocn = float(np.mean(vals))
    assert np.isclose(m_ocn, float(pt(ocn(sst, iss, tgt, cfg["ocn_years"])).values[0]))
    out["ocn"] = {"dates": [str(dd.date()) for dd in dates], "values": vals, "mean": m_ocn,
                  "latest_date": str(max(dates).date()), "gap_to_issue_days": int((ISSUE - max(dates)).days)}

    # --- Ridge: same steps as sst.ridge.ridge_lead, for this one grid point -------------------------------
    alpha = float(cfg["ridge"]["alphas"][REGION][H])
    sst_tr = sst.sel(time=slice(*train))
    full = lambda da: pt(da).reindex(time=pd.date_range(da.time.min().values, da.time.max().values)).values[:, None]  # noqa: E731
    an_full, an_tr = anomaly(sst, clim), anomaly(sst_tr, clim)
    A_full, A_tr = full(an_full), full(an_tr)
    i_tr, t_tr, cnt = _pairs(sst_tr.time.values, train[0], train[1], H)
    y_tr = pt(an_tr).reindex(time=an_tr.time).sel(time=t_tr).values[:, None]
    Xtr = make_features(A_tr, an_tr.time.values[0], i_tr)
    model = fit_ridge(Xtr, y_tr, [alpha])
    Xi = make_features(A_full, an_full.time.values[0], iss)       # (1, 1, 8) raw anomaly features at the issue date
    z = (Xi[0, 0] - model["mu"][0]) / model["sd"][0]
    w = model["coef"][0, 0]
    a_hat = float(model["ymean"][0] + z @ w)
    assert np.isclose(a_hat, float(predict_ridge(model, Xi)[0, 0, 0]))
    f_ridge = c_t + a_hat
    # manual check of the feature definitions straight from the anomaly series
    an_pt = pt(an_full)
    lags = [float(an_pt.sel(time=ISSUE - pd.Timedelta(days=k))) for k in range(N_LAGS)]
    m30 = float(an_pt.sel(time=slice(ISSUE - pd.Timedelta(days=MEAN_WIN - 1), ISSUE)).mean())
    assert np.allclose(lags + [m30], Xi[0, 0])
    # and against the full-region production code
    f_all, _, info = ridge_lead(sst, clim, cfg, H, period="test", alpha=alpha)
    assert np.isclose(float(pt(f_all).sel(time=target).values), f_ridge, atol=1e-8)
    names = [f"a(t-{k})" if k else "a(t)" for k in range(N_LAGS)] + ["mean a(t-29..t)"]
    out["ridge"] = {"alpha": alpha, "n_train_rows": int(len(i_tr)), "features": names, "x": Xi[0, 0].tolist(),
                    "mu": model["mu"][0].tolist(), "sd": model["sd"][0].tolist(), "z": z.tolist(),
                    "w": w.tolist(), "terms": (z * w).tolist(), "intercept": float(model["ymean"][0]),
                    "a_hat": a_hat, "alpha_grid_min": float(min(ALPHA_GRID))}

    # --- the six forecasts from the project's vectorised functions, compared with the hand steps ----------
    r_map = lag_corr(sst, clim, train, H)
    fc = {"persistence": persistence(sst, iss, tgt), "climatology": climatology_forecast(sst, clim, iss, tgt),
          "anomaly_persistence": anomaly_persistence(sst, clim, iss, tgt),
          "damped_persistence": damped_persistence(sst, clim, iss, tgt, r_map), "ocn": ocn(sst, iss, tgt, cfg["ocn_years"])}
    code = {k: float(pt(v).values[0]) for k, v in fc.items()}
    hand = {"persistence": s_t, "climatology": c_t, "anomaly_persistence": c_t + a_t,
            "damped_persistence": c_t + r_h * a_t, "ocn": m_ocn, "ridge": f_ridge}
    code["ridge"] = float(pt(f_all).sel(time=target).values)
    for k in hand:
        assert np.isclose(code[k], hand[k], atol=1e-8), k
    out["forecast"] = hand
    out["error"] = {k: v - s_obs for k, v in hand.items()}   # forecast - observed
    (ROOT / "results" / "methods_example.json").write_text(json.dumps(out, indent=1))
    print("wrote results/methods_example.json; all hand steps match the project code")


if __name__ == "__main__":
    main()
