"""Validation (2016-2019) for ALL baselines + Ridge -> results/val_all.json, results/val_bias.json,
results/ridge_alphas.json. Data after the validation end is dropped before anything is computed."""
import json

import numpy as np
import pandas as pd
import xarray as xr

from sst.baselines import (anomaly_persistence, climatology_forecast, damped_persistence, lag_corr, ocn,
                           persistence)
from sst.bias import area_mean_series, bias_decomposition
from sst.climatology import anomaly, build_climatology, clim_at
from sst.download import ROOT, load_config
from sst.metrics import scores, scores_by, season_labels, skill_score, valid_pairs, year_labels
from sst.ridge import ALPHA_GRID, ridge_lead
from sst.run_val import _clean

METHODS = ["persistence", "climatology", "anomaly_persistence", "damped_persistence", "ocn", "ridge"]
REFS = {"climatology": "climatology", "anomaly_persistence": "anomaly_persistence"}


def run_region(sst, cfg, clim, region):
    train = cfg["split"]["train"]
    v0, v1 = cfg["split"]["val"]
    out, bias, ridge_info = {}, {}, {}
    for h in cfg["leads_days"]:
        issue, target, counts = valid_pairs(sst.time.values, v0, v1, h)
        r = lag_corr(sst, clim, train, h)
        fcs = {
            "persistence": persistence(sst, issue, target),
            "climatology": climatology_forecast(sst, clim, issue, target),
            "anomaly_persistence": anomaly_persistence(sst, clim, issue, target),
            "damped_persistence": damped_persistence(sst, clim, issue, target, r),
            "ocn": ocn(sst, issue, target, cfg["ocn_years"]),
        }
        f_ridge, (i_r, t_r), info = ridge_lead(sst, clim, cfg, h)
        assert (t_r == target).all() and (i_r == issue).all(), "Ridge must be scored on the same pairs"
        fcs["ridge"] = f_ridge
        ridge_info[str(h)] = info
        obs, ct = sst.sel(time=target), clim_at(clim, target)
        seasons, years = season_labels(target), year_labels(target)
        res = {m: {"overall": scores(f, obs, ct), "by_season": scores_by(f, obs, ct, seasons),
                   "by_year": scores_by(f, obs, ct, years)} for m, f in fcs.items()}
        skill = {}
        for m in METHODS:
            skill[m] = {}
            for ref in REFS:
                for k in ("rmse", "mae"):
                    skill[m][f"{k}_skill_vs_{ref}"] = skill_score(res[m]["overall"][k], res[ref]["overall"][k])
                for grp in ("by_season", "by_year"):
                    skill[m][f"rmse_skill_vs_{ref}_{grp}"] = {
                        g: skill_score(res[m][grp][g]["rmse"], res[ref][grp][g]["rmse"]) for g in res[m][grp]}
        out[str(h)] = {"pairs": counts, "methods": res, "skill": skill}
        bias[str(h)] = {m: bias_decomposition(f, obs) for m, f in fcs.items()}
        for m in METHODS:  # consistency: MSE from the bias split == RMSE^2 from the metric code
            assert abs(bias[str(h)][m]["mse"] - res[m]["overall"]["rmse"] ** 2) < 1e-9
        print(f"[{region} h={h}] ridge alpha={info['alpha']:g}  " + "  ".join(
            f"{m}: {res[m]['overall']['rmse']:.3f}" for m in METHODS), flush=True)
    return out, bias, ridge_info


def main():
    cfg = load_config()
    v1 = cfg["split"]["val"][1]
    base = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["val"]},
            "climatology": {"years": "train", "smoothing": cfg["climatology"]["smoothing"], "feb29": "use_feb28"},
            "leads_days": cfg["leads_days"]}
    val_all = {**base, "methods": METHODS,
               "definitions": {
                   "acc": "temporal ACC (primary): per grid point centred correlation over time of forecast vs observed "
                          "anomalies (training climatology), cos(lat)-weighted mean; null = undefined (zero-anomaly forecast)",
                   "acc_spatial": "secondary: daily spatial correlation of anomalies, averaged over days",
                   "skill": "1 - error_model/error_reference for RMSE/MAE; references: climatology and anomaly_persistence",
                   "ridge": "per grid point and lead; features lags 0..6 + mean(t-29..t) of anomalies; fit on 1982-2015; "
                            "alpha chosen on validation RMSE (one per region and lead)"},
               "regions": {}}
    val_bias = {**base,
                "definitions": "bias = cos(lat)-weighted mean of (forecast - observed) over all validation dates and grid "
                               "points; mse = weighted mean squared error; variance_part = mse - bias^2",
                "bias_by_region": {}, "area_mean_anomaly": {}}
    ridge_out = {"alpha_grid": [float(a) for a in ALPHA_GRID], "fit_years": cfg["split"]["train"],
                 "selected_on": cfg["split"]["val"],
                 "final_model": "fitted on TRAINING years only with the chosen alpha (no refit on validation)",
                 "penalty": "mean squared error + alpha*|w|^2 on standardised features (training mean/std)",
                 "regions": {}}
    for region in cfg["regions"]:
        sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"]
        sst = sst.sel(time=slice(None, v1)).load()  # test period is never loaded into memory
        assert sst.time.max() <= pd.Timestamp(v1)
        clim = build_climatology(sst, cfg)
        res, bias, ridge_info = run_region(sst, cfg, clim, region)
        val_all["regions"][region] = res
        val_bias["bias_by_region"][region] = bias
        ridge_out["regions"][region] = ridge_info
        am = area_mean_series(anomaly(sst, clim))
        s = pd.Series(am, index=pd.DatetimeIndex(sst.time.values))
        val_bias["area_mean_anomaly"][region] = {
            "definition": "cos(lat)-weighted area-mean of SST minus TRAINING climatology (harmonics3)",
            "train_end": cfg["split"]["train"][1], "val_start": cfg["split"]["val"][0],
            "mean_train": float(s[:cfg["split"]["train"][1]].mean()), "mean_val": float(s[cfg["split"]["val"][0]:].mean()),
            "yearly_mean": {str(y): float(v) for y, v in s.groupby(s.index.year).mean().items()},
            "dates": [d.strftime("%Y-%m-%d") for d in s.index], "daily": [round(float(v), 4) for v in s.values]}
    for name, obj in (("val_all", val_all), ("val_bias", val_bias), ("ridge_alphas", ridge_out)):
        (ROOT / "results" / f"{name}.json").write_text(json.dumps(_clean(obj), indent=1, allow_nan=False))
        print(f"wrote results/{name}.json")


if __name__ == "__main__":
    main()
