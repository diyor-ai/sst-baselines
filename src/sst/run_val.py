"""Run all baselines on the VALIDATION period only and write results/val_baselines.json.

Data after the validation end is dropped from memory before anything is computed, so the
test period cannot leak into the climatology, r(h), OCN or the scores.
"""
import json

import numpy as np
import pandas as pd
import xarray as xr

from sst.baselines import (anomaly_persistence, climatology_forecast, damped_persistence, lag_corr, ocn,
                           persistence)
from sst.climatology import build_climatology, clim_at
from sst.download import ROOT, load_config
from sst.metrics import scores, scores_by, season_labels, skill_score, valid_pairs, year_labels

METHODS = ["persistence", "climatology", "anomaly_persistence", "damped_persistence", "ocn"]


def _clean(x):
    """NaN -> None so the JSON file is strictly valid (ACC of a zero-anomaly forecast is NaN by design)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return None if isinstance(x, float) and x != x else x


def run_region(sst, cfg, clim, region):
    train = cfg["split"]["train"]
    v0, v1 = cfg["split"]["val"]
    out = {}
    for h in cfg["leads_days"]:
        issue, target, counts = valid_pairs(sst.time.values, v0, v1, h)
        assert target.min() >= pd.Timestamp(v0) and target.max() <= pd.Timestamp(v1)
        r = lag_corr(sst, clim, train, h)
        fcs = {
            "persistence": persistence(sst, issue, target),
            "climatology": climatology_forecast(sst, clim, issue, target),
            "anomaly_persistence": anomaly_persistence(sst, clim, issue, target),
            "damped_persistence": damped_persistence(sst, clim, issue, target, r),
            "ocn": ocn(sst, issue, target, cfg["ocn_years"]),
        }
        obs = sst.sel(time=target)
        ct = clim_at(clim, target)
        seasons, years = season_labels(target), year_labels(target)
        res = {m: {"overall": scores(f, obs, ct), "by_season": scores_by(f, obs, ct, seasons),
                   "by_year": scores_by(f, obs, ct, years)} for m, f in fcs.items()}
        skill = {}
        for m in METHODS:
            skill[m] = {}
            for ref in ("climatology", "persistence"):
                for k in ("rmse", "mae"):
                    skill[m][f"{k}_skill_vs_{ref}"] = skill_score(res[m]["overall"][k], res[ref]["overall"][k])
                skill[m][f"rmse_skill_vs_{ref}_by_season"] = {
                    s: skill_score(res[m]["by_season"][s]["rmse"], res[ref]["by_season"][s]["rmse"])
                    for s in res[m]["by_season"]}
        out[str(h)] = {"pairs": counts, "r_h_cosw_mean": float(np.average(
            r.mean("longitude").values, weights=np.cos(np.deg2rad(r["latitude"].values)))), "methods": res, "skill": skill}
        print(f"[{region} h={h}] pairs {counts['used']}/{counts['expected']}  " + "  ".join(
            f"{m}: rmse {res[m]['overall']['rmse']:.3f}" for m in METHODS), flush=True)
    return out


def main():
    cfg = load_config()
    v1 = cfg["split"]["val"][1]
    result = {
        "period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["val"]},
        "climatology": {"years": "train", "smoothing": cfg["climatology"]["smoothing"], "feb29": "use_feb28"},
        "ocn_years": cfg["ocn_years"], "leads_days": cfg["leads_days"],
        "acc": "acc = temporal ACC (per grid point, centred correlation over time of forecast vs observed "
               "anomalies w.r.t. the training climatology, cos(lat)-weighted mean); acc_spatial = daily spatial ACC",
        "skill": "1 - error_model/error_reference; reference = climatology or persistence",
        "regions": {},
    }
    for region in cfg["regions"]:
        sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"]
        sst = sst.sel(time=slice(None, v1)).load()  # drop everything after the validation end
        assert sst.time.max() <= pd.Timestamp(v1)
        clim = build_climatology(sst, cfg)
        result["regions"][region] = run_region(sst, cfg, clim, region)
    (ROOT / "results" / "val_baselines.json").write_text(json.dumps(_clean(result), indent=2, allow_nan=False))
    print("wrote results/val_baselines.json")


if __name__ == "__main__":
    main()
