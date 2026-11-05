"""Validation (2016-2019) scores of the trained LSTMs next to anomaly persistence and Ridge, on IDENTICAL pairs.

Writes results/lstm_val.json and results/figures/figL1_lstm_curves.png. Only data up to the validation end is loaded.
Pairs = the LSTM validation samples: issue date in 2016-2019, all three targets (t+1, t+7, t+30) in 2016-2019,
30-day history complete. The baselines are re-scored on exactly these issue dates (target = issue + lead).
"""
import json

import numpy as np
import pandas as pd
import torch
import xarray as xr

from sst.baselines import anomaly_persistence, climatology_forecast
from sst.climatology import clim_at
from sst.download import ROOT, load_config
from sst.lstm_data import make_data
from sst.lstm_model import SSTLSTM
from sst.lstm_train import RUNS, load_sst, predict
from sst.metrics import scores, scores_by, season_labels, skill_score
from sst.ridge import ridge_lead
from sst.run_val import _clean

RESULTS = ROOT / "results"


def as_da(values, sst, target):
    return xr.DataArray(values, dims=("time", "latitude", "longitude"),
                        coords={"time": pd.DatetimeIndex(target), "latitude": sst["latitude"], "longitude": sst["longitude"]})


def score_block(f, obs, ct, seasons):
    return {"overall": scores(f, obs, ct), "by_season": scores_by(f, obs, ct, seasons)}


def add_skill(block, ref):
    """RMSE skill 1 - rmse_model/rmse_ref overall and per season."""
    return {"overall": skill_score(block["overall"]["rmse"], ref["overall"]["rmse"]),
            "by_season": {g: skill_score(block["by_season"][g]["rmse"], ref["by_season"][g]["rmse"]) for g in block["by_season"]}}


def agg(dicts):
    """mean and std (sample std, ddof=1) over a list of identically-shaped nested dicts of numbers."""
    if isinstance(dicts[0], dict):
        m = {k: agg([d[k] for d in dicts]) for k in dicts[0]}
        return {"mean": {k: v["mean"] for k, v in m.items()}, "std": {k: v["std"] for k, v in m.items()}}
    x = np.array([np.nan if v is None else v for v in dicts], dtype=float)
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1))}


def evaluate_region(region, cfg):
    l = cfg["lstm"]
    sst = load_sst(region, cfg)
    data = make_data(sst, cfg)
    P = data.n_points
    clim = data.clim
    iss_idx = np.unique(data.val_pairs // P)
    assert len(data.val_pairs) == len(iss_idx) * P, "land/missing points are not expected in these regions"
    issue = data.dates[iss_idx]
    leads = list(cfg["leads_days"])
    # reference forecasts on the same pairs
    ridge = {}
    for h in leads:
        f_r, (_, t_r), info = ridge_lead(sst, clim, cfg, h, period="val")
        ridge[h] = (f_r.sel(time=issue + pd.Timedelta(days=h)), info["alpha"])
    ref, obs_all = {}, {}
    for h in leads:
        tgt = issue + pd.Timedelta(days=h)
        obs, ct, seas = sst.sel(time=tgt), clim_at(clim, tgt), season_labels(tgt)
        obs_all[h] = (obs, ct, seas)
        ref[h] = {"anomaly_persistence": score_block(anomaly_persistence(sst, clim, issue, tgt), obs, ct, seas),
                  "climatology": score_block(climatology_forecast(sst, clim, issue, tgt), obs, ct, seas),
                  "ridge": score_block(ridge[h][0], obs, ct, seas)}
    out = {"n_issue_dates": int(len(issue)), "first_issue": str(issue[0].date()), "last_issue": str(issue[-1].date()),
           "std_deg_c": data.std, "ridge_alpha": {str(h): ridge[h][1] for h in leads},
           "reference": {str(h): ref[h] for h in leads}, "lstm": {}}
    for H in l["hidden_sizes"]:
        per_seed = {}
        for s in l["seeds"]:
            model = SSTLSTM(H)
            model.load_state_dict(torch.load(ROOT / "models" / f"{region}_H{H}_s{s}.pt"))
            pred = predict(model, data, data.val_pairs).reshape(len(issue), data.shape[0], data.shape[1], len(leads))
            res = {}
            for k, h in enumerate(leads):
                obs, ct, seas = obs_all[h]
                f = as_da(ct.values + pred[..., k] * data.std, sst, issue + pd.Timedelta(days=h))
                blk = score_block(f, obs, ct, seas)
                blk["skill_vs_anomaly_persistence"] = add_skill(blk, ref[h]["anomaly_persistence"])
                blk["skill_vs_ridge"] = add_skill(blk, ref[h]["ridge"])
                res[str(h)] = blk
            per_seed[str(s)] = res
            print(f"[{region} H={H} seed={s}] " + "  ".join(f"h={h}: {res[str(h)]['overall']['rmse']:.3f}" for h in leads), flush=True)
        out["lstm"][str(H)] = {"per_seed": per_seed, **agg(list(per_seed.values()))}
    sel = {H: float(np.mean([out["lstm"][H]["mean"][str(h)]["overall"]["rmse"] for h in leads])) for H in out["lstm"]}
    out["selection"] = {"rule": l["selection"], "mean_rmse_over_seeds_and_leads": sel, "chosen_H": int(min(sel, key=sel.get))}
    return out


def curves_figure(cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    l = cfg["lstm"]
    regions = list(cfg["regions"])
    fig, axes = plt.subplots(len(regions), len(l["hidden_sizes"]), figsize=(10, 7), sharex=True, squeeze=False)
    styles = {0: "-", 1: "--", 2: ":"}
    for i, r in enumerate(regions):
        for j, H in enumerate(l["hidden_sizes"]):
            ax = axes[i, j]
            for s in l["seeds"]:
                h = json.loads((RUNS / f"{r}_H{H}_s{s}.json").read_text())
                ep = np.arange(1, len(h["history"]["val_loss"]) + 1)
                ax.plot(ep, h["history"]["train_loss"], styles[s], color="#4477aa", label=f"train, seed {s}")
                ax.plot(ep, h["history"]["val_loss"], styles[s], color="#cc6677", label=f"validation, seed {s}")
                ax.plot(h["best_epoch"], h["history"]["val_loss"][h["best_epoch"] - 1], "o", color="#cc6677", ms=5)
            ax.set_title(f"{r}, hidden size {H}")
            ax.grid(alpha=.3)
            if j == 0:
                ax.set_ylabel("MSE (standardised units)")
            if i == len(regions) - 1:
                ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("LSTM learning curves (dot = best validation epoch, kept as checkpoint)")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    cfg = load_config()
    out = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["val"]}, "leads_days": cfg["leads_days"],
           "definitions": {
               "pairs": "issue date in 2016-2019, all three targets in 2016-2019, 30-day history complete; baselines re-scored on the same pairs",
               "skill": "1 - rmse_model/rmse_reference, references anomaly_persistence and ridge (alphas chosen on validation, fitted on 1982-2015)",
               "mean_std": "mean and sample std (ddof=1) over seeds 0,1,2",
               "forecast_sst": "climatology(t+h) + predicted standardised anomaly * region training anomaly std"},
           "regions": {}}
    for r in cfg["regions"]:
        out["regions"][r] = evaluate_region(r, cfg)
    (RESULTS / "lstm_val.json").write_text(json.dumps(_clean(out), indent=1, allow_nan=False))
    curves_figure(cfg, RESULTS / "figures" / "figL1_lstm_curves.png")
    print("wrote results/lstm_val.json and results/figures/figL1_lstm_curves.png")


if __name__ == "__main__":
    main()
