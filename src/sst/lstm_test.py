"""TEST period (2020-2025) for the LSTM: the single, final evaluation -> results/lstm_test.json.

Protocol: docs/lstm_test_protocol.md (git tag `pre-lstm-test`). Safeguards: refuses to run unless HEAD is the tagged
commit, the working tree is clean, results/lstm_test.json does not exist, and the weights hash of each of the six
H=64 checkpoints equals the one in results/lstm_manifest.json. Only those six checkpoints are loaded.
Pairs = exactly the frozen baselines' pairs (targets every day of 2020-2025, issue = target - lead; the 30-day history
may reach into 2019). Anomaly persistence and Ridge are recomputed on them and must equal results/test_all.json.
"""
import json
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from sst.baselines import anomaly_persistence
from sst.climatology import clim_at
from sst.download import ROOT, load_config
from sst.lstm_data import gather_inputs, make_data
from sst.lstm_eval import add_skill, agg, as_da, score_block
from sst.lstm_manifest import weights_sha
from sst.lstm_model import SSTLSTM
from sst.metrics import season_labels, valid_pairs
from sst.ridge import ridge_lead
from sst.run_val import _clean

TAG = "pre-lstm-test"
H = 64   # chosen on validation (results/lstm_val.json, rule in config.yaml)
OUT = ROOT / "results" / "lstm_test.json"


def _git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def check_guards():
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode != 0:
        raise SystemExit(f"tag {TAG} missing")
    if _git("rev-parse", "HEAD").stdout != _git("rev-parse", f"{TAG}^{{commit}}").stdout:
        raise SystemExit("HEAD is not the tagged commit: refusing to run")
    if _git("status", "--porcelain").stdout.strip():
        raise SystemExit("working tree is dirty: refusing to run")
    if OUT.exists():
        raise SystemExit("results/lstm_test.json exists: the test period is evaluated exactly once")


def load_checked_models(region, cfg):
    man = json.loads((ROOT / "results" / "lstm_manifest.json").read_text())["runs"]
    models = {}
    for s in cfg["lstm"]["seeds"]:
        name = f"{region}_H{H}_s{s}"
        path = ROOT / "models" / f"{name}.pt"
        if weights_sha(path) != man[name]["weights_sha256"]:
            raise SystemExit(f"weights hash of {name} differs from the manifest")
        m = SSTLSTM(H)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[s] = m.eval()
    return models


def predict_inputs(model, data, flat, chunk=32768):
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), chunk):
            x = gather_inputs(data, flat[i:i + chunk])
            assert not np.isnan(x).any(), "incomplete history at an issue date"
            out.append(model(torch.from_numpy(x)).numpy())
    return np.concatenate(out)


def evaluate_region(region, cfg, test_all, last_day=None):
    """`test_all`: frozen results/test_all.json (the recomputed baselines must equal it); None skips that check
    (only used by a dry run on validation data, where cfg["split"]["test"] is pointed at the validation years and
    `last_day` cuts the data there)."""
    leads = list(cfg["leads_days"])
    alphas = {int(h): float(a) for h, a in cfg["ridge"]["alphas"][region].items()}
    sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"].sel(time=slice(None, last_day)).load()   # test period loaded HERE (unless dry run)
    data = make_data(sst, cfg, period="test")
    clim, P = data.clim, data.n_points
    models = load_checked_models(region, cfg)
    t0, t1 = cfg["split"]["test"]
    out = {"reference": {}, "lstm": {"per_seed": {str(s): {} for s in models}}}
    pairs = {}
    for k, h in enumerate(leads):
        issue, target, counts = valid_pairs(sst.time.values, t0, t1, h)
        f_r, (i_r, t_r), info = ridge_lead(sst, clim, cfg, h, period="test", alpha=alphas[h])
        assert (i_r == issue).all() and (t_r == target).all(), "Ridge pairs must equal the baseline pairs"
        obs, ct, seas = sst.sel(time=target), clim_at(clim, target), season_labels(target)
        ref = {"anomaly_persistence": score_block(anomaly_persistence(sst, clim, issue, target), obs, ct, seas),
               "ridge": score_block(f_r, obs, ct, seas)}
        for m in ref if test_all is not None else []:   # must reproduce the frozen test numbers exactly
            frozen = test_all["regions"][region][str(h)]["methods"][m]["overall"]["rmse"]
            assert abs(frozen - ref[m]["overall"]["rmse"]) < 1e-9, (region, h, m)
        out["reference"][str(h)] = ref
        pairs[str(h)] = {"n_pairs": int(len(target)), **{k2: v for k2, v in counts.items() if k2 != "lead_days"}}
        t_idx = np.searchsorted(data.dates.values, issue.values)
        assert (data.dates[t_idx] == issue).all()
        flat = (t_idx[:, None] * P + np.arange(P)[None, :]).ravel()
        for s, model in models.items():
            pred = predict_inputs(model, data, flat)[:, k].reshape(len(issue), data.shape[0], data.shape[1])
            f = as_da(ct.values + pred * data.std, sst, target)
            blk = score_block(f, obs, ct, seas)
            blk["skill_vs_anomaly_persistence"] = add_skill(blk, ref["anomaly_persistence"])
            blk["skill_vs_ridge"] = add_skill(blk, ref["ridge"])
            out["lstm"]["per_seed"][str(s)][str(h)] = blk
            print(f"[{region} H={H} seed={s} lead={h}] rmse {blk['overall']['rmse']:.4f}  AP {ref['anomaly_persistence']['overall']['rmse']:.4f}  Ridge {ref['ridge']['overall']['rmse']:.4f}", flush=True)
    out["lstm"].update(agg([out["lstm"]["per_seed"][s] for s in out["lstm"]["per_seed"]]))
    out["pairs"] = pairs
    return out


def main():
    check_guards()
    cfg = load_config()
    started = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    test_all = json.loads((ROOT / "results" / "test_all.json").read_text())
    res = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["test"]}, "leads_days": cfg["leads_days"],
           "protocol": "docs/lstm_test_protocol.md",
           "run": {"commit": _git("rev-parse", "HEAD").stdout.strip(), "tag": TAG, "started_utc": started, "hidden": H,
                   "seeds": cfg["lstm"]["seeds"]},
           "regions": {r: evaluate_region(r, cfg, test_all) for r in cfg["regions"]}}
    res["run"]["wall_seconds"] = time.time() - t0
    OUT.write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    print("wrote results/lstm_test.json")


if __name__ == "__main__":
    main()
