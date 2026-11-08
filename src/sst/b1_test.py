"""TEST period (2020-2025) for B1 (spatial context): the single, final evaluation -> results/b1_test.json.

Protocol: docs/b1_test_protocol.md (git tag `pre-b1-test`). Safeguards: refuses to run unless the tag exists, HEAD is the tagged
commit, the working tree is clean and results/b1_test.json does not exist; the weights hash of each LSTM-patch checkpoint, the
ridge alphas, and the sha256 of the source/config/data files must equal results/b1_manifest.json; anomaly persistence and the point
Ridge recomputed on the test pairs must equal results/test_all.json (to 1e-9).
Models: Ridge+doy, Ridge-patch (validation alphas, fitted on 1982-2015), LSTM-patch H=64 seeds 0,1,2 (existing checkpoints).
Pairs = exactly the frozen baselines' pairs (every target day of 2020-2025, issue = target - lead; history may reach into 2019).

  python -m sst.b1_test                       # THE test run (guarded, once)
  python -m sst.b1_test --dry-run OUT.json    # validation data only (test split := validation years, data cut at 2019-12-31)
"""
import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from sst.b1_data import gather_inputs_b1, inner_block, load_b1, make_data_b1
from sst.b1_ridge import as_da, fit_predict
from sst.b1_train import file_sha, weights_sha
from sst.baselines import anomaly_persistence
from sst.climatology import anomaly, build_climatology, clim_at
from sst.download import ROOT, load_config
from sst.lstm_data import day_features
from sst.lstm_eval import add_skill, agg, score_block
from sst.lstm_model import SSTLSTM
from sst.metrics import season_labels, valid_pairs
from sst.ridge import _pairs, make_features, ridge_lead
from sst.run_val import _clean

TAG = "pre-b1-test"
H = 64
OUT = ROOT / "results" / "b1_test.json"
NEW_RIDGES = (("ridge_doy", 0), ("ridge_patch", 2))   # (name, patch radius); both with sin/cos doy of the issue day


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
        raise SystemExit("results/b1_test.json exists: the test period is evaluated exactly once")


def load_manifest():
    return json.loads((ROOT / "results" / "b1_manifest.json").read_text())


def verify_provenance(man):
    """Source/config/data files must be byte-identical to those recorded when the models were trained/evaluated."""
    bad = {f: h for f, h in {**man["code_sha256_at_manifest"], **man["data_sha256"]}.items() if file_sha(f) != h}
    if bad:
        raise SystemExit(f"files differ from results/b1_manifest.json: {sorted(bad)}")


def validation_alphas(man, region, cfg):
    """{model: {lead: alpha}} from the manifest; must equal results/b1_ridge_val.json."""
    rv = json.loads((ROOT / "results" / "b1_ridge_val.json").read_text())["regions"][region]
    out = {}
    for m, _ in NEW_RIDGES:
        out[m] = {h: float(man["ridge_runs"][region]["models"][m][str(h)]["alpha"]) for h in cfg["leads_days"]}
        assert out[m] == {h: float(rv[m][str(h)]["alpha"]) for h in cfg["leads_days"]}, (region, m)
    return out


def load_checked_lstm(region, man, n_in, seeds):
    models, hashes = {}, {}
    for s in seeds:
        name = f"b1_{region}_patch_H{H}_s{s}"
        sd = torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu")
        if weights_sha(sd) != man["lstm_patch_runs"][name]["weights_sha256"]:
            raise SystemExit(f"weights hash of {name} differs from the manifest")
        m = SSTLSTM(H, n_in=n_in)
        m.load_state_dict(sd)
        models[s], hashes[name] = m.eval(), man["lstm_patch_runs"][name]["weights_sha256"]
    return models, hashes


def ridge_fixed_alpha(sst_h, clim_h, cfg, halo, issue, h, alphas, train=None):
    """Ridge+doy / Ridge-patch predictions (anomalies, deg C, inner points) at the given issue dates with FIXED alphas.
    Fitted on training years only (rows: the existing Ridge rule); features read days <= issue only.
    alphas: {model: alpha}. Returns {model: (n_issue, nlat, nlon)}."""
    train = train or cfg["split"]["train"]
    sst_tr = sst_h.sel(time=slice(*train))
    full = pd.date_range(sst_h.time.min().values, sst_h.time.max().values)
    anom_full = anomaly(sst_h, clim_h).reindex(time=full)
    anom_tr = anomaly(sst_tr, clim_h).reindex(time=pd.date_range(sst_tr.time.min().values, sst_tr.time.max().values))
    gshape = (anom_full.sizes["latitude"], anom_full.sizes["longitude"])
    n_inner = (gshape[0] - 2 * halo, gshape[1] - 2 * halo)
    a_full, a_tr = anom_full.values.reshape(len(full), -1), anom_tr.values.reshape(anom_tr.sizes["time"], -1)
    i_tr, t_tr, _ = _pairs(sst_tr.time.values, train[0], train[1], h)
    assert pd.Timestamp(t_tr.max()) <= pd.Timestamp(train[1]), "training target after the training period"
    P = n_inner[0] * n_inner[1]
    y_tr = inner_block(anom_tr.sel(time=t_tr).values.reshape(len(t_tr), *gshape), halo).reshape(len(t_tr), P)
    Ftr = make_features(a_tr, anom_tr.time.values[0], i_tr).reshape(len(i_tr), *gshape, -1)
    Fte = make_features(a_full, anom_full.time.values[0], issue).reshape(len(issue), *gshape, -1)
    dtr, dte = day_features(i_tr).astype(float), day_features(issue).astype(float)
    out = {}
    for m, radius in NEW_RIDGES:
        pred = fit_predict(Ftr, y_tr, Fte, dtr, dte, radius, halo, n_inner, np.array([alphas[m]]), doy=True)
        out[m] = pred[0].reshape(len(issue), *n_inner)
    return out


def predict_lstm(model, data, flat, k, chunk=32768):
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), chunk):
            x = gather_inputs_b1(data, flat[i:i + chunk])
            assert not np.isnan(x).any(), "incomplete history at an issue date"
            out.append(model(torch.from_numpy(x))[:, k].numpy())
    return np.concatenate(out)


def evaluate_region(region, cfg, b1, sst_h, sst_i, models, alphas_new, test_all, period="test"):
    """`sst_h` (48x48) and `sst_i` (40x40) are cut at the end of cfg['split'][period]. `test_all`: frozen results/test_all.json
    (recomputed AP and Ridge must equal it); None skips that check (dry run only)."""
    halo, size = b1["halo"]["width_points"], b1["patch"]["size"]
    leads = list(cfg["leads_days"])
    alphas_pt = {int(h): float(a) for h, a in cfg["ridge"]["alphas"][region].items()}
    assert np.array_equal(inner_block(sst_h.values, halo), sst_i.values, equal_nan=True), "inner block differs from the inner file"
    data = make_data_b1(sst_h, cfg, halo, size, period=period)
    clim_i = build_climatology(sst_i, cfg)
    assert np.array_equal(inner_block(data.clim.values, halo), clim_i.values), "halo climatology differs on the inner block"
    P = data.n_points
    t0, t1 = cfg["split"][period]
    out = {"reference": {}, "lstm_patch": {"per_seed": {str(s): {} for s in models}}, "pairs": {}}
    for k, h in enumerate(leads):
        issue, target, counts = valid_pairs(sst_i.time.values, t0, t1, h)
        f_r, (i_r, t_r), _ = ridge_lead(sst_i, clim_i, cfg, h, period=period, alpha=alphas_pt[h])
        assert (i_r == issue).all() and (t_r == target).all(), "Ridge pairs must equal the baseline pairs"
        obs, ct, seas = sst_i.sel(time=target), clim_at(clim_i, target), season_labels(target)
        ref = {"anomaly_persistence": score_block(anomaly_persistence(sst_i, clim_i, issue, target), obs, ct, seas),
               "ridge": score_block(f_r, obs, ct, seas)}
        if test_all is not None:                                   # must reproduce the frozen test numbers
            for m in ref:
                frozen = test_all["regions"][region][str(h)]["methods"][m]["overall"]
                for key in ("rmse", "acc"):
                    assert abs(frozen[key] - ref[m]["overall"][key]) < 1e-9, (region, h, m, key)
        pred = ridge_fixed_alpha(sst_h, data.clim, cfg, halo, issue, h, {m: alphas_new[m][h] for m, _ in NEW_RIDGES})
        for m, _ in NEW_RIDGES:
            ref[m] = score_block(as_da(ct.values + pred[m], sst_i, target), obs, ct, seas)
        for m in ("ridge", "ridge_doy", "ridge_patch"):
            ref[m]["skill_vs_anomaly_persistence"] = add_skill(ref[m], ref["anomaly_persistence"])
        for m in ("ridge_doy", "ridge_patch"):
            ref[m]["skill_vs_ridge"] = add_skill(ref[m], ref["ridge"])
        ref["ridge_doy"]["skill_vs_ridge_patch"] = add_skill(ref["ridge_doy"], ref["ridge_patch"])
        out["reference"][str(h)] = ref
        out["pairs"][str(h)] = {"n_pairs": int(len(target)), **{a: b for a, b in counts.items() if a != "lead_days"}}
        t_idx = np.searchsorted(data.dates.values, issue.values)
        assert (data.dates[t_idx] == issue).all()
        flat = (t_idx[:, None] * P + np.arange(P)[None, :]).ravel()
        for s, model in models.items():
            p = predict_lstm(model, data, flat, k).reshape(len(issue), *data.n_inner)
            blk = score_block(as_da(ct.values + p * data.std, sst_i, target), obs, ct, seas)
            blk["skill_vs_anomaly_persistence"] = add_skill(blk, ref["anomaly_persistence"])
            blk["skill_vs_ridge"] = add_skill(blk, ref["ridge"])
            blk["skill_vs_ridge_patch"] = add_skill(blk, ref["ridge_patch"])
            out["lstm_patch"]["per_seed"][str(s)][str(h)] = blk
        print(f"[{region} lead={h}] rmse AP {ref['anomaly_persistence']['overall']['rmse']:.4f}  Ridge {ref['ridge']['overall']['rmse']:.4f}  "
              f"Ridge+doy {ref['ridge_doy']['overall']['rmse']:.4f}  Ridge-patch {ref['ridge_patch']['overall']['rmse']:.4f}  LSTM-patch seeds "
              + " ".join(f"{out['lstm_patch']['per_seed'][str(s)][str(h)]['overall']['rmse']:.4f}" for s in models), flush=True)
    out["lstm_patch"].update(agg([out["lstm_patch"]["per_seed"][s] for s in out["lstm_patch"]["per_seed"]]))
    return out


def run(dry_out=None):
    cfg, b1 = load_config(), load_b1()
    if dry_out is None:
        check_guards()
    else:                                                   # dry run: the "test" split is the validation years; nothing later is loaded
        cfg = {**cfg, "split": {**cfg["split"], "test": cfg["split"]["val"]}}
    last_day = cfg["split"]["test"][1]
    man = load_manifest()
    verify_provenance(man)
    started = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    test_all = json.loads((ROOT / "results" / "test_all.json").read_text()) if dry_out is None else None
    res = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["test"]}, "leads_days": cfg["leads_days"],
           "protocol": "docs/b1_test_protocol.md",
           "run": {"commit": _git("rev-parse", "HEAD").stdout.strip(), "tag": TAG, "started_utc": started, "hidden": H,
                   "seeds": b1["lstm_patch"]["seeds"], "dry_run": dry_out is not None,
                   "manifest_commit": man["git_commit_at_manifest"]},
           "alphas": {}, "weights_sha256": {}, "regions": {}}
    for r in cfg["regions"]:
        sst_h = xr.open_dataset(ROOT / "data" / "processed" / f"{r}_halo.nc")["sst"].sel(time=slice(None, last_day)).load()   # test period loaded HERE (unless dry run)
        sst_i = xr.open_dataset(ROOT / "data" / "processed" / f"{r}.nc")["sst"].sel(time=slice(None, last_day)).load()
        alphas = validation_alphas(man, r, cfg)
        models, hashes = load_checked_lstm(r, man, b1["patch"]["size"] ** 2 + 2, b1["lstm_patch"]["seeds"])
        res["alphas"][r] = {m: {str(h): a for h, a in d.items()} for m, d in alphas.items()}
        res["weights_sha256"].update(hashes)
        res["regions"][r] = evaluate_region(r, cfg, b1, sst_h, sst_i, models, alphas, test_all, period="test")
    res["run"]["wall_seconds_not_a_result"] = time.time() - t0
    path = OUT if dry_out is None else dry_out
    path.write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", metavar="OUT.json", help="validation data only; output must be outside results/b1_test.json")
    a = p.parse_args()
    if a.dry_run:
        from pathlib import Path
        out = Path(a.dry_run).resolve()
        if out == OUT.resolve():
            raise SystemExit("the dry run may not write results/b1_test.json")
        run(out)
    else:
        run()


if __name__ == "__main__":
    main()
