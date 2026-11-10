"""TEST period (2020-2025) for B2 (whole-region models): the single, final evaluation -> results/b2_test.json.

Protocol: docs/b2_test_protocol.md (git tag `pre-b2-test`). Safeguards: refuses to run unless the tag exists, HEAD is the tagged commit, the working
tree is clean, results/b2_test_choice.json equals the choice re-derived from results/b2_val_res.json, and results/b2_test.json does not exist.
Checks: weights_sha256 of the 12 B2 checkpoints and of the 6 LSTM-patch checkpoints vs the manifests; data-file hashes; the checkpoints reproduce the
recorded best validation loss; anomaly persistence / Ridge == results/test_all.json and Ridge-patch / LSTM-patch == results/b1_test.json (1e-9).
Models: existing checkpoints only, the chosen variant per region and family, seeds 0,1,2. Pairs = the frozen baselines' pairs.

  python -m sst.b2_test                       # THE test run (guarded, once)
  python -m sst.b2_test --dry-run OUT.json    # validation data only (test split := validation years, data cut at 2019-12-31); output outside the repo
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from sst import b1_test as B1
from sst.b1_data import load_b1
from sst.b1_ridge import as_da
from sst.b2_data import make_data_b2
from sst.b2_models import build
from sst.b2_test_choice import OUT as CHOICE, derive
from sst.b2_train import MODELS, RUNS, file_sha, predict_b2, val_loss, weights_sha
from sst.climatology import build_climatology, clim_at
from sst.download import ROOT, load_config
from sst.lstm_eval import add_skill, agg, score_block
from sst.metrics import season_labels, valid_pairs
from sst.run_val import _clean

TAG = "pre-b2-test"
OUT = ROOT / "results" / "b2_test.json"
SEEDS = (0, 1, 2)


def check_guards():
    if B1._git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode != 0:
        raise SystemExit(f"tag {TAG} missing")
    if B1._git("rev-parse", "HEAD").stdout != B1._git("rev-parse", f"{TAG}^{{commit}}").stdout:
        raise SystemExit("HEAD is not the tagged commit: refusing to run")
    if B1._git("status", "--porcelain").stdout.strip():
        raise SystemExit("working tree is dirty: refusing to run")
    if OUT.exists():
        raise SystemExit("results/b2_test.json exists: the test period is evaluated exactly once")


def load_choice():
    """The committed choice file must equal the choice re-derived from validation results."""
    if not CHOICE.exists():
        raise SystemExit("results/b2_test_choice.json missing")
    ch = json.loads(CHOICE.read_text())
    if ch != derive():
        raise SystemExit("results/b2_test_choice.json differs from the choice derived from results/b2_val_res.json")
    return ch


def load_manifests():
    runs = {}
    for f in ("b2_manifest.json", "b2_manifest_res.json"):
        runs.update(json.loads((ROOT / "results" / f).read_text())["runs"])
    return runs


def load_checked(name, runs):
    """Checkpoint state dict with the weights hash (and data hashes) verified against the manifest. Returns (state_dict, changed code files)."""
    r = runs[name]
    sd = torch.load(MODELS / f"{name}.pt", map_location="cpu")
    if weights_sha(sd) != r["weights_sha256"]:
        raise SystemExit(f"weights hash of {name} differs from the manifest")
    bad = {f for f, h in r["data_sha256"].items() if file_sha(f) != h}
    if bad:
        raise SystemExit(f"data files differ from the manifest for {name}: {sorted(bad)}")
    code_diff = sorted(f for f, h in r["code_sha256"].items() if file_sha(f) != h)       # listed, not a failure (protocol s.4)
    return sd, code_diff


def build_net(sd, model, C):
    net = build(model, C)
    net.load_state_dict(sd)
    return net.eval()


def evaluate_region(region, cfg, b1, sst_h, sst_i, choice, runs, frozen_test_all, frozen_b1, period="test"):
    """sst_h / sst_i cut at the end of cfg['split'][period]. frozen_*: None in the dry run (no frozen test numbers to compare with)."""
    halo, size = b1["halo"]["width_points"], b1["patch"]["size"]
    leads = list(cfg["leads_days"])
    t0, t1 = cfg["split"][period]
    # --- references: B1 code path (AP and Ridge vs test_all.json; Ridge-patch and LSTM-patch vs b1_test.json) ---
    man1 = B1.load_manifest()
    alphas = B1.validation_alphas(man1, region, cfg)
    lstm, hashes1 = B1.load_checked_lstm(region, man1, size ** 2 + 2, b1["lstm_patch"]["seeds"])
    ref = B1.evaluate_region(region, cfg, b1, sst_h, sst_i, lstm, alphas, frozen_test_all, period=period)
    if frozen_b1 is not None:
        fb = frozen_b1["regions"][region]
        for h in leads:
            for m in ("ridge_patch", "ridge_doy"):
                for k in ("rmse", "acc"):
                    assert abs(fb["reference"][str(h)][m]["overall"][k] - ref["reference"][str(h)][m]["overall"][k]) < 1e-9, (region, h, m, k)
            for s in b1["lstm_patch"]["seeds"]:
                for k in ("rmse", "acc"):
                    assert abs(fb["lstm_patch"]["per_seed"][str(s)][str(h)]["overall"][k] - ref["lstm_patch"]["per_seed"][str(s)][str(h)]["overall"][k]) < 1e-9, (region, h, s, k)
    # --- B2 data (test years loaded here, unless dry run) and validation-reproduction data ---
    data = make_data_b2(sst_h, cfg, period=period)
    dv = make_data_b2(sst_h.sel(time=slice(None, cfg["split"]["val"][1])), cfg, period="val")
    clim_i = build_climatology(sst_i, cfg)
    out = {"reference": ref["reference"], "lstm_patch": ref["lstm_patch"], "pairs": ref["pairs"], "models": {}, "code_files_changed_since_training": {}}
    pairs = {}
    for h in leads:
        issue, target, _ = valid_pairs(sst_i.time.values, t0, t1, h)
        t_idx = np.searchsorted(data.dates.values, issue.values)
        assert (data.dates[t_idx] == issue).all() and (t_idx >= 29).all(), "incomplete history at an issue date"
        pairs[h] = (issue, target, t_idx)
    union = np.unique(np.concatenate([p[2] for p in pairs.values()]))
    for fam, c in choice["regions"][region].items():
        model, C = c["chosen"], c["C"]
        per_seed = {}
        for s in SEEDS:
            name = f"b2_{model}_{region}_C{C}_s{s}"
            sd, code_diff = load_checked(name, runs)
            out["code_files_changed_since_training"][name] = code_diff
            net = build_net(sd, model, C)
            vl, rec = val_loss(net, dv), json.loads((RUNS / f"{name}.json").read_text())["best_val_loss"]
            assert abs(vl - rec) < 1e-4, f"{name}: checkpoint gives validation loss {vl} but the run recorded {rec}"
            pred = predict_b2(net, data, union)                                   # (n_union, 3, 40, 40), standardised
            assert np.isfinite(pred).all(), name
            res = {}
            for k, h in enumerate(leads):
                issue, target, t_idx = pairs[h]
                p = pred[np.searchsorted(union, t_idx)][:, k]
                obs, ct, seas = sst_i.sel(time=target), clim_at(clim_i, target), season_labels(target)
                blk = score_block(as_da(ct.values + p * data.std, sst_i, target), obs, ct, seas)
                R = ref["reference"][str(h)]
                blk["skill_vs_anomaly_persistence"] = add_skill(blk, R["anomaly_persistence"])
                blk["skill_vs_ridge"] = add_skill(blk, R["ridge"])
                blk["skill_vs_ridge_patch"] = add_skill(blk, R["ridge_patch"])
                blk["skill_vs_lstm_patch"] = add_skill(blk, ref["lstm_patch"]["mean"][str(h)])
                res[str(h)] = blk
            per_seed[str(s)] = res
            out_h = "  ".join(f"h={h}: {res[str(h)]['overall']['rmse']:.4f}" for h in leads)
            print(f"[{region} {model} C={C} seed={s}] {out_h}  (val-loss check {vl:.5f} vs {rec:.5f})", flush=True)
        out["models"][fam] = {"variant": model, "C": C, "weights_sha256": {f"b2_{model}_{region}_C{C}_s{s}": runs[f"b2_{model}_{region}_C{C}_s{s}"]["weights_sha256"] for s in SEEDS},
                              "per_seed": per_seed, **agg([per_seed[str(s)] for s in SEEDS])}
    out["lstm_patch_weights_sha256"] = hashes1
    return out


def run(dry_out=None):
    cfg, b1 = load_config(), load_b1()
    if dry_out is None:
        check_guards()
    else:
        cfg = {**cfg, "split": {**cfg["split"], "test": cfg["split"]["val"]}}
    choice = load_choice()
    last_day = cfg["split"]["test"][1]
    man1 = B1.load_manifest()
    B1.verify_provenance(man1)
    runs = load_manifests()
    started = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    frozen_all = json.loads((ROOT / "results" / "test_all.json").read_text()) if dry_out is None else None
    frozen_b1 = json.loads((ROOT / "results" / "b1_test.json").read_text()) if dry_out is None else None
    res = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["test"]}, "leads_days": cfg["leads_days"], "protocol": "docs/b2_test_protocol.md",
           "run": {"commit": B1._git("rev-parse", "HEAD").stdout.strip(), "tag": TAG, "started_utc": started, "seeds": list(SEEDS), "dry_run": dry_out is not None,
                   "torch_threads": torch.get_num_threads()},
           "choice": choice, "regions": {}}
    for r in cfg["regions"]:
        sst_h = xr.open_dataset(ROOT / "data" / "processed" / f"{r}_halo.nc")["sst"].sel(time=slice(None, last_day)).load()   # test period loaded HERE (unless dry run)
        sst_i = xr.open_dataset(ROOT / "data" / "processed" / f"{r}.nc")["sst"].sel(time=slice(None, last_day)).load()
        res["regions"][r] = evaluate_region(r, cfg, b1, sst_h, sst_i, choice, runs, frozen_all, frozen_b1, period="test")
    res["run"]["wall_seconds_not_a_result"] = time.time() - t0
    path = OUT if dry_out is None else dry_out
    path.write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", metavar="OUT.json", help="validation data only; output must be outside the repository")
    a = p.parse_args()
    if a.dry_run:
        from pathlib import Path
        out = Path(a.dry_run).resolve()
        if ROOT in out.parents or out == OUT.resolve():
            raise SystemExit("the dry run output must be outside the repository")
        run(out)
    else:
        run()


if __name__ == "__main__":
    main()
