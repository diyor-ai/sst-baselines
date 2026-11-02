"""TEST period (2020-2025): the single, final evaluation -> results/test_all.json (+ test_bias.json).

Safeguards: it refuses to run unless (1) the git tag `pre-test-freeze` exists, (2) config.yaml and src/ are
identical to that tag, (3) results/test_all.json does not exist yet (the test period is opened once), and
(4) the Ridge alphas in config.yaml equal the ones selected on validation (results/ridge_alphas.json).
Everything is fitted on TRAINING years only (climatology, r(h), Ridge); OCN uses only days before each issue date.
"""
import json
import subprocess

import pandas as pd
import xarray as xr

from sst.bias import bias_decomposition  # noqa: F401  (bias is computed inside run_region)
from sst.climatology import build_climatology
from sst.download import ROOT, load_config
from sst.run_val import _clean
from sst.run_val_all import METHODS, run_region

TAG = "pre-test-freeze"


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def check_frozen():
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode != 0:
        raise SystemExit(f"tag {TAG} missing: freeze first")
    if _git("diff", "--quiet", TAG, "--", "config.yaml", "src").returncode != 0 or _git(
            "status", "--porcelain", "--", "config.yaml", "src").stdout.strip():
        raise SystemExit("config.yaml / src differ from the freeze tag (committed or uncommitted): refusing to run")
    if (ROOT / "results" / "test_all.json").exists():
        raise SystemExit("results/test_all.json exists: the test period is evaluated exactly once")


def main():
    check_frozen()
    cfg = load_config()
    alphas = {r: {int(h): float(a) for h, a in d.items()} for r, d in cfg["ridge"]["alphas"].items()}
    sel = json.loads((ROOT / "results" / "ridge_alphas.json").read_text())["regions"]
    for r in alphas:
        assert alphas[r] == {int(h): i["alpha"] for h, i in sel[r].items()}, "config alphas != validation-selected alphas"
    commit = _git("rev-parse", "HEAD").stdout.strip()
    started = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    t0, t1 = cfg["split"]["test"]
    out = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["test"]},
           "run": {"commit": commit, "freeze_tag": TAG, "started_utc": started},
           "climatology": {"years": "train", "smoothing": cfg["climatology"]["smoothing"], "feb29": "use_feb28"},
           "leads_days": cfg["leads_days"], "methods": METHODS, "ridge_alphas_used": alphas,
           "definitions": "same as val_all.json; by_season_year groups are season|calendar year of the target date",
           "regions": {}}
    bias = {"period": out["period"], "regions": {}}
    for region in cfg["regions"]:
        sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"].load()
        assert sst.time.min() <= pd.Timestamp(cfg["split"]["train"][0]) and sst.time.max() >= pd.Timestamp(t1)
        clim = build_climatology(sst, cfg)   # uses training years only (checked in tests)
        res, b, info = run_region(sst, cfg, clim, region, period="test", alphas=alphas[region], season_year=True)
        out["regions"][region], bias["regions"][region] = res, b
        out.setdefault("ridge_fit", {})[region] = info
    for name, obj in (("test_all", out), ("test_bias", bias)):
        (ROOT / "results" / f"{name}.json").write_text(json.dumps(_clean(obj), indent=1, allow_nan=False))
        print(f"wrote results/{name}.json")


if __name__ == "__main__":
    main()
