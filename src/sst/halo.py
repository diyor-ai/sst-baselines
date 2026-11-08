"""B1: 1-degree halo (4 grid points per side) around both boxes -> 48x48 grids, same hybrid ERDDAP + PSL pipeline.

Reuses the checks/functions of download.py, fill.py and data.py WITHOUT editing them. Everything is written to NEW files:
  data/raw/erddap/<region>_halo/, data/raw/psl/<region>_halo/, data/raw/manifest_halo.json,
  data/processed/<region>_halo.nc, results/qc_<region>_halo.json, results/halo_psl_validation.json, results/halo_checks.json.
The existing 40x40 files and data/raw/manifest.json are never written.

  python -m sst.halo download | fill | merge | check | all
"""
import calendar
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sst.download as dl
import sst.fill as fl
from sst.b1_data import load_b1
from sst.data import PROCESSED, normalise, qc_report, to_ranges
from sst.download import RAW, ROOT, VALID_RANGE, erddap_axis, load_config, sha256

HALO = 4  # grid points per side (1 degree)
MANIFEST = RAW / "manifest_halo.json"


def halo_cfg():
    """config.yaml with `regions` replaced by the halo boxes, so the existing download/fill helpers work unchanged."""
    cfg = load_config()
    b1 = load_b1()
    cfg = dict(cfg)
    cfg["regions"] = {k: {kk: v[kk] for kk in ("lat_min", "lat_max", "lon_min", "lon_max")}
                      for k, v in b1["halo"]["regions"].items()}
    return cfg, b1


def update_manifest_halo(key, entry):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m[key] = entry
    tmp = MANIFEST.with_suffix(".json.part")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.rename(MANIFEST)


# the shared helpers write their manifest entries through a module-level name; point it at the halo manifest
dl.update_manifest = update_manifest_halo
fl.update_manifest = update_manifest_halo


def do_download(cfg, axis):
    for r in cfg["regions"]:
        dl.download_region(cfg, r, axis)


def do_fill(cfg, axis):
    """Same logic as fill.main (double fetch of every gap month, range/step/land/ERDDAP-overlap checks, year gate)
    for the halo boxes; output paths are the halo ones."""
    expected = pd.date_range(cfg["period"]["start"], cfg["period"]["end"])
    gaps = expected.difference(axis)
    print(f"{len(gaps)} gap days; {len(set(zip(gaps.year, gaps.month)))} (year, month) pieces", flush=True)
    u = fl.union_box(cfg)
    erd = {r: fl.load_erddap_days(r) for r in cfg["regions"]}
    land = {r: np.isnan(erd[r][sorted(erd[r])[0]]) for r in cfg["regions"]}
    maxstep, thr = {}, 0.0
    for r in cfg["regions"]:
        m, t = fl.step_threshold(erd[r])
        maxstep[r] = m
        thr = max(thr, t)
    print(f"largest day-to-day change in ERDDAP halo data: {maxstep}; plausibility threshold {thr:.2f} degC", flush=True)
    report = {"tolerance": fl.TOL, "step_threshold_degC": thr, "erddap_max_step": maxstep, "pieces": [], "years": {}}
    kept = {r: {} for r in cfg["regions"]}
    for year, month in sorted(set(zip(gaps.year, gaps.month))):
        last = calendar.monthrange(year, month)[1]
        t0, t1 = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"
        a, attrs, url = fl.psl_slice(cfg, year, t0, t1, u)   # fetched twice, one after the other (one worker)
        b, _, _ = fl.psl_slice(cfg, year, t0, t1, u)
        for r, box in cfg["regions"].items():
            rep = fl.validate_month(fl.cut(a, box), fl.cut(b, box), box, land[r], erd[r], thr)
            gd = [d for d in gaps if d.year == year and d.month == month]
            rep.update(region=r, year=int(year), month=int(month), n_gap_days=len(gd), gap_dates=[str(d.date()) for d in gd])
            report["pieces"].append(rep)
            kept[r].setdefault(year, {"attrs": attrs, "url": url, "days": {}})
            if rep["month_ok"]:
                ca = fl.cut(a, box)
                for d in gd:
                    kept[r][year]["days"][d] = ca.sel(time=d, method="nearest").values
            print(f"[{r} {year}-{month:02d}] gap days {len(gd)}, overlap {rep['n_overlap_days']}, "
                  f"max|d| {rep['max_abs_diff_vs_erddap']}, ok={rep['month_ok']}", flush=True)
    for r in cfg["regions"]:
        for year in sorted(kept[r]):
            ps = [p for p in report["pieces"] if p["region"] == r and p["year"] == year]
            n_over = sum(p["n_overlap_days"] for p in ps)
            year_ok = all(p["month_ok"] for p in ps) and n_over > 0
            max_d = max([p["max_abs_diff_vs_erddap"] for p in ps if p["max_abs_diff_vs_erddap"] is not None], default=None)
            all_gaps = [d for p in ps for d in p["gap_dates"]]
            accepted = sorted(kept[r][year]["days"]) if year_ok else []
            report["years"][f"{r}/{year}"] = {
                "overlap_days": n_over, "max_abs_diff_vs_erddap": max_d,
                "cells_diff_gt_tol": sum(p["n_cells_diff_gt_tol"] for p in ps), "year_ok": year_ok,
                "gap_days": len(all_gaps), "accepted_days": [str(d.date()) for d in accepted],
                "left_missing": sorted(set(all_gaps) - {str(d.date()) for d in accepted})}
            if accepted:
                box = cfg["regions"][r]
                lat, lon = fl.cut(a, box).lat.values, fl.cut(a, box).lon.values
                da = xr.DataArray(np.stack([kept[r][year]["days"][d] for d in accepted]), dims=("time", "lat", "lon"),
                                  coords={"time": list(accepted), "lat": lat, "lon": lon}, attrs={"units": "degC"}, name="sst")
                out = RAW / "psl" / r / f"{year}.nc"
                out.parent.mkdir(parents=True, exist_ok=True)
                da.to_dataset().to_netcdf(out)
                update_manifest_halo(f"psl/{r}/{year}.nc", {
                    "source": fl.PSL_SOURCE, "url": kept[r][year]["url"], "bytes": out.stat().st_size, "sha256": sha256(out),
                    "path": str(out.relative_to(ROOT)), "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "days": [str(d.date()) for d in accepted],
                    "validation": f"double fetch identical, range, steps<= {thr:.2f}, matches ERDDAP on {n_over} overlap days (max|d|={max_d})"})
    (ROOT / "results" / "halo_psl_validation.json").write_text(json.dumps(report, indent=2, default=str))
    ny = sum(1 for v in report["years"].values() if v["year_ok"])
    print(f"halo years accepted: {ny}/{len(report['years'])}; see results/halo_psl_validation.json", flush=True)


def merge_halo(region):
    """Same as data.merge_region (ERDDAP first, validated PSL gap days after, ERDDAP wins on a clash) -> <region>.nc."""
    parts = []
    for src, code in (("erddap", 0), ("psl", 1)):
        for f in sorted((RAW / src / region).glob("*.nc")):
            year = int(f.stem)
            ds = normalise(xr.open_dataset(f)[["sst"]]).load()
            ds = ds.sel(time=ds.time.dt.year == year)
            parts.append(ds.assign(source=("time", np.full(ds.sizes["time"], code, dtype="int8"))))
    ds = xr.concat(parts, dim="time").sortby("time", ascending=True)
    ds = ds.sel(time=~ds.get_index("time").duplicated(keep="first"))
    ds["source"].attrs["flag_meanings"] = "0=ERDDAP(OISST v2.1 final) 1=PSL(validated gap fill)"
    ds.to_netcdf(PROCESSED / f"{region}.nc")
    update_manifest_halo(f"_day_sources/{region}", {
        "erddap_days": int((ds["source"] == 0).sum()), "psl_days": int((ds["source"] == 1).sum()),
        "psl_dates": to_ranges(ds["time"].values[ds["source"].values == 1])})
    return ds


def do_merge_and_check(cfg):
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    checks = {}
    for region, box in cfg["regions"].items():
        inner = load_b1()["halo"]["regions"][region]["inner"]
        ds = merge_halo(region)
        rep = qc_report(ds["sst"], start, end, ds["source"].values)
        (ROOT / "results" / f"qc_{region}.json").write_text(json.dumps(rep, indent=2))
        old = xr.open_dataset(PROCESSED / f"{inner}.nc")
        assert rep["days_present"] == 16071 and rep["days_expected"] == 16071 and not rep["missing_dates"], rep["missing_dates"][:5]
        assert rep["grid_shape"] == [48, 48] and rep["n_values_outside_valid_range"] == 0
        sl = dict(latitude=slice(HALO, HALO + 40), longitude=slice(HALO, HALO + 40))
        new_in = ds.isel(**sl)
        c = {"time_equal": bool((ds.time.values == old.time.values).all()),
             "lat_equal": bool(np.array_equal(new_in.latitude.values, old.latitude.values)),
             "lon_equal": bool(np.array_equal(new_in.longitude.values, old.longitude.values)),
             "sst_equal_bitwise_incl_nan": bool(np.array_equal(new_in["sst"].values, old["sst"].values, equal_nan=True)),
             "source_equal": bool(np.array_equal(ds["source"].values, old["source"].values))}
        assert all(c.values()), (region, c)
        v = ds["sst"].values
        nanmask = np.isnan(v)
        halo_ring = np.ones((48, 48), bool)
        halo_ring[HALO:HALO + 40, HALO:HALO + 40] = False
        c.update(inner=inner, halo_points=int(halo_ring.sum()),
                 halo_points_with_any_nan=int((nanmask.any(axis=0) & halo_ring).sum()),
                 halo_points_all_nan_land=int((nanmask.all(axis=0) & halo_ring).sum()),
                 inner_points_with_any_nan=int((nanmask.any(axis=0) & ~halo_ring).sum()),
                 nan_values_total=int(nanmask.sum()), grid_shape=rep["grid_shape"], days=rep["days_present"],
                 days_from_erddap=rep["days_from_erddap"], days_from_psl_gapfill=rep["days_from_psl_gapfill"],
                 psl_filled_dates=rep["psl_filled_dates"],
                 file_sha256=sha256(PROCESSED / f"{region}.nc"), inner_file_sha256=sha256(PROCESSED / f"{inner}.nc"),
                 sst_min=rep["sst_min"], sst_max=rep["sst_max"])
        checks[region] = c
        print(f"== {region}: " + json.dumps({k: v for k, v in c.items() if k != "psl_filled_dates"}), flush=True)
    (ROOT / "results" / "halo_checks.json").write_text(json.dumps(checks, indent=2))
    print("wrote results/halo_checks.json")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg, _ = halo_cfg()
    axis = erddap_axis(cfg)
    if step in ("download", "all"):
        do_download(cfg, axis)
    if step in ("fill", "all"):
        do_fill(cfg, axis)
    if step in ("merge", "check", "all"):
        do_merge_and_check(cfg)


if __name__ == "__main__":
    main()
