"""Fill days that ERDDAP lacks from NOAA PSL (OPeNDAP), only from validated pieces.

For every (year, month) containing gap days, the month is fetched TWICE (union box of both
regions, one request at a time) and must pass:
  1. both fetches identical;
  2. values inside the valid range, NaN only on land cells;
  3. day-to-day change below a threshold derived from the ERDDAP data;
  4. exact agreement (<= TOL) with ERDDAP on all overlapping days of that month.
Gap days are accepted only if the whole YEAR also matches ERDDAP on its overlap days.
Anything else stays missing and is listed in results/psl_validation.json.
"""
import calendar
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

from sst.data import normalise
from sst.download import (RAW, ROOT, VALID_RANGE, build_psl_url, erddap_axis, load_config, sha256,
                          update_manifest)

TOL = 1e-5          # float32 rounding of values ~30 is ~2e-6
STEP_MARGIN = 1.25  # plausibility threshold = margin * largest day-to-day change seen in ERDDAP
PSL_SOURCE = "NOAA PSL OPeNDAP (noaa.oisst.v2.highres)"
MAX_RETRIES = 5


def load_erddap_days(region):
    """{date: 2-D array} from every ERDDAP file of the region (spill-over days trimmed per file)."""
    days = {}
    for f in sorted((RAW / "erddap" / region).glob("*.nc")):
        year = int(f.stem)
        da = normalise(xr.open_dataset(f))["sst"]
        da = da.sel(time=da.time.dt.year == year).load()
        for t, arr in zip(da.time.values, da.values):
            days[pd.Timestamp(t)] = arr
    return days


def union_box(cfg):
    boxes = list(cfg["regions"].values())
    return {k: (min if k.endswith("min") else max)(b[k] for b in boxes) for k in boxes[0]}


def psl_slice(cfg, year, t0, t1, box):
    """One OPeNDAP request (retry with long backoff; the server answers 429 when pushed)."""
    url = build_psl_url(cfg, year)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with xr.open_dataset(url) as ds:
                sub = ds[cfg["dataset"]["variable"]].sel(
                    time=slice(t0, t1), lat=slice(box["lat_min"], box["lat_max"]),
                    lon=slice(box["lon_min"], box["lon_max"])).load()
                attrs = dict(ds.attrs)
            n = (pd.Timestamp(t1) - pd.Timestamp(t0)).days + 1
            if sub.sizes["time"] != n:
                raise ValueError(f"got {sub.sizes['time']} days, need {n}")
            time.sleep(1)  # be polite
            return sub, attrs, url
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 10 * 2 ** attempt
            print(f"  [{t0}] attempt {attempt}/{MAX_RETRIES} failed ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)


def cut(arr, box):
    return arr.sel(lat=slice(box["lat_min"], box["lat_max"]), lon=slice(box["lon_min"], box["lon_max"]))


def step_threshold(erddap):
    """Largest day-to-day change of any cell between consecutive ERDDAP days, times a margin."""
    ds = sorted(erddap)
    m = max(float(np.nanmax(np.abs(erddap[b] - erddap[a])))
            for a, b in zip(ds[:-1], ds[1:]) if (b - a).days == 1)
    return m, STEP_MARGIN * m


def validate_month(a, b, box, land, erd, thr):
    """a, b: two fetches (time, lat, lon) of the same month, already cut to the region."""
    rep = {"double_fetch_identical": bool(np.array_equal(a.values, b.values, equal_nan=True))}
    v = a.values
    nan = np.isnan(v)
    rep["nan_only_on_land"] = bool((nan[:, ~land]).sum() == 0)
    ok = v[~nan]
    rep["range_ok"] = bool(ok.size > 0 and ok.min() >= VALID_RANGE[0] and ok.max() <= VALID_RANGE[1])
    steps = np.nanmax(np.abs(np.diff(v, axis=0)), axis=(1, 2)) if len(v) > 1 else np.array([0.0])
    rep["max_day_to_day_change"] = float(steps.max())
    rep["plausible_steps"] = bool(steps.max() <= thr)
    over = [(i, pd.Timestamp(t)) for i, t in enumerate(a.time.values) if pd.Timestamp(t).normalize() in erd]
    rep["n_overlap_days"] = len(over)
    d = [np.abs(v[i] - erd[t.normalize()]) for i, t in over]
    rep["max_abs_diff_vs_erddap"] = float(np.nanmax(d)) if d else None
    rep["n_cells_diff_gt_tol"] = int(sum((x > TOL).sum() for x in d))
    rep["n_cells_compared"] = int(sum(x.size for x in d))
    rep["month_ok"] = (rep["double_fetch_identical"] and rep["nan_only_on_land"] and rep["range_ok"]
                       and rep["plausible_steps"] and rep["n_cells_diff_gt_tol"] == 0)
    return rep


def main():
    cfg = load_config()
    axis = erddap_axis(cfg)
    expected = pd.date_range(cfg["period"]["start"], cfg["period"]["end"])
    gaps = expected.difference(axis)
    print(f"{len(gaps)} gap days in the period; {len(set(zip(gaps.year, gaps.month)))} (year, month) pieces", flush=True)
    u = union_box(cfg)
    erd = {r: load_erddap_days(r) for r in cfg["regions"]}
    land = {}
    for r in cfg["regions"]:
        first = sorted(erd[r])[0]
        land[r] = np.isnan(erd[r][first])
    maxstep, thr = {}, 0.0
    for r in cfg["regions"]:
        m, t = step_threshold(erd[r]); maxstep[r] = m; thr = max(thr, t)
    print(f"largest day-to-day change in ERDDAP data: {maxstep}; plausibility threshold {thr:.2f} degC", flush=True)

    months = sorted(set(zip(gaps.year, gaps.month)))
    report = {"tolerance": TOL, "step_threshold_degC": thr, "erddap_max_step": maxstep, "pieces": [], "years": {}}
    kept = {r: {} for r in cfg["regions"]}  # region -> year -> {date: array}
    for year, month in months:
        last = calendar.monthrange(year, month)[1]
        t0, t1 = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"
        a, attrs, url = psl_slice(cfg, year, t0, t1, u)
        b, _, _ = psl_slice(cfg, year, t0, t1, u)
        for r, box in cfg["regions"].items():
            rep = validate_month(cut(a, box), cut(b, box), box, land[r], erd[r], thr)
            gd = [d for d in gaps if d.year == year and d.month == month]
            rep.update(region=r, year=int(year), month=int(month), n_gap_days=len(gd),
                       gap_dates=[str(d.date()) for d in gd])
            report["pieces"].append(rep)
            kept[r].setdefault(year, {"attrs": attrs, "url": url, "days": {}})
            if rep["month_ok"]:
                ca = cut(a, box)
                for d in gd:
                    kept[r][year]["days"][d] = ca.sel(time=d, method="nearest").values
            print(f"[{r} {year}-{month:02d}] gap days {len(gd)}, overlap {rep['n_overlap_days']}, "
                  f"max|d| {rep['max_abs_diff_vs_erddap']}, ok={rep['month_ok']}", flush=True)

    # year-level gate: every piece of the year must match ERDDAP on its overlap days
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
                "left_missing": sorted(set(all_gaps) - {str(d.date()) for d in accepted}),
            }
            if accepted:
                box = cfg["regions"][r]
                lat = cut(a, box).lat.values; lon = cut(a, box).lon.values
                da = xr.DataArray(np.stack([kept[r][year]["days"][d] for d in accepted]),
                                  dims=("time", "lat", "lon"),
                                  coords={"time": [d for d in accepted], "lat": lat, "lon": lon},
                                  attrs={"units": "degC"}, name="sst")
                out = RAW / "psl" / r / f"{year}.nc"
                out.parent.mkdir(parents=True, exist_ok=True)
                da.to_dataset().to_netcdf(out)
                update_manifest(f"psl/{r}/{year}.nc", {
                    "source": PSL_SOURCE, "url": kept[r][year]["url"], "bytes": out.stat().st_size,
                    "sha256": sha256(out), "path": str(out.relative_to(ROOT)),
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "days": [str(d.date()) for d in accepted],
                    "psl_title": kept[r][year]["attrs"].get("title"),
                    "psl_version": kept[r][year]["attrs"].get("version"),
                    "psl_history": kept[r][year]["attrs"].get("history"),
                    "validation": f"double fetch identical, range, steps<= {thr:.2f}, "
                                  f"matches ERDDAP on {n_over} overlap days (max|d|={max_d})"})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "psl_validation.json").write_text(json.dumps(report, indent=2, default=str))
    ny = sum(1 for v in report["years"].values() if v["year_ok"])
    print(f"years accepted: {ny}/{len(report['years'])}; see results/psl_validation.json", flush=True)


def samples():
    """v2 vs v2.1 label check: single PSL days (1999-2024) against ERDDAP, both regions."""
    cfg = load_config()
    u = union_box(cfg)
    erd = {r: load_erddap_days(r) for r in cfg["regions"]}
    rows = []
    for year in (1999, 2005, 2010, 2015, 2016, 2020, 2024):
        for md in ("01-15", "04-15", "07-15", "10-15"):
            day = pd.Timestamp(f"{year}-{md}")
            a, attrs, _ = psl_slice(cfg, year, str(day.date()), str(day.date()), u)
            for r, box in cfg["regions"].items():
                if day not in erd[r]:
                    continue
                d = np.abs(cut(a, box).values[0] - erd[r][day])
                rows.append({"region": r, "date": str(day.date()), "psl_label": (attrs.get("version") or "v2 (AVHRR only)"),
                             "max_abs_diff": float(d.max()), "n_cells_diff_gt_tol": int((d > TOL).sum())})
    (ROOT / "results" / "psl_version_samples.json").write_text(json.dumps({"tolerance": TOL, "rows": rows}, indent=2))
    by_year = {}
    for x in rows:
        y = x["date"][:4]; by_year.setdefault(y, []).append(x["max_abs_diff"])
    for y, v in by_year.items():
        print(y, "days", len(v), "max|d|", f"{max(v):.2e}")


if __name__ == "__main__":
    samples() if "--samples" in sys.argv else main()
