"""Download OISST `sst` from ERDDAP (primary source), one request per region per year.

Days that ERDDAP lacks are filled from NOAA PSL by `sst.fill` (validated pieces only).
"""
import argparse
import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
# netCDF-C reads HTTP timeouts for OPeNDAP from this rc file (must be set before the first open)
os.environ.setdefault("DAPRCFILE", str(ROOT / ".dodsrc"))
ERDDAP_SOURCE = "NOAA ERDDAP ncdcOisst21Agg (OISST v2.1 final)"
VALID_RANGE = (-3.0, 45.0)  # valid_range of the `sst` variable (degree_C)
MAX_RETRIES = 5


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def build_url(cfg, region, year):
    """ERDDAP griddap query for one region and one calendar year (zlev=0)."""
    b = cfg["regions"][region]
    var = cfg["dataset"]["variable"]
    return (
        f"{cfg['dataset']['erddap_base']}.nc?{var}"
        f"[({year}-01-01T12:00:00Z):1:({year}-12-31T12:00:00Z)][(0.0):1:(0.0)]"
        f"[({b['lat_min']}):1:({b['lat_max']})]"
        f"[({b['lon_min']}):1:({b['lon_max']})]"
    )


def build_psl_url(cfg, year):
    d = cfg["dataset"]
    return f"{d['psl_base']}/{d['psl_file_pattern'].format(year=year)}"


def expected_shape(box):
    """(n_lat, n_lon) of a box whose bounds are 0.25-degree grid-cell centres."""
    return (round((box["lat_max"] - box["lat_min"]) / 0.25) + 1,
            round((box["lon_max"] - box["lon_min"]) / 0.25) + 1)


def erddap_axis(cfg):
    """All dates (midnight) on the server's time axis; cached in data/raw/erddap_time_axis.csv."""
    cache = RAW / "erddap_time_axis.csv"
    if not cache.exists():
        r = requests.get(cfg["dataset"]["erddap_base"] + ".csv?time", timeout=(30, 300))
        r.raise_for_status()
        t = pd.to_datetime(pd.read_csv(io.StringIO(r.text), skiprows=[1])["time"], utc=True)
        pd.Series(pd.DatetimeIndex(t).tz_localize(None).normalize()).to_csv(cache, index=False, header=["date"])
    return pd.DatetimeIndex(pd.read_csv(cache)["date"])


def check_year(ds, year, box, axis_days):
    """Raise unless `ds` holds exactly the days the server has for `year`, on the expected grid.

    ERDDAP snaps requested bounds to the nearest existing day, so a file can spill into the
    neighbouring year; those days are ignored here (and trimmed again when merging).
    """
    days = pd.DatetimeIndex(ds["time"].values).normalize()
    days = days[days.year == year]
    expected = axis_days[axis_days.year == year]
    if not days.equals(expected):
        raise ValueError(f"{year}: {len(days)} days present, server has {len(expected)}")
    shape = (ds.sizes["latitude"], ds.sizes["longitude"])
    if shape != expected_shape(box):
        raise ValueError(f"{year}: grid {shape}, expected {expected_shape(box)}")
    v = ds["sst"].values
    ok = v[~pd.isna(v)]
    if ok.size == 0 or ok.min() < VALID_RANGE[0] or ok.max() > VALID_RANGE[1]:
        raise ValueError(f"{year}: values outside valid range {VALID_RANGE}")


def is_valid_nc(path, year, box, axis_days):
    try:
        with xr.open_dataset(path) as ds:
            check_year(ds, year, box, axis_days)
        return True
    except Exception:
        return False


def fetch(url, dest, year, box, axis_days):
    """Download to dest via a .part file; retry with exponential backoff (max MAX_RETRIES)."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=(30, 600))
            r.raise_for_status()
            part.write_bytes(r.content)
            if not is_valid_nc(part, year, box, axis_days):
                raise ValueError("downloaded file failed validation (days/grid/range)")
            part.rename(dest)
            return
        except Exception as e:
            part.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                raise
            wait = 5 * 2 ** attempt
            print(f"  attempt {attempt}/{MAX_RETRIES} failed ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)


def update_manifest(key, entry):
    mpath = RAW / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[key] = entry
    tmp = mpath.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.rename(mpath)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adopt_old_manifest(cfg, regions):
    """Re-register files downloaded before the manifest was reorganised (sha256 must still match)."""
    old = RAW / "manifest_erddap_attempt.json"
    if not old.exists():
        return
    entries = json.loads(old.read_text())
    mpath = RAW / "manifest.json"
    current = json.loads(mpath.read_text()) if mpath.exists() else {}
    for key, e in entries.items():
        if key.startswith("_") or not isinstance(e, dict) or key in current or "sha256" not in e:
            continue
        f = RAW / "erddap" / key
        if f.exists() and sha256(f) == e["sha256"]:
            update_manifest(key, {"source": ERDDAP_SOURCE, **e, "path": str(f.relative_to(ROOT))})


def download_region(cfg, region, axis_days):
    y0, y1 = int(cfg["period"]["start"][:4]), int(cfg["period"]["end"][:4])
    box = cfg["regions"][region]
    outdir = RAW / "erddap" / region
    outdir.mkdir(parents=True, exist_ok=True)
    for year in range(y0, y1 + 1):
        dest = outdir / f"{year}.nc"
        tag = f"[{region} {year}]"
        if dest.exists() and is_valid_nc(dest, year, box, axis_days):
            print(f"{tag} exists and matches the server's days, skipping", flush=True)
            continue
        url = build_url(cfg, region, year)
        t0 = time.time()
        fetch(url, dest, year, box, axis_days)
        update_manifest(f"{region}/{year}.nc", {
            "source": ERDDAP_SOURCE, "url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "path": str(dest.relative_to(ROOT)),
        })
        print(f"{tag} done: {dest.stat().st_size/1e3:.0f} KB in {time.time()-t0:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--region")
    g.add_argument("--all", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    regions = list(cfg["regions"]) if args.all else [args.region]
    axis = erddap_axis(cfg)
    adopt_old_manifest(cfg, regions)
    for r in regions:
        download_region(cfg, r, axis)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
