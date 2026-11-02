"""Merge yearly files per region and run quality checks (report only, no filling)."""
import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

from sst.download import RAW, ROOT, load_config, update_manifest

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def normalise(ds):
    """Calendar-date time axis (PSL 00:00 / ERDDAP 12:00 -> 00:00), standard names and units,
    and fill values (|x| >= 1e20, e.g. netCDF default fill 9.96921e36) turned into NaN."""
    ds = ds.rename({k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in ds.dims})
    ds = ds.assign_coords(time=pd.DatetimeIndex(ds["time"].values).normalize())
    if "zlev" in ds.dims:
        ds = ds.squeeze("zlev", drop=True)
    ds["sst"] = ds["sst"].where(np.abs(ds["sst"]) < 1e20)
    ds["sst"].attrs["units"] = "degree_C"  # PSL says 'degC', ERDDAP 'degree_C'
    return ds


def to_ranges(dates):
    """['2000-01-01','2000-01-02','2000-01-05'] -> ['2000-01-01..2000-01-02', '2000-01-05']"""
    out, run = [], []
    for d in pd.to_datetime(sorted(dates)):
        if run and (d - run[-1]).days == 1:
            run.append(d)
        else:
            if run:
                out.append(run)
            run = [d]
    if run:
        out.append(run)
    return [str(r[0].date()) if len(r) == 1 else f"{r[0].date()}..{r[-1].date()}" for r in out]


def merge_region(region):
    """ERDDAP files first, then validated PSL gap days. No interpolation; missing days stay missing."""
    parts = []
    for src, code in (("erddap", 0), ("psl", 1)):
        for f in sorted((RAW / src / region).glob("*.nc")):
            year = int(f.stem)
            ds = normalise(xr.open_dataset(f)[["sst"]]).load()
            ds = ds.sel(time=ds.time.dt.year == year)  # ERDDAP may spill into the neighbouring year
            parts.append(ds.assign(source=("time", np.full(ds.sizes["time"], code, dtype="int8"))))
    ds = xr.concat(parts, dim="time").sortby("time", ascending=True)
    ds = ds.sel(time=~ds.get_index("time").duplicated(keep="first"))  # ERDDAP wins on a clash
    ds["source"].attrs["flag_meanings"] = "0=ERDDAP(OISST v2.1 final) 1=PSL(validated gap fill)"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(PROCESSED / f"{region}.nc")
    t = ds["time"].values
    update_manifest(f"_day_sources/{region}", {
        "erddap_days": int((ds["source"] == 0).sum()), "psl_days": int((ds["source"] == 1).sum()),
        "psl_dates": to_ranges(t[ds["source"].values == 1]),
    })
    return ds


def qc_report(sst, start, end, source=None):
    """sst: DataArray (time, latitude, longitude). Returns a JSON-able dict."""
    # Daily stamps are at 12:00; compare calendar days only.
    days = pd.DatetimeIndex(sst["time"].values).normalize()
    expected = pd.date_range(start, end, freq="D")
    uniq = pd.DatetimeIndex(days.unique())
    missing = expected.difference(uniq)
    dup = days[days.duplicated()].unique()
    vals = sst.values
    nan_frac = np.isnan(vals).mean(axis=0)  # per grid point
    n_land = int((nan_frac == 1).sum())
    n_partial = int(((nan_frac > 0) & (nan_frac < 1)).sum())
    rep = {
        "first_date": str(days.min().date()),
        "last_date": str(days.max().date()),
        "days_expected": int(len(expected)),
        "days_present": int(len(uniq)),
        "missing_dates": [str(d.date()) for d in missing],
        "duplicate_dates": [str(d.date()) for d in dup],
        "grid_shape": [int(sst.sizes["latitude"]), int(sst.sizes["longitude"])],
        "n_allnan_points_land": n_land,
        "n_partial_nan_points": n_partial,
        "sst_min": float(np.nanmin(vals)),
        "sst_max": float(np.nanmax(vals)),
        "sst_mean": float(np.nanmean(vals)),
        "units": sst.attrs.get("units"),
        "n_values_outside_valid_range": int(((vals < -3.0) | (vals > 45.0)).sum()),
    }
    if source is not None:
        rep["days_from_erddap"] = int((source == 0).sum())
        rep["days_from_psl_gapfill"] = int((source == 1).sum())
        rep["psl_filled_dates"] = to_ranges(sst["time"].values[source == 1])
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", help="default: all regions")
    args = p.parse_args()
    cfg = load_config()
    RESULTS.mkdir(exist_ok=True)
    for region in [args.region] if args.region else cfg["regions"]:
        ds = merge_region(region)
        rep = qc_report(ds["sst"], cfg["period"]["start"], cfg["period"]["end"], ds["source"].values)
        (RESULTS / f"qc_{region}.json").write_text(json.dumps(rep, indent=2))
        print(f"== QC {region} ==")
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
