"""Compare PSL files with the kept ERDDAP (v2.1) files on every overlapping day, per region and year."""
import json
import re
import sys

import numpy as np
import xarray as xr

from sst.data import normalise
from sst.download import RAW, ROOT

TOL = 1e-5  # float32 rounding of values ~30 is ~2e-6; anything above this is a real difference


def erddap_files():
    out = {}
    for d in [*RAW.glob("_erddap*"), RAW / "_incomplete"]:
        for f in d.rglob("*.nc"):
            out[(f.parent.name, int(re.match(r"\d{4}", f.name).group()))] = f
    return out


def main(psl_dir=RAW, out_name="crosscheck_erddap_psl.json"):
    rows = []
    for (region, year), ef in sorted(erddap_files().items()):
        pf = psl_dir / region / f"{year}.nc"
        if not pf.exists():
            continue
        p = normalise(xr.open_dataset(pf))["sst"]
        e = normalise(xr.open_dataset(ef))["sst"]
        common = p.time.to_index().intersection(e.time.to_index())
        a, b = p.sel(time=common).values, e.sel(time=common).values
        d = np.abs(a - b)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        rows.append({
            "region": region, "year": year, "erddap_file": str(ef.relative_to(ROOT)),
            "n_overlap_days": int(len(common)), "n_cells_compared": int(d.size),
            "max_abs_diff": float(np.nanmax(d)),
            "n_cells_diff_gt_tol": int((d > TOL).sum()), "n_nan_mismatch": nan_mismatch,
        })
        print(rows[-1]["region"], year, "days", len(common), "max|d|", f"{rows[-1]['max_abs_diff']:.2e}",
              "cells>tol", rows[-1]["n_cells_diff_gt_tol"], "nan_mismatch", nan_mismatch, flush=True)
    (ROOT / "results" / out_name).write_text(
        json.dumps({"tolerance": TOL, "rows": rows}, indent=2))


if __name__ == "__main__":
    main(RAW / sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else main()
