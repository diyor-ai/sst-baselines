"""Region stats with the FINAL training climatology (harmonics3, training years only).

Only data up to the training end is used. The earlier numbers (unsmoothed day-of-year mean) are kept in
results/region_stats.json under "preliminary"; the new ones go under "final". Safe to re-run.
"""
import json

import xarray as xr

from sst.climatology import anomaly, build_climatology
from sst.download import ROOT, load_config
from sst.figures import region_stats

PATH = ROOT / "results" / "region_stats.json"


def main():
    cfg = load_config()
    train = cfg["split"]["train"]
    old = json.loads(PATH.read_text())
    prelim = old["preliminary"] if "preliminary" in old else old
    final = {"anomaly_definition": "SST minus FINAL training climatology (%s, years %s to %s); cos(lat) weights"
             % (cfg["climatology"]["smoothing"], *train)}
    for region in cfg["regions"]:
        sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"].sel(time=slice(*train)).load()
        clim = build_climatology(sst, cfg)
        final[region] = region_stats(anomaly(sst, clim))
    PATH.write_text(json.dumps({"preliminary": prelim, "final": final}, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
