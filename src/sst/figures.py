"""Figure 1: training-period mean SST and preliminary-anomaly std per region."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sst.download import ROOT, load_config

RESULTS = ROOT / "results"


def _md_key(time):
    """month*100+day key; Feb 29 is mapped to Feb 28 (config: feb29 = use_feb28)."""
    t = time.dt
    key = t.month * 100 + t.day
    return key.where(key != 229, 228)


def train_doy_mean(sst, train):
    """Day-of-year mean computed from training years ONLY."""
    tr = sst.sel(time=slice(*train))
    return tr.groupby(_md_key(tr["time"])).mean("time")


def preliminary_anomaly(sst, train):
    """SST minus the training-only day-of-year mean (applied to all dates)."""
    clim = train_doy_mean(sst, train)
    key = _md_key(sst["time"])
    return sst.groupby(key) - clim


def lag1_autocorr(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / (x * x).sum())


def region_stats(anom_train):
    series = anom_train.mean(["latitude", "longitude"], skipna=True).values
    return {
        "region_mean_anomaly_std": float(np.std(series, ddof=1)),
        "region_mean_anomaly_lag1_autocorr": lag1_autocorr(series),
        "n_days": int(series.size),
    }


def main():
    cfg = load_config()
    train = cfg["split"]["train"]
    names = list(cfg["regions"])
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 5 * len(names)))
    stats = {"anomaly_definition": "SST minus training-years (%s to %s) day-of-year mean; preliminary" % tuple(train)}
    for i, region in enumerate(names):
        sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"]
        b = cfg["regions"][region]
        box = f"{b['lat_min']}–{b['lat_max']}°N, {b['lon_min']}–{b['lon_max']}°E"
        tr = sst.sel(time=slice(*train))
        anom_tr = preliminary_anomaly(sst, train).sel(time=slice(*train))
        panels = [
            (tr.mean("time"), "viridis", "Mean SST (°C)", "(a) Mean SST, training years"),
            (anom_tr.std("time"), "magma", "Std of anomaly (°C)",
             "(b) Std of daily anomaly\n(preliminary anomaly; final climatology on day 2)"),
        ]
        for ax, (field, cmap, label, title) in zip(axes[i], panels):
            m = ax.pcolormesh(field.longitude, field.latitude, field, cmap=cmap, shading="auto")
            fig.colorbar(m, ax=ax, label=label)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"{region}: {title}\n{box}", fontsize=9)
        stats[region] = region_stats(anom_tr)
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fig1_regions.png", dpi=150)
    (RESULTS / "region_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
