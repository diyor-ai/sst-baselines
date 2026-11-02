"""Figure 1: training-period mean SST and preliminary-anomaly std per region."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sst.climatology import anomaly, build_climatology
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
    """Training-period statistics of the (preliminary) anomaly, cos(lat)-weighted where spatial.

    (a) point_*: per-grid-point std / lag-1 autocorrelation, then cos(lat)-weighted mean over points.
    (b) area_mean_*: std / lag-1 autocorrelation of the cos(lat)-weighted area-averaged anomaly series.
    """
    w = np.cos(np.deg2rad(anom_train["latitude"].values))
    w = w / w.mean()
    x = anom_train.values  # (time, lat, lon)
    wgrid = np.broadcast_to(w[None, :, None], x.shape)
    series = (x * wgrid).sum(axis=(1, 2)) / wgrid.sum(axis=(1, 2))
    std_pt = np.nanstd(x, axis=0, ddof=1)
    xc = x - np.nanmean(x, axis=0)
    r1_pt = np.nansum(xc[:-1] * xc[1:], axis=0) / np.nansum(xc * xc, axis=0)
    wpt = np.broadcast_to(w[:, None], std_pt.shape)
    return {
        "point_anomaly_std_cosw_mean": float((std_pt * wpt).sum() / wpt.sum()),
        "point_lag1_autocorr_cosw_mean": float((r1_pt * wpt).sum() / wpt.sum()),
        "area_mean_anomaly_std": float(np.std(series, ddof=1)),
        "area_mean_lag1_autocorr": lag1_autocorr(series),
        "n_days": int(series.size),
    }


def main():
    """Figure 1 with panel (b) from the FINAL training climatology (cfg smoothing, training years only).

    region_stats.json layout: {"preliminary": <kept from the existing file, or computed from the unsmoothed
    day-of-year mean if absent>, "final": <computed here from the final climatology>}."""
    cfg = load_config()
    train = cfg["split"]["train"]
    names = list(cfg["regions"])
    path = RESULTS / "region_stats.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    prelim = old.get("preliminary") or (old if "kuroshio" in old else None)
    compute_prelim = prelim is None
    if compute_prelim:
        prelim = {"anomaly_definition": "SST minus training-years (%s to %s) unsmoothed day-of-year mean; preliminary" % tuple(train)}
    final = {"anomaly_definition": "SST minus FINAL training climatology (%s, years %s to %s); cos(lat) weights"
             % (cfg["climatology"]["smoothing"], *train)}
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 5 * len(names)))
    for i, region in enumerate(names):
        sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"].sel(time=slice(None, train[1])).load()
        b = cfg["regions"][region]
        box = f"{b['lat_min']}–{b['lat_max']}°N, {b['lon_min']}–{b['lon_max']}°E"
        tr = sst.sel(time=slice(*train))
        anom_tr = anomaly(tr, build_climatology(sst, cfg))
        panels = [
            (tr.mean("time"), "viridis", "Mean SST (°C)", "(a) Mean SST, training years"),
            (anom_tr.std("time"), "magma", "Std of anomaly (°C)",
             f"(b) Std of daily anomaly\n(anomaly vs final {cfg['climatology']['smoothing']} climatology, training years)"),
        ]
        for ax, (field, cmap, label, title) in zip(axes[i], panels):
            m = ax.pcolormesh(field.longitude, field.latitude, field, cmap=cmap, shading="auto")
            fig.colorbar(m, ax=ax, label=label)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"{region}: {title}\n{box}", fontsize=9)
        final[region] = region_stats(anom_tr)
        if compute_prelim:
            prelim[region] = region_stats(preliminary_anomaly(sst, train).sel(time=slice(*train)))
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fig1_regions.png", dpi=150)
    path.write_text(json.dumps({"preliminary": prelim, "final": final}, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
