"""Scores for forecasts: RMSE, MAE, ACC, skill, cos(lat) weights, seasons, per-year scores."""
import numpy as np
import pandas as pd

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def lat_weights(lat):
    """cos(latitude) weights normalised to mean 1."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))
    return w / w.mean()


def valid_pairs(available, eval_start, eval_end, h):
    """Issue/target dates for one lead. A forecast is skipped (and counted) if its input day or
    its target day is missing from `available`. Targets are every day in [eval_start, eval_end]."""
    available = pd.DatetimeIndex(available)
    target = pd.date_range(eval_start, eval_end)
    issue = target - pd.Timedelta(days=h)
    in_ok, tg_ok = issue.isin(available), target.isin(available)
    ok = in_ok & tg_ok
    counts = {"lead_days": h, "expected": int(len(target)), "used": int(ok.sum()),
              "skipped": int((~ok).sum()), "skipped_input_missing": int((~in_ok).sum()),
              "skipped_target_missing": int((~tg_ok).sum())}
    return issue[ok], target[ok], counts


def _w3(da, w_lat):
    return np.broadcast_to(w_lat[None, :, None], da.shape)


def _wmean(x, w):
    m = ~np.isnan(x)
    return float((x[m] * w[m]).sum() / w[m].sum())


def rmse(f, o):
    w = _w3(f, lat_weights(f["latitude"]))
    return float(np.sqrt(_wmean((f.values - o.values) ** 2, w)))


def mae(f, o):
    w = _w3(f, lat_weights(f["latitude"]))
    return _wmean(np.abs(f.values - o.values), w)


def acc(f, o, clim_t):
    """Anomaly correlation: per date, weighted spatial correlation of (f - clim) and (o - clim)
    (not re-centred), then averaged over dates. Dates with zero variance are ignored."""
    w = _w3(f, lat_weights(f["latitude"]))
    fa, oa = f.values - clim_t.values, o.values - clim_t.values
    vals = []
    for i in range(fa.shape[0]):
        m = ~(np.isnan(fa[i]) | np.isnan(oa[i]))
        wi = w[i][m]
        den = np.sqrt((wi * fa[i][m] ** 2).sum() * (wi * oa[i][m] ** 2).sum())
        if den > 0:
            vals.append((wi * fa[i][m] * oa[i][m]).sum() / den)
    return float(np.mean(vals)) if vals else float("nan")


def skill_score(model, reference):
    """1 - model/reference for an error score (RMSE or MAE): 0 = no better than reference, 1 = perfect."""
    return 1.0 - model / reference


def scores(f, o, clim_t):
    return {"rmse": rmse(f, o), "mae": mae(f, o), "acc": acc(f, o, clim_t), "n_days": int(f.sizes["time"])}


def scores_by(f, o, clim_t, labels):
    """Scores per group label (labels has one entry per time step)."""
    labels = np.asarray(labels)
    return {str(g): scores(f.isel(time=labels == g), o.isel(time=labels == g), clim_t.isel(time=labels == g))
            for g in pd.unique(labels)}


def season_labels(times):
    return np.array([SEASONS[m] for m in pd.DatetimeIndex(times).month])


def year_labels(times):
    return pd.DatetimeIndex(times).year.values
