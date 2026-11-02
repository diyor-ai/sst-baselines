import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst.baselines import (anomaly_persistence, climatology_forecast, damped_persistence, lag_corr, ocn,
                           persistence)
from sst.climatology import (build_climatology, clim_at, doy365, raw_doy_mean, smooth_harmonics,
                             smooth_window)
from sst.metrics import acc, mae, rmse, scores, season_labels, skill_score, valid_pairs


def synth(start, end, seed=0, lats=(10.0, 20.0), lons=(1.0, 2.0, 3.0), drop=()):
    t = pd.date_range(start, end)
    rng = np.random.default_rng(seed)
    da = xr.DataArray(rng.normal(20, 2, (len(t), len(lats), len(lons))), dims=("time", "latitude", "longitude"),
                      coords={"time": t, "latitude": list(lats), "longitude": list(lons)})
    return da.drop_sel(time=pd.to_datetime(list(drop))) if drop else da


def cfg(smoothing, train):
    return {"split": {"train": train}, "climatology": {"years": "train", "smoothing": smoothing}}


# ---- climatology -------------------------------------------------------------------------
def test_feb29_maps_to_feb28():
    assert doy365(["2004-02-29"])[0] == doy365(["2004-02-28"])[0] == 58
    assert doy365(["2004-03-01"])[0] == 59 == doy365(["2003-03-01"])[0]


@pytest.mark.parametrize("smoothing", ["harmonics3", "window31"])
def test_no_leakage_climatology_and_damping_use_training_only(smoothing):
    train = ["2000-01-01", "2002-12-31"]
    a = synth("2000-01-01", "2005-12-31")
    b = a.copy(deep=True)
    b.loc[dict(time=slice("2003-01-01", "2005-12-31"))] += 100.0  # change validation/test only
    ca, cb = build_climatology(a, cfg(smoothing, train)), build_climatology(b, cfg(smoothing, train))
    np.testing.assert_array_equal(ca.values, cb.values)
    np.testing.assert_array_equal(lag_corr(a, ca, train, 7).values, lag_corr(b, cb, train, 7).values)


def test_invalid_or_undecided_smoothing_raises():
    a = synth("2000-01-01", "2002-12-31")
    with pytest.raises(ValueError, match="harmonics3"):
        build_climatology(a, cfg("TBD_day2", ["2000-01-01", "2002-12-31"]))
    bad = cfg("window31", ["2000-01-01", "2002-12-31"]); bad["climatology"]["years"] = "all"
    with pytest.raises(ValueError, match="leaks"):
        build_climatology(a, bad)


def test_harmonics_recover_annual_cycle():
    t = pd.date_range("2000-01-01", "2003-12-31")
    d = doy365(t)
    cycle = 15 + 5 * np.cos(2 * np.pi * d / 365 + 0.3) + 1.5 * np.sin(4 * np.pi * d / 365)
    sst = xr.DataArray(cycle[:, None, None] * np.ones((1, 2, 2)), dims=("time", "latitude", "longitude"),
                       coords={"time": t, "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]})
    out = smooth_harmonics(raw_doy_mean(sst, ["2000-01-01", "2003-12-31"]), 3)
    np.testing.assert_allclose(out.values[:, 0, 0], 15 + 5 * np.cos(2 * np.pi * np.arange(365) / 365 + 0.3)
                               + 1.5 * np.sin(4 * np.pi * np.arange(365) / 365), atol=1e-9)


def test_window_is_circular():
    spike = np.zeros((365, 1, 1)); spike[0] = 31.0
    clim = xr.DataArray(spike, dims=("doy", "latitude", "longitude"),
                        coords={"latitude": [0.0], "longitude": [0.0]})
    out = smooth_window(clim, 31).values[:, 0, 0]
    assert np.allclose(out[:16], 1.0) and np.allclose(out[-15:], 1.0)  # wraps across Dec 31 -> Jan 1
    assert np.allclose(out[16:-15], 0.0)


# ---- baselines + metrics ---------------------------------------------------------------------
def one_cell(values, start="2000-01-01"):
    t = pd.date_range(start, periods=len(values))
    return xr.DataArray(np.array(values, float)[:, None, None], dims=("time", "latitude", "longitude"),
                        coords={"time": t, "latitude": [0.0], "longitude": [0.0]})


def test_persistence_rmse_matches_hand_value():
    sst = one_cell([10, 12, 15, 14, 20])
    issue, target, counts = valid_pairs(sst.time.values, "2000-01-02", "2000-01-05", 1)
    f = persistence(sst, issue, target)
    # errors: 10-12=-2, 12-15=-3, 15-14=1, 14-20=-6 -> squares 4+9+1+36 = 50 -> sqrt(50/4)
    assert rmse(f, sst.sel(time=target)) == pytest.approx(np.sqrt(12.5))
    assert mae(f, sst.sel(time=target)) == pytest.approx((2 + 3 + 1 + 6) / 4)
    assert counts["used"] == 4 and counts["skipped"] == 0


def test_ocn_uses_only_past_years():
    sst = synth("2000-01-01", "2005-12-31", seed=3)
    issue, target = pd.DatetimeIndex(["2005-03-10"]), pd.DatetimeIndex(["2005-03-17"])
    hand = np.mean([sst.sel(time=f"{y}-03-17").values for y in (2002, 2003, 2004)], axis=0)
    np.testing.assert_allclose(ocn(sst, issue, target, 3).values[0], hand)
    tampered = sst.copy(deep=True)
    tampered.loc[dict(time=slice("2005-03-10", None))] = 1e6   # issue date and later
    tampered.loc[dict(time=slice(None, "2001-12-31"))] = 1e6   # older than the N=3 window
    np.testing.assert_allclose(ocn(tampered, issue, target, 3).values[0], hand)


def test_ocn_feb29_uses_feb28_of_past_years():
    sst = synth("2000-01-01", "2004-12-31", seed=4)
    out = ocn(sst, pd.DatetimeIndex(["2004-02-22"]), pd.DatetimeIndex(["2004-02-29"]), 2)
    hand = np.mean([sst.sel(time=f"{y}-02-28").values for y in (2002, 2003)], axis=0)
    np.testing.assert_allclose(out.values[0], hand)


def test_all_baselines_run_and_relate():
    train = ["2000-01-01", "2002-12-31"]
    sst = synth("2000-01-01", "2004-12-31", seed=5)
    clim = build_climatology(sst, cfg("window31", train))
    issue, target, _ = valid_pairs(sst.time.values, "2004-01-01", "2004-12-31", 7)
    r = lag_corr(sst, clim, train, 7)
    pers = persistence(sst, issue, target)
    ap = anomaly_persistence(sst, clim, issue, target)
    dp = damped_persistence(sst, clim, issue, target, r)
    cf = climatology_forecast(sst, clim, issue, target)
    assert pers.shape == ap.shape == dp.shape == cf.shape
    # r = 1 gives anomaly persistence; r = 0 gives climatology
    one = r.copy(data=np.ones(r.shape)); zero = r.copy(data=np.zeros(r.shape))
    np.testing.assert_allclose(damped_persistence(sst, clim, issue, target, one).values, ap.values)
    np.testing.assert_allclose(damped_persistence(sst, clim, issue, target, zero).values, cf.values)
    assert np.all(np.abs(r.values) <= 1)


def test_perfect_forecast_scores():
    o = synth("2000-01-01", "2000-03-31", seed=6)
    c = o * 0 + 20.0
    assert rmse(o, o) == 0 and mae(o, o) == 0
    assert acc(o, o, c) == pytest.approx(1.0)
    s = scores(o, o, c)
    assert s["rmse"] == 0 and s["acc"] == pytest.approx(1.0) and s["n_days"] == 91


def test_skill_zero_when_model_equals_reference():
    o = synth("2000-01-01", "2000-02-29", seed=7)
    f = o + 1.0
    assert skill_score(rmse(f, o), rmse(f, o)) == 0.0
    assert skill_score(0.0, rmse(f, o)) == 1.0


def test_lat_weights_make_equator_count_more():
    t = pd.date_range("2000-01-01", periods=1)
    o = xr.DataArray(np.zeros((1, 2, 1)), dims=("time", "latitude", "longitude"),
                     coords={"time": t, "latitude": [0.0, 60.0], "longitude": [0.0]})
    f = o.copy(data=np.array([[[1.0], [3.0]]]))  # error 1 at the equator, 3 at 60N
    w = np.array([1.0, 0.5]); w = w / w.mean()
    assert rmse(f, o) == pytest.approx(np.sqrt((w * np.array([1.0, 9.0])).sum() / w.sum()))


def test_missing_days_skipped_and_counted():
    sst = synth("2000-01-01", "2000-01-10", drop=["2000-01-04", "2000-01-08"])
    issue, target, c = valid_pairs(sst.time.values, "2000-01-02", "2000-01-10", 1)
    assert c == {"lead_days": 1, "expected": 9, "used": 5, "skipped": 4,
                 "skipped_input_missing": 2, "skipped_target_missing": 2}
    assert not any(d in (pd.Timestamp("2000-01-04"), pd.Timestamp("2000-01-08")) for d in list(issue) + list(target))
    f = persistence(sst, issue, target)
    assert not np.isnan(f.values).any()
    zero_clim = xr.DataArray(np.zeros((365, 2, 3)), dims=("doy", "latitude", "longitude"),
                             coords={"latitude": sst.latitude, "longitude": sst.longitude})
    assert scores(f, sst.sel(time=target), clim_at(zero_clim, target))["n_days"] == 5


def test_season_labels():
    assert list(season_labels(["2000-12-15", "2001-01-10", "2001-04-01", "2001-07-01", "2001-10-31"])) == \
        ["DJF", "DJF", "MAM", "JJA", "SON"]
