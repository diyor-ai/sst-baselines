"""Synthetic tests for the test-period path (fixed alpha, training-only fit, no look-ahead)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst.climatology import build_climatology
from sst.ridge import ridge_lead
from sst.run_val_all import run_region

CFG = {"split": {"train": ["2000-01-01", "2003-12-31"], "val": ["2004-01-01", "2004-12-31"],
                 "test": ["2005-01-01", "2005-12-31"]},
       "climatology": {"years": "train", "smoothing": "harmonics3"}, "leads_days": [1, 7], "ocn_years": 2}


def synth(seed=0):
    t = pd.date_range("2000-01-01", "2005-12-31")
    rng = np.random.default_rng(seed)
    x = np.zeros((len(t), 3, 2))
    for i in range(1, len(t)):
        x[i] = 0.9 * x[i - 1] + rng.normal(size=(3, 2))
    season = 5 * np.sin(2 * np.pi * t.day_of_year.values / 365.25)[:, None, None]
    return xr.DataArray(20 + season + x, dims=("time", "latitude", "longitude"),
                        coords={"time": t, "latitude": [10.0, 20.0, 30.0], "longitude": [1.0, 2.0]})


def test_ridge_test_period_needs_alpha_and_val_forbids_it():
    sst = synth()
    clim = build_climatology(sst, CFG)
    with pytest.raises(ValueError):
        ridge_lead(sst, clim, CFG, 1, period="test")
    with pytest.raises(ValueError):
        ridge_lead(sst.sel(time=slice(None, "2004-12-31")), clim, CFG, 1, period="val", alpha=0.1)


def test_test_forecasts_do_not_use_data_after_the_issue_date():
    a = synth()
    cutoff = pd.Timestamp("2005-07-01")
    b = a.copy(deep=True)
    b.loc[dict(time=slice(cutoff + pd.Timedelta(days=1), None))] += 100.0   # change everything after the cutoff
    ca, cb = build_climatology(a, CFG), build_climatology(b, CFG)
    np.testing.assert_array_equal(ca.values, cb.values)  # climatology is training-only
    for h in (1, 7):
        fa, (ia, ta), _ = ridge_lead(a, ca, CFG, h, period="test", alpha=0.05)
        fb, (ib, tb), _ = ridge_lead(b, cb, CFG, h, period="test", alpha=0.05)
        keep = ia <= cutoff
        assert keep.any() and (~keep).any()
        np.testing.assert_allclose(fa.values[keep], fb.values[keep])
        assert not np.allclose(fa.values[~keep], fb.values[~keep])  # the check can fail


def test_run_region_test_period_fixed_alpha_and_counts():
    sst = synth()
    clim = build_climatology(sst, CFG)
    res, bias, info = run_region(sst, CFG, clim, "synthetic", period="test", alphas={1: 0.1, 7: 1.0}, season_year=True)
    assert info["1"]["alpha"] == 0.1 and info["7"]["alpha"] == 1.0
    p = res["1"]["pairs"]
    assert p["expected"] == 365 and p["used"] == 365 and p["skipped"] == 0
    assert res["1"]["methods"]["ridge"]["overall"]["n_days"] == 365
    assert all(k.split("|")[1] == "2005" for k in res["1"]["methods"]["ridge"]["by_season_year"])
    assert set(res["1"]["methods"]) == {"persistence", "climatology", "anomaly_persistence", "damped_persistence",
                                         "ocn", "ridge"}


def test_different_pair_sets_between_methods_fail_loudly():
    """A missing day makes Ridge lose extra pairs (incomplete 30-day history); scoring must refuse, not mix pair sets."""
    sst = synth().drop_sel(time=pd.to_datetime(["2005-03-10"]))
    clim = build_climatology(sst, CFG)
    with pytest.raises(AssertionError):
        run_region(sst, CFG, clim, "synthetic", period="test", alphas={1: 0.1, 7: 1.0})
