import numpy as np
import pytest
import pandas as pd
import xarray as xr

from sst.data import qc_report
from sst.download import build_psl_url, build_url, check_year, expected_shape
from sst.figures import preliminary_anomaly, train_doy_mean

CFG = {
    "dataset": {"erddap_base": "https://example.org/erddap/griddap/ds", "variable": "sst",
                "psl_base": "https://example.org/dodsC/oisst", "psl_file_pattern": "sst.day.mean.{year}.nc"},
    "regions": {"r": {"lat_min": 30.125, "lat_max": 39.875, "lon_min": 145.125, "lon_max": 154.875}},
}


def make_sst(start, end, drop=(), dup=(), land_col=None, seed=0):
    t = pd.date_range(start, end, freq="D") + pd.Timedelta(hours=12)
    rng = np.random.default_rng(seed)
    data = rng.normal(20, 1, (len(t), 3, 4))
    da = xr.DataArray(data, dims=("time", "latitude", "longitude"),
                      coords={"time": t, "latitude": [1.0, 2.0, 3.0], "longitude": [1.0, 2.0, 3.0, 4.0]},
                      attrs={"units": "degree_C"})
    if land_col is not None:
        da[:, :, land_col] = np.nan
    if dup:
        da = xr.concat([da, da.sel(time=[pd.Timestamp(d) + pd.Timedelta(hours=12) for d in dup])], dim="time")
    if drop:
        keep = ~da["time"].dt.floor("D").isin(pd.to_datetime(list(drop)))
        da = da.isel(time=np.where(keep)[0])
    return da


def test_url_builder_exact():
    assert build_url(CFG, "r", 2020) == (
        "https://example.org/erddap/griddap/ds.nc?sst"
        "[(2020-01-01T12:00:00Z):1:(2020-12-31T12:00:00Z)][(0.0):1:(0.0)]"
        "[(30.125):1:(39.875)][(145.125):1:(154.875)]"
    )
    assert build_psl_url(CFG, 1985) == "https://example.org/dodsC/oisst/sst.day.mean.1985.nc"


def test_expected_shape_is_40x40():
    assert expected_shape(CFG["regions"]["r"]) == (40, 40)


def _year_ds(days, ny=2, nx=3):
    t = pd.DatetimeIndex(days) + pd.Timedelta(hours=12)
    return xr.Dataset({"sst": (("time", "latitude", "longitude"), np.full((len(t), ny, nx), 20.0))},
                      coords={"time": t})


def test_check_year_compares_with_server_days():
    box = {"lat_min": 0.0, "lat_max": 0.25, "lon_min": 0.0, "lon_max": 0.5}  # 2 x 3 grid
    server = pd.date_range("2000-01-01", "2000-12-31")
    server = server.delete([40, 41])                     # server itself lacks two days
    check_year(_year_ds(server), 2000, box, server)      # file has exactly the server's days -> ok
    with pytest.raises(ValueError, match="days present"):
        check_year(_year_ds(server[:-5]), 2000, box, server)   # truncated transfer -> rejected


def test_check_year_ignores_spillover_days():
    box = {"lat_min": 0.0, "lat_max": 0.25, "lon_min": 0.0, "lon_max": 0.5}
    server = pd.date_range("1999-12-31", "2000-12-31")
    ds = _year_ds(pd.date_range("1999-12-31", "2000-12-31"))   # 1999-12-31 spills into the file
    check_year(ds, 2000, box, server)


def test_fill_values_become_nan():
    from sst.data import normalise
    t = pd.date_range("2000-01-01", periods=2, freq="D")
    v = np.full((2, 2, 2), 15.0, dtype="float32")
    v[0, 0, 0] = np.float32(9.96921e36)    # netCDF default fill
    v[1, 1, 1] = np.float32(-9.96921e36)   # PSL missing_value
    da = xr.DataArray(v, dims=("time", "lat", "lon"), coords={"time": t, "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
                      attrs={"units": "degC"})
    out = normalise(da.to_dataset(name="sst"))["sst"].values
    assert np.isnan(out[0, 0, 0]) and np.isnan(out[1, 1, 1])
    assert np.isnan(out).sum() == 2 and np.nanmax(out) == 15.0


def test_qc_missing_and_duplicate():
    da = make_sst("2000-01-01", "2000-03-31", drop=["2000-02-10", "2000-02-11"], dup=["2000-03-01"])
    rep = qc_report(da, "2000-01-01", "2000-03-31")
    assert rep["missing_dates"] == ["2000-02-10", "2000-02-11"]
    assert rep["duplicate_dates"] == ["2000-03-01"]
    assert rep["days_expected"] == 91
    assert rep["days_present"] == 89
    assert rep["units"] == "degree_C"


def test_qc_land_detection():
    da = make_sst("2000-01-01", "2000-01-10", land_col=1)
    rep = qc_report(da, "2000-01-01", "2000-01-10")
    assert rep["n_allnan_points_land"] == 3  # whole column of 3 lats
    assert rep["n_partial_nan_points"] == 0


def test_qc_partial_nan_not_land():
    da = make_sst("2000-01-01", "2000-01-10")
    da[0, 0, 0] = np.nan
    rep = qc_report(da, "2000-01-01", "2000-01-10")
    assert rep["n_allnan_points_land"] == 0
    assert rep["n_partial_nan_points"] == 1


def test_anomaly_uses_training_years_only():
    train = ["2000-01-01", "2001-12-31"]
    a = make_sst("2000-01-01", "2003-12-31")
    b = a.copy(deep=True)
    b.loc[dict(time=slice("2002-01-01", "2003-12-31"))] += 100.0  # change val/test only
    clim_a, clim_b = train_doy_mean(a, train), train_doy_mean(b, train)
    np.testing.assert_array_equal(clim_a.values, clim_b.values)
    # anomalies in the training period are therefore identical too
    an_a = preliminary_anomaly(a, train).sel(time=slice(*train))
    an_b = preliminary_anomaly(b, train).sel(time=slice(*train))
    np.testing.assert_allclose(an_a.values, an_b.values)
    # and the shifted period has anomalies ~ +100 relative to before
    diff = preliminary_anomaly(b, train).sel(time="2003") - preliminary_anomaly(a, train).sel(time="2003")
    np.testing.assert_allclose(diff.values, 100.0)


def test_normalise_time_and_units():
    from sst.data import normalise
    t = pd.date_range("2000-01-01", periods=3, freq="D")
    for hour, units in ((0, "degC"), (12, "degree_C")):
        da = xr.DataArray(np.zeros((3, 2, 2)), dims=("time", "lat", "lon"),
                          coords={"time": t + pd.Timedelta(hours=hour), "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
                          attrs={"units": units})
        out = normalise(da.to_dataset(name="sst"))
        assert list(out["time"].values) == list(t.values)  # midnight calendar dates
        assert out["sst"].attrs["units"] == "degree_C"
        assert set(out.dims) == {"time", "latitude", "longitude"}
