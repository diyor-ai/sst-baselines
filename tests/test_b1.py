"""Synthetic tests for B1 (spatial context). No network; real data only in the one test that is skipped if the files are absent
(and that one reads days <= 2019-12-31 only). Tests needing torch are skipped cleanly if torch is absent."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst.b1_data import (SEQ_LEN, gather_b1, gather_inputs_b1, inner_block, make_data_b1, patch_index)
from sst.b1_ridge import fit_point, point_block, predict_point, run_region
from sst.lstm_data import day_features, gather, make_data
from sst.ridge import ALPHA_GRID, fit_ridge, make_features, predict_ridge, ridge_lead

CFG = {"split": {"train": ["1982-01-01", "2015-12-31"], "val": ["2016-01-01", "2019-12-31"], "test": ["2020-01-01", "2025-12-31"]},
       "leads_days": [1, 7, 30], "climatology": {"years": "train", "smoothing": "harmonics3", "feb29": "use_feb28"}}
DATES = pd.date_range("1982-01-01", "2019-12-31")
HALO, SIZE, NI = 4, 5, (3, 4)


def make_sst(n_inner, halo, seed=0):
    rng = np.random.default_rng(seed)
    ny, nx = n_inner[0] + 2 * halo, n_inner[1] + 2 * halo
    doy = DATES.day_of_year.values[:, None, None]
    ar = np.zeros((len(DATES), ny, nx))
    shock = rng.normal(size=ar.shape)
    for i in range(1, len(DATES)):
        ar[i] = 0.9 * ar[i - 1] + shock[i]
    vals = 20 + 5 * np.sin(2 * np.pi * doy / 365) + ar
    return xr.DataArray(vals, dims=("time", "latitude", "longitude"),
                        coords={"time": DATES, "latitude": 20 + 0.25 * np.arange(ny), "longitude": 150 + 0.25 * np.arange(nx)}, name="sst")


@pytest.fixture(scope="module")
def sst():
    return make_sst(NI, HALO)


@pytest.fixture(scope="module")
def data(sst):
    return make_data_b1(sst, CFG, HALO, SIZE)


# ---------------------------------------------------------------- patch indexing
def test_patch_indexing_centre_neighbours_and_edges():
    pidx = patch_index(NI, HALO, SIZE)
    ny, nx = NI[0] + 2 * HALO, NI[1] + 2 * HALO
    code = (np.arange(ny)[:, None] * 1000 + np.arange(nx)[None, :]).reshape(-1)    # value = 1000*row + col on the FULL grid
    assert pidx.shape == (NI[0] * NI[1], 25)
    for p in range(pidx.shape[0]):
        i, j = divmod(p, NI[1])
        vals = code[pidx[p]]
        assert vals[12] == (i + HALO) * 1000 + (j + HALO)                           # centre = the target point itself
        want = [(i + HALO + di) * 1000 + (j + HALO + dj) for di in range(-2, 3) for dj in range(-2, 3)]
        assert list(vals) == want                                                   # right neighbours, row-major
    # edge point (0, 0): its patch rows/cols start 2 points outside the inner box = inside the halo (full-grid rows/cols 2..6)
    v = code[pidx[0]]
    assert v.min() == 2 * 1000 + 2 and v.max() == 6 * 1000 + 6
    assert (v // 1000).min() < HALO and (v % 1000).min() < HALO                    # halo rows/cols are used
    # corner (last) point likewise on the far side
    v = code[pidx[-1]]
    i, j = NI[0] - 1, NI[1] - 1
    assert v.max() == (i + HALO + 2) * 1000 + (j + HALO + 2) and (v // 1000).max() >= NI[0] + HALO


def test_gathered_inputs_match_patch_of_anomalies(data):
    flat = data.train_pairs[[10, 3000]]
    x = gather_inputs_b1(data, flat)
    assert x.shape == (2, 30, 27) and x.dtype == np.float32
    ny, nx = NI[0] + 2 * HALO, NI[1] + 2 * HALO
    for k, f in enumerate(flat):
        t, p = divmod(int(f), data.n_points)
        i, j = divmod(p, NI[1])
        for d in range(30):
            blk = data.a[t - 29 + d].reshape(ny, nx)[i + HALO - 2:i + HALO + 3, j + HALO - 2:j + HALO + 3].reshape(-1)
            assert np.array_equal(x[k, d, :25], blk)
        assert np.allclose(x[k, :, 25:], day_features(data.dates[t - 29:t + 1]))


def test_halo_values_enter_edge_patches_but_not_interior_ones(sst):
    d0 = make_data_b1(sst, CFG, HALO, SIZE)
    s2 = sst.copy(deep=True)
    s2.values[:, 2, 2] += 3.0                                   # full-grid (2, 2) = a halo point; in the patch of inner (0, 0) only
    d1 = make_data_b1(s2, CFG, HALO, SIZE, clim=d0.clim)
    t = int(d0.train_pairs[50] // d0.n_points)
    f_edge, f_far = t * d0.n_points + 0, t * d0.n_points + (NI[0] * NI[1] - 1)
    assert not np.array_equal(gather_inputs_b1(d0, np.array([f_edge])), gather_inputs_b1(d1, np.array([f_edge])))
    assert np.array_equal(gather_inputs_b1(d0, np.array([f_far])), gather_inputs_b1(d1, np.array([f_far])))
    assert d0.std == d1.std                                     # the std comes from the inner points only


# ---------------------------------------------------------------- no leakage
def test_inputs_use_only_days_up_to_t(data):
    flat = data.train_pairs[[100, 5000]]
    x0 = gather_inputs_b1(data, flat)
    t = flat // data.n_points
    for k in range(2):
        a2 = data.a.copy()
        a2[t[k] + 1:] = 123.0
        x1 = gather_inputs_b1(type(data)(**{**data.__dict__, "a": a2}), flat[k:k + 1])
        assert np.array_equal(x1[0], x0[k])
    a3 = data.a.copy()
    a3[t[0] - 3] += 1.0
    assert not np.array_equal(gather_inputs_b1(type(data)(**{**data.__dict__, "a": a3}), flat[:1])[0], x0[0])   # can fail


def test_no_training_target_after_2015(data):
    t = data.train_pairs // data.n_points
    assert data.dates[t].max() + pd.Timedelta(days=30) <= pd.Timestamp("2015-12-31")
    assert data.dates[t].min() - pd.Timedelta(days=SEQ_LEN - 1) >= pd.Timestamp("1982-01-01")
    x, y = gather_b1(data, data.train_pairs[:50])
    p = data.train_pairs[:50] % data.n_points
    assert np.array_equal(y[:, 2], data.a[t[:50] + 30, data.centre[p]])               # target = centre point at t+30


def test_validation_samples(data):
    t = data.val_pairs // data.n_points
    assert data.dates[t].min() == pd.Timestamp("2016-01-01") and data.dates[t].max() == pd.Timestamp("2019-12-01")
    assert len(data.val_pairs) == len(np.unique(t)) * data.n_points


def test_changing_validation_data_does_not_change_training_set(sst, data):
    s2 = sst.copy(deep=True)
    after = (s2.time > pd.Timestamp("2015-12-31")).values
    s2.values[after] += np.random.default_rng(1).normal(5, 3, size=s2.values[after].shape)
    d2 = make_data_b1(s2, CFG, HALO, SIZE)
    assert np.array_equal(d2.train_pairs, data.train_pairs) and d2.std == data.std
    assert np.array_equal(d2.clim.values, data.clim.values)
    sel = data.train_pairs[::997]
    xa, ya = gather_b1(data, sel)
    xb, yb = gather_b1(d2, sel)
    assert np.array_equal(xa, xb) and np.array_equal(ya, yb)


def test_test_period_cannot_be_loaded(sst):
    longer = xr.concat([sst, sst.assign_coords(time=sst.time + pd.Timedelta(days=len(DATES)))], dim="time")
    with pytest.raises(ValueError):
        make_data_b1(longer, CFG, HALO, SIZE)


# ---------------------------------------------------------------- inner 40x40 unchanged
def test_inner_block_gives_the_same_samples_as_the_existing_inner_only_data(sst, data):
    inner = sst.isel(latitude=slice(HALO, HALO + NI[0]), longitude=slice(HALO, HALO + NI[1]))
    old = make_data(inner, CFG)
    assert np.array_equal(old.train_pairs, data.train_pairs) and np.array_equal(old.val_pairs, data.val_pairs)
    assert np.isclose(old.std, data.std, rtol=1e-12, atol=0)
    assert np.allclose(inner_block(data.clim.values, HALO), old.clim.values, atol=1e-10, rtol=0)
    assert np.allclose(data.a[:, data.centre], old.a, atol=1e-6)                      # targets are the same numbers


def test_equals_existing_data_when_patch_is_one_point_and_no_halo(sst):
    inner = sst.isel(latitude=slice(HALO, HALO + NI[0]), longitude=slice(HALO, HALO + NI[1]))
    old = make_data(inner, CFG)
    new = make_data_b1(inner, CFG, 0, 1)
    assert new.n_in == 3 and np.array_equal(new.train_pairs, old.train_pairs) and np.array_equal(new.val_pairs, old.val_pairs)
    for flat in (old.train_pairs[::1234][:20], old.val_pairs[::777][:20]):
        xo, yo = gather(old, flat)
        xn, yn = gather_b1(new, flat)
        assert np.array_equal(xo, xn) and np.array_equal(yo, yn)


REAL = ["kuroshio", "subtropical"]


@pytest.mark.parametrize("region", REAL)
def test_real_halo_file_inner_block_equals_existing_file(region):
    from sst.download import ROOT
    fh, fi = ROOT / "data" / "processed" / f"{region}_halo.nc", ROOT / "data" / "processed" / f"{region}.nc"
    if not (fh.exists() and fi.exists()):
        pytest.skip("halo data not downloaded")
    h = xr.open_dataset(fh)["sst"].sel(time=slice(None, "2019-12-31")).load()      # validation end: nothing later is read
    i = xr.open_dataset(fi)["sst"].sel(time=slice(None, "2019-12-31")).load()
    assert h.shape[1:] == (48, 48) and i.shape[1:] == (40, 40)
    assert np.array_equal(inner_block(h.values, HALO), i.values, equal_nan=True)
    assert np.array_equal(h.latitude.values[HALO:-HALO], i.latitude.values) and np.array_equal(h.longitude.values[HALO:-HALO], i.longitude.values)


# ---------------------------------------------------------------- ridge
def test_ridge_point_fit_equals_existing_ridge_code():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 1, 8)) * [1, 2, 3, 4, 5, 6, 7, 8]
    y = X[:, :, 0] * 0.5 + rng.normal(size=(300, 1))
    m = fit_ridge(X, y, ALPHA_GRID)
    mu, sd, ym, coef = fit_point(X[:, 0, :], y[:, 0], ALPHA_GRID)
    assert np.allclose(coef, m["coef"][:, 0, :], atol=1e-10) and np.allclose(mu, m["mu"][0]) and np.allclose(sd, m["sd"][0])
    Xn = rng.normal(size=(20, 1, 8))
    assert np.allclose(predict_point(mu, sd, ym, coef, Xn[:, 0, :]).T, predict_ridge(m, Xn)[:, :, 0], atol=1e-10)


def test_ridge_patch_features_centre_is_the_existing_feature_and_indexing_is_right(sst, data):
    ny, nx = NI[0] + 2 * HALO, NI[1] + 2 * HALO
    a = (sst.values - sst.values.mean(0)).reshape(len(DATES), -1)
    issue = pd.date_range("2000-01-01", periods=5)
    F = make_features(a, DATES[0], issue).reshape(5, ny, nx, 8)
    for p in (0, 5, NI[0] * NI[1] - 1):
        i, j = divmod(p, NI[1])
        blk0 = point_block(F, i, j, HALO, 0)
        assert np.array_equal(blk0, make_features(a[:, [(i + HALO) * nx + j + HALO]], DATES[0], issue)[:, 0, :])   # = existing Ridge features
        blk = point_block(F, i, j, HALO, 2).reshape(5, 25, 8)
        assert np.array_equal(blk[:, 12], blk0)                                 # centre of the 25 = own point
        for k, (di, dj) in enumerate((di, dj) for di in range(-2, 3) for dj in range(-2, 3)):
            assert np.array_equal(blk[:, k], F[:, i + HALO + di, j + HALO + dj, :])


def test_ridge_features_use_only_days_up_to_issue(sst):
    ny, nx = NI[0] + 2 * HALO, NI[1] + 2 * HALO
    a = (sst.values - sst.values.mean(0)).reshape(len(DATES), -1)
    issue = pd.DatetimeIndex(["2005-06-01", "2010-03-15"])
    F0 = make_features(a, DATES[0], issue)
    for k, d in enumerate(issue):
        a2 = a.copy()
        a2[(d - DATES[0]).days + 1:] = 99.0
        assert np.array_equal(make_features(a2, DATES[0], issue[k:k + 1])[0], F0[k])


def _small_cfg_b1():
    return {"halo": {"width_points": HALO}, "patch": {"size": SIZE}}


def test_ridge_without_patch_and_doy_reproduces_existing_ridge_forecasts(sst):
    """radius 0, no doy == existing Ridge (ridge_lead with the same alpha); radius-0 + doy differs (can fail)."""
    res, preds, issue = run_region("x", CFG, _small_cfg_b1(), models=(("plain", 0, False), ("with_doy", 0, True)), log=lambda *_: None, sst=sst)
    inner = sst.isel(latitude=slice(HALO, HALO + NI[0]), longitude=slice(HALO, HALO + NI[1]))
    cfg_t = {**CFG, "split": {**CFG["split"], "test": CFG["split"]["val"]}}                 # existing code path with a fixed alpha
    from sst.climatology import build_climatology, clim_at
    clim = build_climatology(inner, CFG)
    for h in CFG["leads_days"]:
        alpha = res["plain"][str(h)]["alpha"]
        f, _, _ = ridge_lead(inner, clim, cfg_t, h, period="test", alpha=alpha)
        tgt = issue + pd.Timedelta(days=h)
        ct = clim_at(clim, tgt).values
        old_anom = f.sel(time=tgt).values - ct
        assert np.allclose(preds["plain"][str(h)], old_anom, atol=1e-8)
        assert not np.allclose(preds["with_doy"][str(h)], old_anom, atol=1e-8)


def test_ridge_alpha_chosen_on_validation_and_training_rows_end_in_2015(sst):
    res, _, _ = run_region("x", CFG, _small_cfg_b1(), models=(("patch", 2, True),), log=lambda *_: None, sst=sst)
    for h in CFG["leads_days"]:
        r = res["patch"][str(h)]
        assert r["alpha"] == r["alphas"][int(np.argmin(r["val_rmse"]))] and r["n_features"] == 202
        assert r["train_pairs"]["used"] > 10000


# ---------------------------------------------------------------- LSTM training
LCFG = {"lr": 1e-3, "batch_size": 64, "max_epochs": 2, "patience": 5, "num_threads": 1, "samples_per_epoch": 512}


def test_same_seed_gives_identical_loss(data):
    pytest.importorskip("torch")
    from sst.b1_train import train_run_b1
    r1 = train_run_b1(data, 4, 0, LCFG, verbose=False)
    r2 = train_run_b1(data, 4, 0, LCFG, verbose=False)
    r3 = train_run_b1(data, 4, 1, LCFG, verbose=False)
    assert r1["history"]["train_loss"] == r2["history"]["train_loss"] and r1["history"]["val_loss"] == r2["history"]["val_loss"]
    assert r1["history"]["val_loss"] != r3["history"]["val_loss"]
    assert data.n_in == 27


def test_training_loop_equals_existing_loop_for_one_point_patch(sst):
    """The B1 loop is a copy of lstm_train.train_run: with a 1x1 patch and no halo the whole loss history must be identical."""
    pytest.importorskip("torch")
    import torch
    from sst.b1_train import train_run_b1
    from sst.lstm_train import train_run
    inner = sst.isel(latitude=slice(HALO, HALO + NI[0]), longitude=slice(HALO, HALO + NI[1]))
    old = train_run(make_data(inner, CFG), 4, 0, LCFG, verbose=False)
    new = train_run_b1(make_data_b1(inner, CFG, 0, 1), 4, 0, LCFG, verbose=False)
    assert old["history"]["train_loss"] == new["history"]["train_loss"] and old["history"]["val_loss"] == new["history"]["val_loss"]
    assert old["best_epoch"] == new["best_epoch"]
    assert all(torch.equal(old["best_state"][k], new["best_state"][k]) for k in old["best_state"])
