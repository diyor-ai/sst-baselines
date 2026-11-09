"""Tests for B2 (whole-region models). Synthetic data except the tests marked real (they read days <= 2019-12-31 only, skipped if files absent)."""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from sst.b1_data import make_data_b1
from sst.b2_data import GRID, HALO, crop, gather_x, gather_y, make_data_b2
from sst.b2_models import build
from sst.b2_train import train_run_b2, val_loss
from sst.download import ROOT
from sst.lstm_data import make_data

CFG = {"split": {"train": ["1982-01-01", "2015-12-31"], "val": ["2016-01-01", "2019-12-31"], "test": ["2020-01-01", "2025-12-31"]},
       "leads_days": [1, 7, 30], "climatology": {"years": "train", "smoothing": "harmonics3", "feb29": "use_feb28"}}
DATES = pd.date_range("1982-01-01", "2019-12-31")
TCFG = {"lr": 1e-3, "batch_size": 4, "max_epochs": 2, "patience": 8, "num_threads": 2}


def make_sst(seed=0):
    rng = np.random.default_rng(seed)
    doy = DATES.day_of_year.values[:, None, None]
    ar = np.zeros((len(DATES), GRID, GRID))
    shock = rng.normal(size=ar.shape)
    for i in range(1, len(DATES)):
        ar[i] = 0.9 * ar[i - 1] + shock[i]
    return xr.DataArray(20 + 5 * np.sin(2 * np.pi * doy / 365) + ar, dims=("time", "latitude", "longitude"),
                        coords={"time": DATES, "latitude": 20 + 0.25 * np.arange(GRID), "longitude": 150 + 0.25 * np.arange(GRID)}, name="sst")


@pytest.fixture(scope="module")
def sst():
    return make_sst()


@pytest.fixture(scope="module")
def data(sst):
    return make_data_b2(sst, CFG)


def tiny(data, n_tr=8, n_va=4):
    """Small copy of the data (a few issue dates) for fast training tests."""
    d = type(data)(**{**data.__dict__})
    d.train_issue, d.val_issue = data.train_issue[1000:1000 + n_tr], data.val_issue[:n_va]
    return d


# ---------------------------------------------------------------- no future inputs
def test_no_future_inputs(data):
    for ti in (5000, 9000, int(data.val_issue[10])):
        t = np.array([ti])
        x0, y0 = gather_x(data, t), gather_y(data, t)
        d2 = type(data)(**{**data.__dict__, "a": data.a.copy()})
        d2.a[ti + 1:] = 1e6                      # destroy EVERYTHING after the issue day
        assert np.array_equal(x0, gather_x(d2, t))       # inputs unchanged -> they never read a day later than t
        assert not np.array_equal(y0, gather_y(d2, t))   # targets do read the future (sanity: the perturbation is visible)
    t = np.array([5000, 9000, int(data.val_issue[10])])
    x0 = gather_x(data, t)
    # input window is exactly days t-29..t, and doy channels are those of the same days
    a_t = data.a[t[0] - 29:t[0] + 1].reshape(30, GRID, GRID)
    assert np.array_equal(x0[0, :, 0], a_t)
    assert np.array_equal(x0[0, :, 1, 0, 0], data.dayf[t[0] - 29:t[0] + 1, 0])
    assert np.all(x0[0, :, 2] == x0[0, :, 2, :1, :1])      # doy channels are constant maps


def test_doy_channels_are_sin_cos(data):
    t = np.array([4000])
    x = gather_x(data, t)
    doy = pd.DatetimeIndex(data.dates[t[0] - 29:t[0] + 1]).day_of_year.values
    assert x.shape == (1, 30, 3, GRID, GRID)
    assert np.allclose(x[0, :, 1, 3, 4] ** 2 + x[0, :, 2, 3, 4] ** 2, 1, atol=1e-6)
    assert np.corrcoef(x[0, :, 1, 0, 0], np.sin(2 * np.pi * doy / 365))[0, 1] > 0.999    # Feb 29 -> Feb 28 convention in doy365


# ---------------------------------------------------------------- no training target after 2015-12-31
def test_no_training_target_after_2015(data):
    tr = data.dates[data.train_issue]
    assert tr.min() - pd.Timedelta(days=29) >= pd.Timestamp("1982-01-01")             # all 30 input days inside the training years
    assert (tr + pd.Timedelta(days=max(data.leads))).max() <= pd.Timestamp("2015-12-31")   # all targets inside
    assert tr.max() == pd.Timestamp("2015-12-01")
    assert len(tr) == (pd.Timestamp("2015-12-01") - pd.Timestamp("1982-01-30")).days + 1


def test_training_loop_reads_only_training_targets(data, monkeypatch):
    seen = []
    import sst.b2_train as bt
    orig = bt.gather_y

    def spy(d, t):
        seen.append(np.asarray(t))
        return orig(d, t)
    monkeypatch.setattr(bt, "gather_y", spy)
    d = tiny(data)
    train_run_b2(d, "unet", 16, 0, TCFG, max_epochs=1, verbose=False)
    train_t = set(d.train_issue.tolist())
    n_train_calls = [s for s in seen if set(s.tolist()) <= train_t]
    assert len(n_train_calls) >= 2                                                    # training batches
    for s in seen:                                                                    # others are validation batches
        assert set(s.tolist()) <= train_t or set(s.tolist()) <= set(d.val_issue.tolist())
    assert max(d.dates[t] + pd.Timedelta(days=30) for s in n_train_calls for t in s) <= pd.Timestamp("2015-12-31")


# ---------------------------------------------------------------- crop
def test_crop_is_exactly_the_inner_box():
    code = (np.arange(GRID)[:, None] * 1000 + np.arange(GRID)[None, :]).astype(float)
    c = crop(code[None, None])[0, 0]
    assert c.shape == (40, 40)
    assert c[0, 0] == 4 * 1000 + 4 and c[-1, -1] == 43 * 1000 + 43
    assert np.array_equal(c, code[4:44, 4:44])


def test_gather_y_is_inner_anomaly_in_row_major_order(data):
    t = np.array([6000])
    y = gather_y(data, t)
    assert y.shape == (1, 3, 40, 40)
    for k, h in enumerate(data.leads):
        assert np.array_equal(y[0, k], data.a[t[0] + h].reshape(GRID, GRID)[4:44, 4:44])
    # same ordering as B1's inner-point list (centre of each 5x5 patch)
    d1 = make_data_b1(make_sst(), CFG, HALO, 5)
    assert np.array_equal(y[0, 0].reshape(-1), d1.a[t[0] + 1, d1.centre])


def test_crop_matches_real_inner_grid_lat_lon():
    import xarray as xr
    from sst.download import ROOT
    for r in ("kuroshio", "subtropical"):
        ph, pi = ROOT / "data" / "processed" / f"{r}_halo.nc", ROOT / "data" / "processed" / f"{r}.nc"
        if not (ph.exists() and pi.exists()):
            pytest.skip("processed data absent")
        h, i = xr.open_dataset(ph), xr.open_dataset(pi)
        lat, lon = crop(h["latitude"].values[:, None] * np.ones((1, GRID)))[:, 0], crop(h["longitude"].values[None, :] * np.ones((GRID, 1)))[0]
        assert h.sizes["latitude"] == h.sizes["longitude"] == GRID
        assert np.array_equal(lat, i["latitude"].values) and np.array_equal(lon, i["longitude"].values)
        assert np.array_equal(h["sst"].isel(time=slice(0, 5)).values[:, 4:44, 4:44], i["sst"].isel(time=slice(0, 5)).values)


# ---------------------------------------------------------------- evaluation pairs = baseline pairs
def test_eval_pairs_equal_lstm_and_b1_pairs(sst, data):
    d1 = make_data_b1(sst, CFG, HALO, 5)
    P = d1.n_points
    assert np.array_equal(data.val_issue, np.unique(d1.val_pairs // P))
    assert len(d1.val_pairs) == len(data.val_issue) * P
    assert np.array_equal(data.train_issue, np.unique(d1.train_pairs // P))
    inner = sst.isel(latitude=slice(4, 44), longitude=slice(4, 44))
    d0 = make_data(inner, CFG)
    assert np.array_equal(data.val_issue, np.unique(d0.val_pairs // d0.n_points))
    assert data.dates[data.val_issue].min() == pd.Timestamp("2016-01-01") and data.dates[data.val_issue].max() == pd.Timestamp("2019-12-01")


def test_eval_pairs_equal_frozen_baseline_pairs_real():
    import json
    from sst.b2_data import load_data_b2
    from sst.download import load_config
    cfg = load_config()
    for r in cfg["regions"]:
        npz = ROOT / "data" / "derived" / f"b1_ridge_{r}.npz"
        if not npz.exists() or not (ROOT / "data" / "processed" / f"{r}_halo.nc").exists():
            pytest.skip("real data absent")
        d = load_data_b2(r, cfg)
        frozen = np.load(npz)["issue"]                                 # the issue dates the frozen Ridge-patch predictions are for
        assert list(frozen) == [str(x.date()) for x in d.dates[d.val_issue]]
        old = json.loads((ROOT / "results" / "lstm_val.json").read_text())["regions"][r]
        assert old["n_issue_dates"] == len(d.val_issue)
        assert (old["first_issue"], old["last_issue"]) == (str(d.dates[d.val_issue[0]].date()), str(d.dates[d.val_issue[-1]].date()))
        assert d.std == old["std_deg_c"]                               # same standardisation number as the LSTM


# ---------------------------------------------------------------- determinism and overfitting
@pytest.mark.parametrize("model", ["unet", "convlstm"])
def test_same_seed_same_loss(data, model):
    d = tiny(data)
    r1 = train_run_b2(d, model, 16, 3, TCFG, max_epochs=2, verbose=False)
    r2 = train_run_b2(d, model, 16, 3, TCFG, max_epochs=2, verbose=False)
    r3 = train_run_b2(d, model, 16, 4, TCFG, max_epochs=2, verbose=False)
    assert r1["history"]["train_loss"] == r2["history"]["train_loss"] and r1["history"]["val_loss"] == r2["history"]["val_loss"]
    assert r1["history"]["train_loss"] != r3["history"]["train_loss"]


@pytest.mark.parametrize("model", ["unet", "convlstm"])
def test_tiny_batch_can_be_overfitted(data, model):
    torch.manual_seed(0)
    d = tiny(data, n_tr=4, n_va=2)
    net = build(model, 16)
    x, y = torch.from_numpy(gather_x(d, d.train_issue)), torch.from_numpy(gather_y(d, d.train_issue))
    if model == "convlstm":
        x = x[:, -6:]                                  # last 6 input days only: keeps the test fast (the recurrence is the same)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3 if model == "unet" else 1e-2)
    first = None
    for i in range(150 if model == "unet" else 120):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(crop(net(x)), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.15 * first, (first, loss.item())


def test_val_loss_matches_manual(data):
    d = tiny(data, 4, 5)
    set = torch.manual_seed(0)
    net = build("unet", 16)
    net.eval()
    with torch.no_grad():
        err = crop(net(torch.from_numpy(gather_x(d, d.val_issue)))) - torch.from_numpy(gather_y(d, d.val_issue))
    assert abs(val_loss(net, d, chunk=2) - float((err.double() ** 2).mean())) < 1e-6
