"""Synthetic tests for the LSTM experiment (no real data, no network, fixed seeds).

Tests that need torch are skipped cleanly if torch is not installed."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sst.lstm_data import SEQ_LEN, day_features, gather, make_data, valid_mask

CFG = {"split": {"train": ["1982-01-01", "2015-12-31"], "val": ["2016-01-01", "2019-12-31"]},
       "leads_days": [1, 7, 30], "climatology": {"years": "train", "smoothing": "harmonics3", "feb29": "use_feb28"}}
DATES = pd.date_range("1982-01-01", "2019-12-31")


def make_sst(seed=0):
    rng = np.random.default_rng(seed)
    doy = DATES.day_of_year.values[:, None, None]
    ar = np.zeros((len(DATES), 2, 3))
    for i in range(1, len(DATES)):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(size=(2, 3))
    vals = 20 + 5 * np.sin(2 * np.pi * doy / 365) + ar
    return xr.DataArray(vals, dims=("time", "latitude", "longitude"),
                        coords={"time": DATES, "latitude": [20.0, 20.25], "longitude": [150.0, 150.25, 150.5]}, name="sst")


@pytest.fixture(scope="module")
def sst():
    return make_sst()


@pytest.fixture(scope="module")
def data(sst):
    return make_data(sst, CFG)


def test_inputs_use_only_days_up_to_t(data):
    flat = data.train_pairs[[100, 5000]]
    x0, _ = gather(data, flat)
    t = flat // data.n_points
    p = flat % data.n_points
    # samples are at different dates; check each separately so later samples do not hide earlier days
    for k in range(2):
        a2 = data.a.copy()
        a2[t[k] + 1:] = 123.0
        x1, _ = gather(type(data)(**{**data.__dict__, "a": a2}), flat[k:k + 1])
        assert np.array_equal(x1[0], x0[k])
    # a change AT or before t does change the input (the test can fail)
    a3 = data.a.copy()
    a3[t[0] - 3, p[0]] += 1.0
    x3, _ = gather(type(data)(**{**data.__dict__, "a": a3}), flat[:1])
    assert not np.array_equal(x3[0], x0[0])
    # the day-of-year channels belong to days t-29..t
    d = data.dates[t[0] - SEQ_LEN + 1:t[0] + 1]
    assert np.allclose(x0[0][:, 1:], day_features(d))


def test_no_training_target_after_2015(data):
    t = data.train_pairs // data.n_points
    assert data.dates[t].max() + pd.Timedelta(days=30) <= pd.Timestamp("2015-12-31")
    assert data.dates[t].min() - pd.Timedelta(days=SEQ_LEN - 1) >= pd.Timestamp("1982-01-01")
    x, y = gather(data, data.train_pairs[:50])
    p = data.train_pairs[:50] % data.n_points
    assert np.array_equal(y[:, 2], data.a[t[:50] + 30, p])


def test_validation_samples(data):
    t = data.val_pairs // data.n_points
    assert data.dates[t].min() == pd.Timestamp("2016-01-01")        # history reaches back into 2015 (allowed)
    assert data.dates[t].max() == pd.Timestamp("2019-12-01")        # t+30 must still be in 2019
    assert len(data.val_pairs) == len(np.unique(t)) * data.n_points


def test_changing_validation_data_does_not_change_training_set(sst, data):
    sst2 = sst.copy(deep=True)
    after = sst2.time > pd.Timestamp("2015-12-31")
    sst2.values[after.values] += np.random.default_rng(1).normal(5, 3, size=sst2.values[after.values].shape)
    d2 = make_data(sst2, CFG)
    assert np.array_equal(d2.train_pairs, data.train_pairs)
    assert d2.std == data.std
    assert np.array_equal(d2.clim.values, data.clim.values)
    sel = data.train_pairs[::997]
    xa, ya = gather(data, sel)
    xb, yb = gather(d2, sel)
    assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    # ...while the validation inputs DO change (the test can fail)
    assert not np.array_equal(gather(data, data.val_pairs[:5])[0], gather(d2, data.val_pairs[:5])[0])


def test_test_period_cannot_be_loaded(sst):
    longer = xr.concat([sst, sst.assign_coords(time=sst.time + pd.Timedelta(days=len(DATES)))], dim="time")
    with pytest.raises(ValueError):
        make_data(longer, CFG)


def test_valid_mask_excludes_missing_inputs_and_targets():
    a = np.zeros((100, 2), dtype=np.float32)
    a[50, 0] = np.nan
    m = valid_mask(a, (1, 7, 30))
    assert not m[:29].any()                 # not enough history
    assert not m[70:].any()                 # t+30 beyond the end
    assert not m[50:80, 0].any()            # NaN at day 50 poisons windows 50..79 for point 0
    assert not m[43, 0] and not m[49, 0]    # NaN is the target t+7 / t+1 of day 43 / 49
    assert m[45, 1] and m[69, 1]            # other point is fine
    assert m[40, 0]                         # window 11..40 and targets 41,47,70 are fine


def test_shapes(data):
    torch = pytest.importorskip("torch")
    from sst.lstm_model import SSTLSTM
    x, y = gather(data, data.train_pairs[:7])
    assert x.shape == (7, 30, 3) and y.shape == (7, 3) and x.dtype == np.float32
    assert SSTLSTM(8)(torch.from_numpy(x)).shape == (7, 3)


LCFG = {"lr": 1e-3, "batch_size": 64, "max_epochs": 1, "patience": 5, "num_threads": 1, "samples_per_epoch": 512}


def test_same_seed_gives_identical_loss(data):
    pytest.importorskip("torch")
    from sst.lstm_train import train_run
    r1 = train_run(data, 4, 0, LCFG, verbose=False)
    r2 = train_run(data, 4, 0, LCFG, verbose=False)
    r3 = train_run(data, 4, 1, LCFG, verbose=False)
    assert r1["history"]["train_loss"] == r2["history"]["train_loss"]
    assert r1["history"]["val_loss"] == r2["history"]["val_loss"]
    assert r1["history"]["val_loss"] != r3["history"]["val_loss"]


def test_model_can_overfit_a_tiny_batch(data):
    torch = pytest.importorskip("torch")
    from torch import nn
    from sst.lstm_model import SSTLSTM
    from sst.lstm_train import set_seed
    set_seed(0)
    x, y = gather(data, data.train_pairs[::50000][:32])
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    model, loss_fn = SSTLSTM(32), nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = float(loss_fn(model(x), y))
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    assert float(loss) < 0.05 * first and float(loss) < 0.02


# ---- test-period plumbing (synthetic; no real data) ----
def _sst_to_2025():
    dates = pd.date_range("1982-01-01", "2025-12-31")
    rng = np.random.default_rng(1)
    vals = 20 + 5 * np.sin(2 * np.pi * dates.day_of_year.values[:, None, None] / 365) + rng.normal(size=(len(dates), 2, 3))
    return xr.DataArray(vals, dims=("time", "latitude", "longitude"),
                        coords={"time": dates, "latitude": [20.0, 20.25], "longitude": [150.0, 150.25, 150.5]}, name="sst")


def test_make_data_rejects_data_past_the_period_end():
    cfg = dict(CFG, split=dict(CFG["split"], test=["2020-01-01", "2025-12-31"]))
    with pytest.raises(ValueError):
        make_data(_sst_to_2025(), cfg)                      # default period "val": 2020+ must not be accepted
    d = make_data(_sst_to_2025(), cfg, period="test")       # allowed only when asked for explicitly
    assert d.dates.max() == pd.Timestamp("2025-12-31")


def test_test_period_keeps_training_statistics_and_inputs_only_past():
    pytest.importorskip("torch")
    from sst.lstm_data import gather_inputs
    cfg = dict(CFG, split=dict(CFG["split"], test=["2020-01-01", "2025-12-31"]))
    full = _sst_to_2025()
    d_val = make_data(full.sel(time=slice(None, "2019-12-31")), cfg)
    d_test = make_data(full, cfg, period="test")
    assert d_val.std == d_test.std                          # standardisation = training years only
    # inputs of an issue date in 2024 are unchanged if everything after it is replaced by garbage
    flat = np.array([pd.Timestamp("2024-06-01").to_datetime64()])
    t = int(np.searchsorted(d_test.dates.values, flat[0]))
    x1 = gather_inputs(d_test, np.array([t * d_test.n_points + 2]))
    d_test.a[t + 1:] = 99.0
    x2 = gather_inputs(d_test, np.array([t * d_test.n_points + 2]))
    assert np.array_equal(x1, x2)


def test_lstm_test_guards_refuse(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import sst.lstm_test as T
    monkeypatch.setattr(T, "_git", lambda *a: type("R", (), {"returncode": 1, "stdout": ""})())
    with pytest.raises(SystemExit):
        T.check_guards()                                    # tag missing -> refuse
