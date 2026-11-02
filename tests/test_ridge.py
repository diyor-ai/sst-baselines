"""Synthetic tests for the Ridge baseline (no real data, fixed seed)."""
import numpy as np
import pandas as pd

from sst.ridge import (MEAN_WIN, N_FEAT, N_LAGS, fit_ridge, make_features, predict_ridge, coef_original_units,
                       feature_ready_days, _pairs)

T0 = pd.Timestamp("2000-01-01")


def _series(n=400, p=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_features_do_not_depend_on_future_values():
    a = _series()
    t = pd.Timestamp("2000-09-01")
    f0 = make_features(a, T0, pd.DatetimeIndex([t]))
    b = a.copy()
    b[(t - T0).days + 1:] += 1000.0  # change everything after the issue date
    f1 = make_features(b, T0, pd.DatetimeIndex([t]))
    assert np.array_equal(f0, f1)
    # and a change AT or before t does change them (the test can fail)
    c = a.copy()
    c[(t - T0).days - 3] += 1.0
    assert not np.array_equal(f0, make_features(c, T0, pd.DatetimeIndex([t])))


def test_feature_definitions():
    a = _series()
    t = pd.Timestamp("2000-09-01")
    d = (t - T0).days
    f = make_features(a, T0, pd.DatetimeIndex([t]))[0]
    assert f.shape == (3, N_FEAT)
    for k in range(N_LAGS):
        assert np.allclose(f[:, k], a[d - k])
    assert np.allclose(f[:, N_LAGS], a[d - MEAN_WIN + 1:d + 1].mean(axis=0))


def test_short_history_and_missing_day_give_nan_features():
    a = _series()
    early = make_features(a, T0, pd.DatetimeIndex([T0 + pd.Timedelta(days=10)]))
    assert np.isnan(early[..., N_LAGS]).all()  # 30-day mean needs 30 days of history
    assert np.isnan(make_features(a, T0, pd.DatetimeIndex([T0 - pd.Timedelta(days=5)]))).all()
    a[100] = np.nan
    f = make_features(a, T0, pd.DatetimeIndex([T0 + pd.Timedelta(days=105), T0 + pd.Timedelta(days=200)]))
    assert np.isnan(f[0]).any() and not np.isnan(f[1]).any()


def test_standardisation_uses_training_statistics_only():
    rng = np.random.default_rng(1)
    Xtr, ytr = rng.normal(2.0, 3.0, size=(500, 2, N_FEAT)), rng.normal(size=(500, 2))
    m = fit_ridge(Xtr, ytr, [0.1])
    assert np.allclose(m["mu"], Xtr.mean(0)) and np.allclose(m["sd"], Xtr.std(0))
    # predicting on very different data must not change the stored statistics or the weights
    mu, sd, coef = m["mu"].copy(), m["sd"].copy(), m["coef"].copy()
    Xva = rng.normal(50.0, 20.0, size=(30, 2, N_FEAT))
    predict_ridge(m, Xva)
    assert np.array_equal(mu, m["mu"]) and np.array_equal(sd, m["sd"]) and np.array_equal(coef, m["coef"])
    # shifting validation data shifts the prediction exactly through the TRAINING mean/std
    p0 = predict_ridge(m, Xva)
    p1 = predict_ridge(m, Xva + 1.0)
    shift = ((1.0 / m["sd"]) * m["coef"][0]).sum(-1)
    assert np.allclose(p1[0] - p0[0], shift[None, :])


def test_ar1_coefficient_recovered_at_lead_1():
    rng = np.random.default_rng(42)
    phi, n = 0.8, 8000
    x = np.zeros((n, 1))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    issue = pd.date_range(T0 + pd.Timedelta(days=40), periods=n - 41)
    X = make_features(x, T0, issue)
    y = x[(issue - T0).days + 1]
    m = fit_ridge(X, y, [1e-6])
    w = coef_original_units(m, 0)[0]
    assert abs(w[0] - phi) < 0.05, w          # weight on the lag-0 anomaly
    assert np.abs(w[1:N_LAGS]).max() < 0.05   # older lags carry no extra information in an AR(1)
    pred = predict_ridge(m, X)[0, :, 0]
    assert abs(np.corrcoef(pred, phi * x[(issue - T0).days, 0])[0, 1] - 1) < 0.01


def test_larger_alpha_shrinks_weights():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(300, 1, N_FEAT)), rng.normal(size=(300, 1))
    m = fit_ridge(X, y, [1e-3, 1e3])
    assert np.abs(m["coef"][1]).sum() < 0.01 * np.abs(m["coef"][0]).sum()


def test_pairs_skip_incomplete_history_and_missing_target():
    days = pd.date_range("2000-01-01", "2000-06-30").delete([60, 100])  # two missing days
    issue, target, c = _pairs(days, "2000-02-01", "2000-06-30", 7)
    assert c["used"] + c["skipped"] == c["expected"] and c["skipped"] > 0
    assert c["skipped_target_missing"] == 2
    assert (issue + pd.Timedelta(days=7) == target).all()
    assert issue.isin(feature_ready_days(days)).all()
