"""Tests for the guarded B1 test runner (synthetic data; real files only where stated and skipped if absent)."""
import copy

import numpy as np
import pandas as pd
import pytest

import sst.b1_test as bt
from sst.b1_data import inner_block, make_data_b1
from sst.b1_ridge import run_region
from sst.climatology import build_climatology
from test_b1 import CFG, HALO, NI, SIZE, make_sst

# the dry-run situation: the "test" split is the validation years and the data end there
TCFG = {**CFG, "split": {**CFG["split"], "test": CFG["split"]["val"]},
        "ridge": {"alphas": {"x": {"1": 0.1, "7": 1.0, "30": 1.0}}}}
B1 = {"halo": {"width_points": HALO}, "patch": {"size": SIZE}}


@pytest.fixture(scope="module")
def sst_h():
    return make_sst(NI, HALO)


class FakeGit:
    def __init__(self, tag=True, same=True, dirty=False):
        self.tag, self.same, self.dirty = tag, same, dirty

    def __call__(self, *a):
        class R:
            pass
        r = R()
        r.returncode, r.stdout = 0, ""
        if a[0] == "rev-parse" and a[1] == "-q":
            r.returncode = 0 if self.tag else 1
        elif a[0] == "rev-parse" and a[1] == "HEAD":
            r.stdout = "aaa\n"
        elif a[0] == "rev-parse":
            r.stdout = "aaa\n" if self.same else "bbb\n"
        elif a[0] == "status":
            r.stdout = " M file\n" if self.dirty else ""
        return r


def test_guards_refuse(monkeypatch, tmp_path):
    monkeypatch.setattr(bt, "OUT", tmp_path / "b1_test.json")
    monkeypatch.setattr(bt, "_git", FakeGit())
    bt.check_guards()                                           # all fine: no exception (the test can pass)
    for kw in ({"tag": False}, {"same": False}, {"dirty": True}):
        monkeypatch.setattr(bt, "_git", FakeGit(**kw))
        with pytest.raises(SystemExit):
            bt.check_guards()
    monkeypatch.setattr(bt, "_git", FakeGit())
    (tmp_path / "b1_test.json").write_text("{}")
    with pytest.raises(SystemExit):
        bt.check_guards()                                       # second run refused


def test_dry_run_cannot_overwrite_the_test_result(monkeypatch):
    monkeypatch.setattr("sys.argv", ["b1_test", "--dry-run", str(bt.OUT)])
    with pytest.raises(SystemExit):
        bt.main()


def test_provenance_mismatch_is_refused():
    man = {"code_sha256_at_manifest": {"config.yaml": "0" * 64}, "data_sha256": {}}
    with pytest.raises(SystemExit):
        bt.verify_provenance(man)
    ok = {"code_sha256_at_manifest": {"config.yaml": bt.file_sha("config.yaml")}, "data_sha256": {}}
    bt.verify_provenance(ok)


def test_wrong_weights_hash_is_refused():
    from sst.download import ROOT
    if not (ROOT / "models" / f"b1_kuroshio_patch_H64_s0.pt").exists():
        pytest.skip("checkpoints not present")
    man = {"lstm_patch_runs": {"b1_kuroshio_patch_H64_s0": {"weights_sha256": "0" * 64}}}
    with pytest.raises(SystemExit):
        bt.load_checked_lstm("kuroshio", man, 27, [0])
    real = bt.load_manifest()
    models, hashes = bt.load_checked_lstm("kuroshio", real, 27, [0])
    assert list(models) == [0] and len(hashes) == 1             # the real hash passes


def test_fixed_alpha_ridge_equals_the_validation_ridge_and_has_no_look_ahead(sst_h):
    res, preds, issue = run_region("x", CFG, B1, models=(("ridge_doy", 0, True), ("ridge_patch", 2, True)), log=lambda *_: None, sst=sst_h)
    clim = make_data_b1(sst_h, CFG, HALO, SIZE).clim
    for h in CFG["leads_days"]:
        alphas = {m: res[m][str(h)]["alpha"] for m in ("ridge_doy", "ridge_patch")}
        got = bt.ridge_fixed_alpha(sst_h, clim, CFG, HALO, issue, h, alphas)
        for m in alphas:
            assert np.allclose(got[m], preds[m][str(h)], atol=1e-9)
    # no look-ahead: everything after the cutoff may change without affecting forecasts issued before it
    cutoff = pd.Timestamp("2018-06-01")
    s2 = sst_h.copy(deep=True)
    s2.values[(s2.time > cutoff).values] += 50.0
    iss = pd.DatetimeIndex(["2017-03-01", "2018-05-20"])
    a = {"ridge_doy": 0.1, "ridge_patch": 1.0}
    ga = bt.ridge_fixed_alpha(sst_h, clim, CFG, HALO, iss, 7, a)
    gb = bt.ridge_fixed_alpha(s2, clim, CFG, HALO, iss, 7, a)
    for m in a:
        assert np.allclose(ga[m], gb[m])
    later = bt.ridge_fixed_alpha(s2, clim, CFG, HALO, pd.DatetimeIndex(["2018-09-01"]), 7, a)       # the check can fail
    assert not np.allclose(later["ridge_patch"], bt.ridge_fixed_alpha(sst_h, clim, CFG, HALO, pd.DatetimeIndex(["2018-09-01"]), 7, a)["ridge_patch"])


def test_training_rows_of_fixed_alpha_ridge_do_not_depend_on_later_data(sst_h):
    """Changing validation/test-period data must not change the fitted model: forecasts for an issue date in 2016 that only
    reads <= 2016 data are identical whatever happens after 2016-03-01."""
    clim = make_data_b1(sst_h, CFG, HALO, SIZE).clim
    s2 = sst_h.copy(deep=True)
    s2.values[(s2.time > pd.Timestamp("2016-03-01")).values] *= 2.0
    iss = pd.DatetimeIndex(["2016-02-15"])
    a = {"ridge_doy": 0.1, "ridge_patch": 1.0}
    ga, gb = bt.ridge_fixed_alpha(sst_h, clim, CFG, HALO, iss, 1, a), bt.ridge_fixed_alpha(s2, clim, CFG, HALO, iss, 1, a)
    assert all(np.allclose(ga[m], gb[m]) for m in a)


def _models(n=3):
    torch = pytest.importorskip("torch")
    from sst.lstm_model import SSTLSTM
    out = {}
    for s in range(n):
        torch.manual_seed(s)
        out[s] = SSTLSTM(4, n_in=27).eval()
    return out


def test_evaluate_region_smoke_and_frozen_baseline_check(sst_h):
    models = _models()
    inner = sst_h.isel(latitude=slice(HALO, HALO + NI[0]), longitude=slice(HALO, HALO + NI[1]))
    alphas = {"ridge_doy": {1: 0.1, 7: 0.1, 30: 0.1}, "ridge_patch": {1: 1.0, 7: 1.0, 30: 1.0}}
    out = bt.evaluate_region("x", TCFG, B1, sst_h, inner, models, alphas, None, period="test")
    for h in ("1", "7", "30"):
        ref = out["reference"][h]
        assert set(ref) == {"anomaly_persistence", "ridge", "ridge_doy", "ridge_patch"}
        assert ref["ridge_patch"]["skill_vs_ridge"]["overall"] == pytest.approx(1 - ref["ridge_patch"]["overall"]["rmse"] / ref["ridge"]["overall"]["rmse"])
        assert set(out["lstm_patch"]["per_seed"]) == {"0", "1", "2"} and out["lstm_patch"]["std"][h]["overall"]["rmse"] > 0
        assert set(ref["ridge_patch"]["by_season"]) == {"DJF", "MAM", "JJA", "SON"}
        assert out["pairs"][h]["n_pairs"] + out["pairs"][h]["skipped"] == out["pairs"][h]["expected"]
    # a "frozen" file that does not match must abort the run (the check can fail)
    frozen = {"regions": {"x": {h: {"methods": {m: {"overall": {"rmse": out["reference"][h][m]["overall"]["rmse"],
                                                                "acc": out["reference"][h][m]["overall"]["acc"]}}
                                                for m in ("anomaly_persistence", "ridge")}} for h in ("1", "7", "30")}}}
    bt.evaluate_region("x", TCFG, B1, sst_h, inner, models, alphas, frozen, period="test")       # matches: passes
    bad = copy.deepcopy(frozen)
    bad["regions"]["x"]["7"]["methods"]["ridge"]["overall"]["rmse"] += 1e-6
    with pytest.raises(AssertionError):
        bt.evaluate_region("x", TCFG, B1, sst_h, inner, models, alphas, bad, period="test")


def test_evaluate_region_refuses_data_beyond_the_period(sst_h):
    import xarray as xr
    longer = xr.concat([sst_h, sst_h.assign_coords(time=sst_h.time + pd.Timedelta(days=len(sst_h.time)))], dim="time")
    inner = longer.isel(latitude=slice(HALO, HALO + NI[0]), longitude=slice(HALO, HALO + NI[1]))
    with pytest.raises(ValueError):
        bt.evaluate_region("x", TCFG, B1, longer, inner, _models(1), {"ridge_doy": {1: .1, 7: .1, 30: .1}, "ridge_patch": {1: 1., 7: 1., 30: 1.}},
                           None, period="test")
