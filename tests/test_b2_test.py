import json

from sst.b2_eval_res import select_variants
from sst.b2_test_choice import RES, derive


def test_choice_follows_rule_and_matches_validation_json():
    ch = derive()
    v = json.loads((RES / "b2_val_res.json").read_text())
    for r, d in ch["regions"].items():
        for fam, c in d.items():
            s = v["variant_selection"][r][fam]
            assert c["chosen"] == (s["residual"] if s["mean_rmse_residual"] < s["mean_rmse_plain"] else s["plain"])
            assert c["C"] == {"unet": 32, "convlstm_ed": 16}[fam]
    assert select_variants(v) == v["variant_selection"]


def test_guard_refuses_without_tag(monkeypatch):
    import pytest
    from sst import b2_test
    monkeypatch.setattr(b2_test, "TAG", "no-such-tag-xyz")
    with pytest.raises(SystemExit):
        b2_test.check_guards()
