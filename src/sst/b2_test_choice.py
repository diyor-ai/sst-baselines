"""B2 test choice (decided by code from VALIDATION numbers only): plain vs residual per region and family -> results/b2_test_choice.json.

  python -m sst.b2_test_choice          # writes results/b2_test_choice.json (refuses to overwrite)
Rule (docs/b2_test_protocol.md s.1): lowest mean validation RMSE over seeds 0,1,2 and leads 1,7,30; ties -> plain. C = the plain-model selection (b2_val.json).
Source: results/b2_val_res.json ('variant_selection').
"""
import json

from sst.download import ROOT

RES = ROOT / "results"
OUT = RES / "b2_test_choice.json"
FAMILIES = {"unet": "resunet", "convlstm_ed": "resconvlstm_ed"}


def derive():
    v = json.loads((RES / "b2_val_res.json").read_text())
    plain = json.loads((RES / "b2_val.json").read_text())
    out = {"rule": "lowest mean validation RMSE over seeds 0,1,2 and leads 1,7,30 (plain vs residual; ties -> plain)",
           "source": "results/b2_val_res.json", "regions": {}}
    for r, sel in v["variant_selection"].items():
        out["regions"][r] = {}
        for fam, resname in FAMILIES.items():
            s = sel[fam]
            assert s["plain"] == fam and s["residual"] == resname
            chosen = resname if s["mean_rmse_residual"] < s["mean_rmse_plain"] else fam
            C = int(plain["regions"][r]["models"][fam]["selection"]["chosen_C"])
            out["regions"][r][fam] = {"chosen": chosen, "C": C, "mean_rmse_plain": s["mean_rmse_plain"], "mean_rmse_residual": s["mean_rmse_residual"]}
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists; not overwriting")
    out = derive()
    OUT.write_text(json.dumps(out, indent=1))
    for r, d in out["regions"].items():
        for f, c in d.items():
            print(f"{r:12s} {f:12s} plain {c['mean_rmse_plain']:.4f}  residual {c['mean_rmse_residual']:.4f}  -> {c['chosen']} (C={c['C']})")


if __name__ == "__main__":
    main()
