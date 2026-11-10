"""B2 validation (2016-2019 only): plain vs residual variants side by side.  python -m sst.b2_eval_res

Evaluates unet (C=32), resunet (C=32), convlstm_ed (C=16), resconvlstm_ed (C=16) on the identical validation pairs with the same code as b2_eval
(plain predictions are asserted equal to the committed results/b2_val.json). Variant choice per model and region: lowest mean (seeds 0,1,2 x leads 1,7,30)
validation RMSE. NEW files: results/b2_val_res.json, b2_val_res_report.md, b2_val_maps_res.npz, figures/figB2_curves_res.png, figB2_maps_res.png.
"""
import copy
import json

import numpy as np

from sst.b2_eval import (LABEL, RESULTS, answers, chosen, chosen_c_md, curves_md, evaluate_region, f3, figures, season_md, spatial_md)
from sst.b2_train import load_b2
from sst.download import load_config
from sst.run_val import _clean

FAMILY = {"unet": ("unet", "resunet"), "convlstm_ed": ("convlstm_ed", "resconvlstm_ed")}
TAG = "_res"


def mean_rmse(R, m):
    C = chosen(R, m)
    return float(np.mean([R["models"][m][C]["mean"][h]["overall"]["rmse"] for h in ("1", "7", "30")]))


def select_variants(res):
    """Plain vs residual per model and region; returns {region: {family: {plain, residual, chosen}}}."""
    out = {}
    for r, R in res["regions"].items():
        out[r] = {}
        for fam, (pl, rs) in FAMILY.items():
            a, b = mean_rmse(R, pl), mean_rmse(R, rs)
            out[r][fam] = {"plain": pl, "residual": rs, "mean_rmse_plain": a, "mean_rmse_residual": b, "chosen": rs if b < a else pl,
                           "rule": "lowest mean validation RMSE over seeds 0,1,2 and leads 1,7,30"}
    return out


def chosen_view(res, sel):
    """Same structure as a b2_eval result, but with each family (unet, convlstm_ed) replaced by its chosen variant, so the b2_eval sections apply unchanged."""
    v = copy.deepcopy(res)
    for r, R in v["regions"].items():
        R["models"] = {fam: res["regions"][r]["models"][sel[r][fam]["chosen"]] for fam in FAMILY}
    return v


def check_plain_equals_committed(res):
    old = json.loads((RESULTS / "b2_val.json").read_text())
    worst = 0.0
    for r, R in res["regions"].items():
        for m in FAMILY:
            C = chosen(R, m)
            for h in ("1", "7", "30"):
                for k in ("rmse", "acc"):
                    d = abs(R["models"][m][C]["mean"][h]["overall"][k] - old["regions"][r]["models"][m][C]["mean"][h]["overall"][k])
                    worst = max(worst, d)
    assert worst < 1e-6, f"plain models no longer reproduce the committed validation numbers (max diff {worst})"
    return worst


def report_res_md(res, sel, cfg):
    leads = [str(h) for h in cfg["leads_days"]]
    models = list(next(iter(res["regions"].values()))["models"])
    pick = {(r, sel[r][f]["chosen"]) for r in sel for f in sel[r]}
    L = ["# B2 plain vs residual whole-region models - validation 2016-2019 (inner 40x40, identical pairs for every method)", "",
         "RMSE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation; skill = 1 - RMSE/RMSE(reference). Deep models: mean ± sample std (ddof=1) over seeds 0,1,2. "
         "Residual = last input anomaly (day t, same for all 3 leads) + network output; same architecture, C, seeds and training settings as the plain model. "
         "C fixed from the plain-model selection (U-Net 32, ConvLSTM-ED 16). Reference rows are the frozen B1 numbers (re-checked). Variant chosen per model and region by the lowest mean validation RMSE "
         "over seeds and leads (marked ◀). Timing is not a result.", "", "## Variant selection (plain vs residual)", "",
         "| region | model | mean RMSE plain | mean RMSE residual | chosen |", "|---|---|---|---|---|"]
    for r in sel:
        for f, v in sel[r].items():
            L.append(f"| {r} | {LABEL[f]} | {v['mean_rmse_plain']:.4f} | {v['mean_rmse_residual']:.4f} | {'residual' if v['chosen'].startswith('res') else 'plain'} |")
    L += [""]
    for r, R in res["regions"].items():
        L += [f"## {r}  (issue dates {R['first_issue']}..{R['last_issue']}, n={R['n_issue_dates']})", ""]
        for h in leads:
            ref = R["reference"][h]
            L += [f"### lead {h} d", "", "| method | RMSE | ACC | skill vs Ridge-patch | skill vs anomaly persistence |", "|---|---|---|---|---|"]
            for m, lab in (("anomaly_persistence", "anomaly persistence"), ("ridge", "Ridge"), ("ridge_patch", "Ridge-patch")):
                L.append(f"| {lab} | {f3(ref[m]['overall']['rmse'])} | {f3(ref[m]['overall']['acc'])} | {1 - ref[m]['overall']['rmse'] / ref['ridge_patch']['overall']['rmse']:+.3f} | "
                         f"{1 - ref[m]['overall']['rmse'] / ref['anomaly_persistence']['overall']['rmse']:+.3f} |")
            v = R["lstm_patch"]
            L.append(f"| LSTM-patch | {f3(v['mean'][h]['overall']['rmse'], v['std'][h]['overall']['rmse'])} | {f3(v['mean'][h]['overall']['acc'], v['std'][h]['overall']['acc'])} | "
                     f"{v['mean'][h]['skill_vs_ridge_patch']['overall']:+.3f} ± {v['std'][h]['skill_vs_ridge_patch']['overall']:.3f} | "
                     f"{v['mean'][h]['skill_vs_anomaly_persistence']['overall']:+.3f} ± {v['std'][h]['skill_vs_anomaly_persistence']['overall']:.3f} |")
            for m in models:
                C = chosen(R, m)
                v = R["models"][m][C]
                mark = " ◀" if (r, m) in pick else ""
                L.append(f"| {LABEL[m]} (C={C}){mark} | {f3(v['mean'][h]['overall']['rmse'], v['std'][h]['overall']['rmse'])} | {f3(v['mean'][h]['overall']['acc'], v['std'][h]['overall']['acc'])} | "
                         f"{v['mean'][h]['skill_vs_ridge_patch']['overall']:+.3f} ± {v['std'][h]['skill_vs_ridge_patch']['overall']:.3f} | "
                         f"{v['mean'][h]['skill_vs_anomaly_persistence']['overall']:+.3f} ± {v['std'][h]['skill_vs_anomaly_persistence']['overall']:.3f} |")
            L.append("")
    L += ["## Residual vs plain (RMSE skill of residual relative to plain = 1 - RMSE_res/RMSE_plain; mean over seeds, ± std of the per-seed values)", "",
          "| region | model | " + " | ".join(f"{h} d" for h in leads) + " |", "|---|---|" + "---|" * len(leads)]
    for r, R in res["regions"].items():
        for f, (pl, rs) in FAMILY.items():
            cells = []
            for h in leads:
                a = [R["models"][pl][chosen(R, pl)]["per_seed"][str(s)][h]["overall"]["rmse"] for s in (0, 1, 2)]
                b = [R["models"][rs][chosen(R, rs)]["per_seed"][str(s)][h]["overall"]["rmse"] for s in (0, 1, 2)]
                sk = [1 - y / x for x, y in zip(a, b)]
                cells.append(f"{np.mean(sk):+.3f} ± {np.std(sk, ddof=1):.3f}")
            L.append(f"| {r} | {LABEL[f]} | " + " | ".join(cells) + " |")
    L += [""] + chosen_c_md(json.loads((RESULTS / "b2_val.json").read_text()))[0:1] + ["(C selection of the plain models: see the table in results/b2_val_report.md, Correction 2026-11-10. Used here: U-Net 32, ConvLSTM-ED 16.)", ""]
    L += curves_md(res, cfg) + season_md(res, cfg) + spatial_md(res, cfg)
    view = chosen_view(res, sel)
    L += ["", "## Answers (generated from the numbers above by code; whole-region models = the CHOSEN variant per model and region; read with the std)", ""]
    L += [f"- **Chosen variants:** " + "; ".join(f"{r} {LABEL[f]}: {'residual' if v['chosen'].startswith('res') else 'plain'}" for r in sel for f, v in sel[r].items())]
    L += answers(view, cfg)
    return "\n".join(L)


def main():
    cfg = load_config()
    b2 = load_b2()
    models = {"unet": [32], "resunet": [b2["residual"]["resunet"]["C"]], "convlstm_ed": [16], "resconvlstm_ed": [b2["residual"]["resconvlstm_ed"]["C"]]}
    ref_all = json.loads((RESULTS / "b1_val.json").read_text())
    res = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["val"]}, "leads_days": cfg["leads_days"], "regions": {}}
    maps = {}
    for r in cfg["regions"]:
        res["regions"][r], maps[r] = evaluate_region(r, cfg, models, ref_all)
    worst = check_plain_equals_committed(res)
    print(f"plain models reproduce results/b2_val.json (max |diff| in RMSE/ACC {worst:.2e})")
    sel = select_variants(res)
    res["variant_selection"] = sel
    (RESULTS / f"b2_val{TAG}.json").write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    np.savez_compressed(RESULTS / f"b2_val_maps{TAG}.npz", **{f"{r}__{k}": v for r, mp in maps.items() for k, v in mp.items()})
    (RESULTS / f"b2_val{TAG}_report.md").write_text(report_res_md(res, sel, cfg))
    figures(res, maps, cfg, TAG)
    print(f"wrote results/b2_val{TAG}.json, b2_val{TAG}_report.md, b2_val_maps{TAG}.npz, figures/figB2_*{TAG}.png")


if __name__ == "__main__":
    main()
