"""Markdown tables from results/b2_test.json -> results/b2_test_report.md (pure reporting; no model is run).  python -m sst.b2_test_report"""
import json

from sst.download import ROOT

RES = ROOT / "results"
SEAS = ["DJF", "MAM", "JJA", "SON"]
FAM = {"unet": "U-Net", "convlstm_ed": "ConvLSTM-ED"}


def pm(m, s, f="{:.3f}"):
    return f.format(m) + " ± " + f.format(s)


def cell(blk, key, std=None):
    v = blk["overall"][key] if key in blk["overall"] else None
    return v


def report(res):
    leads = [str(h) for h in res["leads_days"]]
    L = ["# B2 whole-region models - TEST 2020-2025 (inner 40x40, the frozen baselines' pairs)", "",
         f"Protocol: docs/b2_test_protocol.md. Commit {res['run']['commit'][:7]}, tag {res['run']['tag']}. RMSE/MAE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation (spatial ACC in the JSON). "
         "Skill = 1 - RMSE/RMSE(reference). Deep models (LSTM-patch, U-Net, ConvLSTM-ED): mean ± sample std (ddof=1) over seeds 0,1,2 of per-seed scores. Variants chosen on validation: "
         + "; ".join(f"{r} {FAM[f]}: {c['chosen']}" for r, d in res["choice"]["regions"].items() for f, c in d.items()) + ".", ""]
    for r, R in res["regions"].items():
        L += [f"## {r}", ""]
        for h in leads:
            ref = R["reference"][h]
            apr = ref["anomaly_persistence"]["overall"]["rmse"]
            L += [f"### lead {h} d  (n pairs {R['pairs'][h]['n_pairs']})", "",
                  "| method | RMSE | MAE | ACC | skill vs persistence | skill vs Ridge | skill vs Ridge-patch | skill vs LSTM-patch |", "|---|---|---|---|---|---|---|---|"]
            for m, lab in (("anomaly_persistence", "anomaly persistence"), ("ridge", "Ridge"), ("ridge_patch", "Ridge-patch")):
                o = ref[m]["overall"]
                L.append(f"| {lab} | {o['rmse']:.3f} | {o['mae']:.3f} | {o['acc']:.3f} | {1 - o['rmse'] / apr:+.3f} | {1 - o['rmse'] / ref['ridge']['overall']['rmse']:+.3f} | "
                         f"{1 - o['rmse'] / ref['ridge_patch']['overall']['rmse']:+.3f} | {1 - o['rmse'] / R['lstm_patch']['mean'][h]['overall']['rmse']:+.3f} |")
            v = R["lstm_patch"]
            mm, ss = v["mean"][h], v["std"][h]
            L.append(f"| LSTM-patch | {pm(mm['overall']['rmse'], ss['overall']['rmse'])} | {pm(mm['overall']['mae'], ss['overall']['mae'])} | {pm(mm['overall']['acc'], ss['overall']['acc'])} | "
                     f"{pm(mm['skill_vs_anomaly_persistence']['overall'], ss['skill_vs_anomaly_persistence']['overall'], '{:+.3f}')} | {pm(mm['skill_vs_ridge']['overall'], ss['skill_vs_ridge']['overall'], '{:+.3f}')} | "
                     f"{pm(mm['skill_vs_ridge_patch']['overall'], ss['skill_vs_ridge_patch']['overall'], '{:+.3f}')} | - |")
            for f, lab in FAM.items():
                M = R["models"][f]
                mm, ss = M["mean"][h], M["std"][h]
                L.append(f"| {lab} ({M['variant']}, C={M['C']}) | {pm(mm['overall']['rmse'], ss['overall']['rmse'])} | {pm(mm['overall']['mae'], ss['overall']['mae'])} | {pm(mm['overall']['acc'], ss['overall']['acc'])} | "
                         + " | ".join(pm(mm[k]["overall"], ss[k]["overall"], "{:+.3f}") for k in ("skill_vs_anomaly_persistence", "skill_vs_ridge", "skill_vs_ridge_patch", "skill_vs_lstm_patch")) + " |")
            L.append("")
    L += ["## RMSE by season (deg C; deep models mean ± std over seeds)", "", "| region | lead | method | " + " | ".join(SEAS) + " |", "|---|---|---|" + "---|" * 4]
    for r, R in res["regions"].items():
        for h in leads:
            ref = R["reference"][h]
            for m, lab in (("anomaly_persistence", "anomaly persistence"), ("ridge_patch", "Ridge-patch")):
                L.append(f"| {r} | {h} | {lab} | " + " | ".join(f"{ref[m]['by_season'][s]['rmse']:.3f}" for s in SEAS) + " |")
            v = R["lstm_patch"]
            L.append(f"| {r} | {h} | LSTM-patch | " + " | ".join(pm(v["mean"][h]["by_season"][s]["rmse"], v["std"][h]["by_season"][s]["rmse"]) for s in SEAS) + " |")
            for f, lab in FAM.items():
                M = R["models"][f]
                L.append(f"| {r} | {h} | {lab} | " + " | ".join(pm(M["mean"][h]["by_season"][s]["rmse"], M["std"][h]["by_season"][s]["rmse"]) for s in SEAS) + " |")
    L += ["", "## Skill vs Ridge-patch by season (mean over seeds)", "", "| region | lead | model | " + " | ".join(SEAS) + " |", "|---|---|---|" + "---|" * 4]
    for r, R in res["regions"].items():
        for h in leads:
            for f, lab in FAM.items():
                d = R["models"][f]["mean"][h]["skill_vs_ridge_patch"]["by_season"]
                L.append(f"| {r} | {h} | {lab} | " + " | ".join(f"{d[s]:+.3f}" for s in SEAS) + " |")
    L += ["", "## Generated statements (numbers from the table above)", ""]
    for f, lab in FAM.items():
        for ref in ("ridge_patch", "lstm_patch"):
            cells = [(r, h, R["models"][f]["mean"][h][f"skill_vs_{ref}"]["overall"], R["models"][f]["std"][h][f"skill_vs_{ref}"]["overall"]) for r, R in res["regions"].items() for h in leads]
            L.append(f"- **{lab} vs {ref.replace('_', '-').capitalize()}:** positive in {sum(c[2] > 0 for c in cells)}/{len(cells)} region-lead cells (by more than one seed-std in "
                     f"{sum(c[2] - c[3] > 0 for c in cells)}): " + "; ".join(f"{r} {h} d {m:+.3f} ± {s:.3f}" for r, h, m, s in cells))
    for r, R in res["regions"].items():
        L.append(f"- **U-Net vs ConvLSTM-ED, {r} (mean RMSE):** " + "; ".join(f"{h} d {R['models']['unet']['mean'][h]['overall']['rmse']:.3f} / {R['models']['convlstm_ed']['mean'][h]['overall']['rmse']:.3f}" for h in leads))
    return "\n".join(L) + "\n"


def main():
    res = json.loads((RES / "b2_test.json").read_text())
    (RES / "b2_test_report.md").write_text(report(res))
    print("wrote results/b2_test_report.md")


if __name__ == "__main__":
    main()
