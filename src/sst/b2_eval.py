"""B2 validation report (2016-2019 only): anomaly persistence, Ridge, Ridge-patch, LSTM-patch (frozen B1 numbers, re-checked), U-Net, ConvLSTM.

  python -m sst.b2_eval --models unet --tag _unet                 # U-Net part   -> results/b2_val_unet.json, b2_val_unet_report.md, ...
  python -m sst.b2_eval --models unet,convlstm_ed --tag ""        # full report  -> results/b2_val.json, b2_val_report.md, figures/figB2_*.png

All methods are scored on IDENTICAL pairs (the LSTM validation issue dates; target = issue + lead), inner 40x40, existing metric code.
Reference numbers (anomaly persistence, Ridge, Ridge-patch, LSTM-patch) come from results/b1_val.json (frozen, read only); the Ridge-patch and anomaly-persistence
scores are re-computed here from the stored predictions / data and asserted equal to b1_val.json (<1e-9). Only data up to the validation end is loaded.
Spatial check: per-point temporal RMSE of each deep model vs Ridge-patch; edge = outer 4 rings of the 40x40, centre = inner 32x32 (cos-lat weighted MSE).
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from sst.b1_ridge import DERIVED, as_da
from sst.b2_data import load_data_b2
from sst.b2_models import build
from sst.b2_train import RUNS, predict_b2
from sst.baselines import anomaly_persistence
from sst.climatology import clim_at
from sst.download import ROOT, load_config
from sst.lstm_data import make_data
from sst.lstm_eval import add_skill, agg, score_block
from sst.lstm_train import load_sst
from sst.metrics import lat_weights, season_labels
from sst.run_val import _clean

RESULTS = ROOT / "results"
SEASONS = ["DJF", "MAM", "JJA", "SON"]
LABEL = {"unet": "U-Net", "convlstm_ed": "ConvLSTM"}
EDGE = 4


def zone_masks(n=40, edge=EDGE):
    m = np.zeros((n, n), bool)
    m[edge:n - edge, edge:n - edge] = True
    return ~m, m                                   # edge ring, centre


def zone_rmse(mse_map, lat):
    """cos-lat weighted RMSE of a (40,40) MSE map over the edge ring and the centre."""
    w = np.broadcast_to(lat_weights(lat)[:, None], mse_map.shape)
    out = {}
    for name, m in zip(("edge", "centre"), zone_masks(mse_map.shape[0])):
        out[name] = float(np.sqrt((mse_map[m] * w[m]).sum() / w[m].sum()))
    return out


def evaluate_region(region, cfg, models, ref_all):
    sst_i = load_sst(region, cfg)
    old = make_data(sst_i, cfg)
    d2 = load_data_b2(region, cfg)
    P = old.n_points
    iss_idx = np.unique(old.val_pairs // P)
    assert np.array_equal(iss_idx, d2.val_issue) and old.dates.equals(d2.dates) and old.std == d2.std
    issue = d2.dates[d2.val_issue]
    leads = list(cfg["leads_days"])
    lat = sst_i["latitude"].values
    npz = np.load(DERIVED / f"b1_ridge_{region}.npz")
    assert list(npz["issue"]) == [str(d.date()) for d in issue]
    ref_json = ref_all["regions"][region]
    ref, obs_all, rp_mse = {}, {}, {}
    for h in leads:
        tgt = issue + pd.Timedelta(days=h)
        obs, ct, seas = sst_i.sel(time=tgt), clim_at(old.clim, tgt), season_labels(tgt)
        obs_all[h] = (obs, ct, seas)
        f_rp = ct.values + npz[f"ridge_patch_h{h}"]
        blocks = {"anomaly_persistence": score_block(anomaly_persistence(sst_i, old.clim, issue, tgt), obs, ct, seas),
                  "ridge_patch": score_block(as_da(f_rp, sst_i, tgt), obs, ct, seas)}
        for m in blocks:                                              # re-computed == frozen B1 numbers
            for k in ("rmse", "acc"):
                assert abs(blocks[m]["overall"][k] - ref_json["reference"][str(h)][m]["overall"][k]) < 1e-9, (region, h, m, k)
        ref[h] = {**blocks, "ridge": ref_json["reference"][str(h)]["ridge"]}
        rp_mse[h] = ((f_rp - obs.values) ** 2).mean(axis=0)           # (40, 40) per-point MSE of Ridge-patch
    out = {"n_issue_dates": int(len(issue)), "first_issue": str(issue[0].date()), "last_issue": str(issue[-1].date()), "std_deg_c": d2.std,
           "reference": {str(h): ref[h] for h in leads},
           "lstm_patch": ref_json["lstm_patch"], "models": {}}
    maps = {}
    for mname, Cs in models.items():
        out["models"][mname] = {}
        for C in Cs:
            per_seed, zones = {}, {}
            for s in (0, 1, 2):
                name = f"b2_{mname}_{region}_C{C}_s{s}"
                net = build(mname, C)
                net.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt"))
                pred = predict_b2(net, d2, d2.val_issue)                # (n, 3, 40, 40) standardised
                assert np.isfinite(pred).all(), name
                res, zres = {}, {}
                for k, h in enumerate(leads):
                    obs, ct, seas = obs_all[h]
                    f = ct.values + pred[:, k] * d2.std
                    blk = score_block(as_da(f, sst_i, issue + pd.Timedelta(days=h)), obs, ct, seas)
                    for rname in ("anomaly_persistence", "ridge", "ridge_patch"):
                        blk[f"skill_vs_{rname}"] = add_skill(blk, ref[h][rname])
                    res[str(h)] = blk
                    mse = ((f - obs.values) ** 2).mean(axis=0)
                    maps[f"{mname}_C{C}_s{s}_h{h}"] = (1 - np.sqrt(mse) / np.sqrt(rp_mse[h])).astype(np.float32)   # per-point skill vs Ridge-patch
                    zm, zr = zone_rmse(mse, lat), zone_rmse(rp_mse[h], lat)
                    zres[str(h)] = {z: 1 - zm[z] / zr[z] for z in zm}
                per_seed[str(s)] = res
                zones[str(s)] = zres
                print(f"[{region} {mname} C={C} seed={s}] " + "  ".join(f"h={h}: {res[str(h)]['overall']['rmse']:.3f}" for h in leads), flush=True)
            out["models"][mname][str(C)] = {"per_seed": per_seed, **agg(list(per_seed.values())), "zone_skill_vs_ridge_patch": agg(list(zones.values())),
                                            "curves": curves(mname, region, C)}
        sel = {str(C): float(np.mean([out["models"][mname][str(C)]["mean"][str(h)]["overall"]["rmse"] for h in leads])) for C in Cs}
        out["models"][mname]["selection"] = {"rule": "lowest mean validation RMSE over seeds 0,1,2 and leads 1,7,30", "mean_rmse": sel, "chosen_C": int(min(sel, key=sel.get))}   # keys are str so that the report (and the JSON) look them up the same way
    return out, maps


def curves(mname, region, C):
    cur = {}
    for s in (0, 1, 2):
        j = json.loads((RUNS / f"b2_{mname}_{region}_C{C}_s{s}.json").read_text())
        hv, ht = j["history"]["val_loss"], j["history"]["train_loss"]
        cur[str(s)] = {"best_epoch": j["best_epoch"], "epochs_run": j["epochs_run"], "best_val_loss": j["best_val_loss"], "last_val_loss": hv[-1],
                       "train_loss_at_best": ht[best_i(j)], "val_loss": hv, "train_loss": ht, "num_threads": j["settings"]["num_threads"]}
    return cur


def best_i(j):
    return j["best_epoch"] - 1


def f3(m, s=None):
    return f"{m:.3f}" if s is None else f"{m:.3f} ± {s:.3f}"


def chosen(R, mname):
    return str(R["models"][mname]["selection"]["chosen_C"])


def answers(res, cfg):
    """Rule-based statements; every number comes from res."""
    leads = [str(h) for h in cfg["leads_days"]]
    models = list(next(iter(res["regions"].values()))["models"])
    L = []
    cells = {m: {} for m in models}
    for r, R in res["regions"].items():
        for m in models:
            C = chosen(R, m)
            for h in leads:
                cells[m][(r, h)] = (R["models"][m][C]["mean"][h]["skill_vs_ridge_patch"]["overall"], R["models"][m][C]["std"][h]["skill_vs_ridge_patch"]["overall"],
                                    R["models"][m][C]["mean"][h]["overall"]["rmse"])
    for m in models:
        pos = [k for k, v in cells[m].items() if v[0] > 0]
        sure = [k for k, v in cells[m].items() if v[0] - v[1] > 0]
        L.append(f"- **{LABEL[m]} vs Ridge-patch (RMSE skill, mean over seeds):** positive in {len(pos)}/{len(cells[m])} region-lead cells "
                 f"(positive by more than one seed-std in {len(sure)}); range {min(v[0] for v in cells[m].values()):+.3f} to {max(v[0] for v in cells[m].values()):+.3f}. "
                 + "; ".join(f"{r} {h} d {v[0]:+.3f}" for (r, h), v in cells[m].items()))
    if len(models) == 2:
        a, b = models
        better = [k for k in cells[a] if cells[a][k][2] < cells[b][k][2]]
        L.append(f"- **{LABEL[a]} vs {LABEL[b]}:** {LABEL[a]} has the lower mean RMSE in {len(better)}/{len(cells[a])} cells (RMSE {LABEL[a]} / {LABEL[b]}: "
                 + "; ".join(f"{r} {h} d {cells[a][(r, h)][2]:.3f}/{cells[b][(r, h)][2]:.3f}" for (r, h) in cells[a]) + ").")
    regs = list(res["regions"])
    for m in models:
        diff = []
        for h in leads:
            s0, s1 = cells[m][(regs[0], h)][0], cells[m][(regs[1], h)][0]
            diff.append(f"{h} d: {regs[0]} {s0:+.3f} vs {regs[1]} {s1:+.3f}" + (" (opposite sign)" if s0 * s1 < 0 else ""))
        L.append(f"- **{LABEL[m]}, {regs[0]} vs {regs[1]} (skill vs Ridge-patch):** " + "; ".join(diff))
    for m in models:
        parts = []
        for r, R in res["regions"].items():
            C = chosen(R, m)
            for h in ("7", "30"):
                z = R["models"][m][C]["zone_skill_vs_ridge_patch"]["mean"][h]
                parts.append(f"{r} {h} d edge {z['edge']:+.3f} / centre {z['centre']:+.3f}")
        L.append(f"- **{LABEL[m]} edge vs centre (skill vs Ridge-patch, outer 4 rings vs inner 32x32):** " + "; ".join(parts))
    return L


def chosen_c_md(res):
    """Chosen-C table. (Fixed 2026-11-10: the first version looked up str keys in an int-keyed dict and printed '-'.)"""
    models = list(next(iter(res["regions"].values()))["models"])
    L = ["## Chosen C (lowest mean validation RMSE over seeds and leads)", "", "| region | model | " + " | ".join(f"mean RMSE C={c}" for c in ("16", "32")) + " | chosen |", "|---|---|---|---|---|"]
    for r, R in res["regions"].items():
        for m in models:
            sel = R["models"][m]["selection"]
            L.append(f"| {r} | {LABEL[m]} | " + " | ".join(f"{sel['mean_rmse'][c]:.4f}" if c in sel["mean_rmse"] else "-" for c in ("16", "32")) + f" | {sel['chosen_C']} |")
    return L


def append_chosen_c_correction():
    """Append (never overwrite) a dated correction with the right Chosen-C table to the two committed reports, from their own JSON."""
    for tag in ("_unet", ""):
        res = json.loads((RESULTS / f"b2_val{tag}.json").read_text())
        rep = RESULTS / f"b2_val{tag}_report.md"
        txt = rep.read_text()
        assert "Correction 2026-11-10" not in txt
        add = ["", "## Correction 2026-11-10 (the 'Chosen C' table above printed '-' for the mean RMSE columns: report-code bug, str vs int keys)", "",
               "The selection itself (`chosen`) was computed correctly; the mean RMSE values below are read from " + f"results/b2_val{tag}.json (unchanged).", ""] + chosen_c_md(res)
        rep.write_text(txt.rstrip("\n") + "\n" + "\n".join(add) + "\n")
        print("appended correction to", rep.name)


def report_md(res, cfg):
    leads = [str(h) for h in cfg["leads_days"]]
    models = list(next(iter(res["regions"].values()))["models"])
    L = ["# B2 whole-region models - validation 2016-2019 (inner 40x40, identical pairs for every method)", "",
         "RMSE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation; skill = 1 - RMSE/RMSE(Ridge-patch). Deep models: mean ± sample std (ddof=1) over seeds 0,1,2 "
         "at the chosen C. Reference rows are the frozen B1 numbers (re-checked). Timing is not a result.", ""]
    for r, R in res["regions"].items():
        L += [f"## {r}  (issue dates {R['first_issue']}..{R['last_issue']}, n={R['n_issue_dates']})", ""]
        for h in leads:
            ref = R["reference"][h]
            L += [f"### lead {h} d", "", "| method | RMSE | ACC | skill vs Ridge-patch |", "|---|---|---|---|"]
            for m, lab in (("anomaly_persistence", "anomaly persistence"), ("ridge", "Ridge"), ("ridge_patch", "Ridge-patch")):
                L.append(f"| {lab} | {f3(ref[m]['overall']['rmse'])} | {f3(ref[m]['overall']['acc'])} | {1 - ref[m]['overall']['rmse'] / ref['ridge_patch']['overall']['rmse']:+.3f} |")
            v = R["lstm_patch"]
            L.append(f"| LSTM-patch | {f3(v['mean'][h]['overall']['rmse'], v['std'][h]['overall']['rmse'])} | {f3(v['mean'][h]['overall']['acc'], v['std'][h]['overall']['acc'])} | "
                     f"{v['mean'][h]['skill_vs_ridge_patch']['overall']:+.3f} ± {v['std'][h]['skill_vs_ridge_patch']['overall']:.3f} |")
            for m in models:
                C = chosen(R, m)
                v = R["models"][m][C]
                L.append(f"| {LABEL[m]} (C={C}) | {f3(v['mean'][h]['overall']['rmse'], v['std'][h]['overall']['rmse'])} | {f3(v['mean'][h]['overall']['acc'], v['std'][h]['overall']['acc'])} | "
                         f"{v['mean'][h]['skill_vs_ridge_patch']['overall']:+.3f} ± {v['std'][h]['skill_vs_ridge_patch']['overall']:.3f} |")
            L.append("")
    L += chosen_c_md(res)
    L += ["", "## Learning curves (all C; best epoch / epochs run; last val - best val; train loss at best epoch; threads)", "",
          "| region | model | C | seed | best epoch / run | best val | last val - best val | train at best | threads |", "|---|---|---|---|---|---|---|---|---|"]
    for r, R in res["regions"].items():
        for m in models:
            for C, v in R["models"][m].items():
                if C == "selection":
                    continue
                for s, c in v["curves"].items():
                    L.append(f"| {r} | {LABEL[m]} | {C} | {s} | {c['best_epoch']}/{c['epochs_run']} | {c['best_val_loss']:.4f} | {c['last_val_loss'] - c['best_val_loss']:+.4f} | {c['train_loss_at_best']:.4f} | {c['num_threads']} |")
    L += ["", "## Skill vs Ridge-patch by season (mean over seeds, chosen C)", "", "| region | lead | model | " + " | ".join(SEASONS) + " |", "|---|---|---|" + "---|" * 4]
    for r, R in res["regions"].items():
        for h in leads:
            for m in models:
                d = R["models"][m][chosen(R, m)]["mean"][h]["skill_vs_ridge_patch"]["by_season"]
                L.append(f"| {r} | {h} | {LABEL[m]} | " + " | ".join(f"{d[s]:+.3f}" for s in SEASONS) + " |")
    L += ["", "## Spatial check: skill vs Ridge-patch, edge ring (outer 4) vs centre (inner 32x32), mean ± std over seeds (maps: figB2_maps.png)", "",
          "| region | lead | model | edge | centre |", "|---|---|---|---|---|"]
    for r, R in res["regions"].items():
        for h in leads:
            for m in models:
                z = R["models"][m][chosen(R, m)]["zone_skill_vs_ridge_patch"]
                L.append(f"| {r} | {h} | {LABEL[m]} | {z['mean'][h]['edge']:+.3f} ± {z['std'][h]['edge']:.3f} | {z['mean'][h]['centre']:+.3f} ± {z['std'][h]['centre']:.3f} |")
    L += ["", "## Answers (generated from the numbers above by code; read with the std)", ""] + answers(res, cfg)
    return "\n".join(L)


def figures(res, maps_by_region, cfg, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(next(iter(res["regions"].values()))["models"])
    regions = list(res["regions"])
    # learning curves
    fig, axes = plt.subplots(len(regions), len(models), figsize=(5 * len(models), 3.6 * len(regions)), squeeze=False)
    sty = {"0": "-", "1": "--", "2": ":"}
    for i, r in enumerate(regions):
        for j, m in enumerate(models):
            ax, C = axes[i, j], chosen(res["regions"][r], m)
            for s, c in res["regions"][r]["models"][m][C]["curves"].items():
                ep = np.arange(1, len(c["val_loss"]) + 1)
                ax.plot(ep, c["train_loss"], sty[s], color="#4477aa")
                ax.plot(ep, c["val_loss"], sty[s], color="#cc6677")
                ax.plot(c["best_epoch"], c["val_loss"][c["best_epoch"] - 1], "o", color="#cc6677", ms=4)
            ax.set_title(f"{r}, {LABEL[m]} C={C}  (blue train, red validation, dot = best)", fontsize=8)
            ax.set_xlabel("epoch")
            ax.set_ylabel("MSE (standardised)")
            ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / f"figB2_curves{tag}.png", dpi=150)
    plt.close(fig)
    # skill maps
    lats = {}
    rows = [(r, m) for r in regions for m in models]
    fig = plt.figure(figsize=(12, 3.1 * len(rows)))
    gs = fig.add_gridspec(len(rows), 4, width_ratios=[1, 1, 0.05, 1.4], wspace=0.25, hspace=0.35)
    axes = np.empty((len(rows), 3), dtype=object)
    for i in range(len(rows)):
        axes[i, 0], axes[i, 1], axes[i, 2] = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), fig.add_subplot(gs[i, 3])
    cax = fig.add_subplot(gs[:, 2])
    for i, (r, m) in enumerate(rows):
        C = chosen(res["regions"][r], m)
        mp = maps_by_region[r]
        for j, h in enumerate((7, 30)):
            a = np.mean([mp[f"{m}_C{C}_s{s}_h{h}"] for s in (0, 1, 2)], axis=0)
            im = axes[i, j].imshow(a, origin="lower", cmap="RdBu", vmin=-0.3, vmax=0.3)
            axes[i, j].set_title(f"{r}, {LABEL[m]} C={C}, {h} d", fontsize=8)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].axhline(3.5, color="k", lw=.4); axes[i, j].axhline(35.5, color="k", lw=.4)
            axes[i, j].axvline(3.5, color="k", lw=.4); axes[i, j].axvline(35.5, color="k", lw=.4)
        ax = axes[i, 2]
        yy, xx = np.indices((40, 40))
        ring = np.minimum.reduce([yy, xx, 39 - yy, 39 - xx])
        for h, col in ((7, "#4477aa"), (30, "#cc6677")):
            a = np.mean([mp[f"{m}_C{C}_s{s}_h{h}"] for s in (0, 1, 2)], axis=0)
            ax.plot(range(20), [a[ring == k].mean() for k in range(20)], "-o", ms=3, color=col, label=f"{h} d")
        ax.axhline(0, color="k", lw=.5)
        ax.set_xlabel("distance from the edge (grid points)", fontsize=7)
        ax.set_ylabel("mean per-point skill", fontsize=7)
        ax.set_title("skill vs distance to the box edge", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
    fig.colorbar(im, cax=cax, label="per-point RMSE skill vs Ridge-patch (mean over seeds)")
    fig.savefig(RESULTS / "figures" / f"figB2_maps{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models")
    p.add_argument("--tag", default="")
    p.add_argument("--fix-chosen-c", action="store_true", help="append the corrected Chosen-C table to b2_val_unet_report.md and b2_val_report.md, then exit")
    a = p.parse_args()
    if a.fix_chosen_c:
        append_chosen_c_correction()
        return
    cfg = load_config()
    from sst.b2_train import load_b2
    b2 = load_b2()
    models = {m: (b2["convlstm_ed"]["C"] if m == "convlstm_ed" else b2["C_grid"]) for m in a.models.split(",")}
    ref_all = json.loads((RESULTS / "b1_val.json").read_text())
    res = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["val"]}, "leads_days": cfg["leads_days"], "regions": {}}
    maps_by_region = {}
    for r in cfg["regions"]:
        res["regions"][r], maps_by_region[r] = evaluate_region(r, cfg, models, ref_all)
    (RESULTS / f"b2_val{a.tag}.json").write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    np.savez_compressed(RESULTS / f"b2_val_maps{a.tag}.npz", **{f"{r}__{k}": v for r, mp in maps_by_region.items() for k, v in mp.items()})
    (RESULTS / f"b2_val{a.tag}_report.md").write_text(report_md(res, cfg))
    (RESULTS / "figures").mkdir(exist_ok=True)
    figures(res, maps_by_region, cfg, a.tag)
    print(f"wrote results/b2_val{a.tag}.json, b2_val{a.tag}_report.md, b2_val_maps{a.tag}.npz, figures/figB2_*{a.tag}.png")


if __name__ == "__main__":
    main()
