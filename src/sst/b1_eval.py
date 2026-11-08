"""B1 validation report (2016-2019 only): anomaly persistence, Ridge, Ridge+doy, Ridge-patch, LSTM (point), LSTM-patch.

  python -m sst.b1_eval --checks   # inner-block / pair / std equality checks (needs halo data only) -> results/b1_checks.json
  python -m sst.b1_eval            # scores -> results/b1_val.json, results/b1_val_report.md, results/figures/figB1_curves.png

All methods are scored on IDENTICAL pairs (the existing LSTM validation samples), on the inner 40x40, with the existing metric code.
LSTM (point) numbers are read from results/lstm_val.json (H=64, not retrained); the re-computed anomaly persistence and Ridge are
asserted equal to the ones stored there. Only data up to the validation end is loaded.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from sst.b1_data import inner_block, load_b1, load_halo_sst, make_data_b1
from sst.b1_ridge import DERIVED, as_da
from sst.b1_train import RUNS, predict_b1
from sst.baselines import anomaly_persistence
from sst.climatology import clim_at
from sst.download import ROOT, load_config
from sst.lstm_data import make_data
from sst.lstm_eval import add_skill, agg, score_block
from sst.lstm_model import SSTLSTM
from sst.lstm_train import load_sst
from sst.metrics import season_labels
from sst.ridge import ridge_lead
from sst.run_val import _clean

RESULTS = ROOT / "results"
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def inner_checks(region, cfg, b1):
    """Halo-based data vs the existing inner-only data: climatology, std, anomalies of inner points, training/validation pairs."""
    halo = b1["halo"]["width_points"]
    sst_h = load_halo_sst(f"{region}_halo", cfg)
    sst_i = load_sst(region, cfg)
    d_h = make_data_b1(sst_h, cfg, halo, b1["patch"]["size"])
    d_i = make_data(sst_i, cfg)
    clim_in = inner_block(d_h.clim.values, halo)
    chk = {"inner_sst_equal": bool(np.array_equal(inner_block(sst_h.values, halo), sst_i.values, equal_nan=True)),
           "clim_inner_max_abs_diff": float(np.abs(clim_in - d_i.clim.values).max()),
           "clim_inner_bitwise_equal": bool(np.array_equal(clim_in, d_i.clim.values)),
           "std_halo_data": d_h.std, "std_existing_lstm_data": d_i.std, "std_bitwise_equal": d_h.std == d_i.std,
           "train_pairs_equal": bool(np.array_equal(d_h.train_pairs, d_i.train_pairs)),
           "val_pairs_equal": bool(np.array_equal(d_h.val_pairs, d_i.val_pairs)),
           "n_train_pairs": int(len(d_h.train_pairs)), "n_val_pairs": int(len(d_h.val_pairs)),
           "nan_in_halo_grid_up_to_val_end": int(np.isnan(sst_h.values).sum()),
           "centre_anomaly_equal_existing_max_abs_diff": float(np.abs(d_h.a[:, d_h.centre] - d_i.a).max())}
    return chk


def sst_forecast(ct_inner, pred_anom, sst_inner, times):
    return as_da(ct_inner + pred_anom, sst_inner, times)


def evaluate_region(region, cfg, b1, old_val):
    halo = b1["halo"]["width_points"]
    H = b1["lstm_patch"]["hidden"]
    sst_h = load_halo_sst(f"{region}_halo", cfg)
    data = make_data_b1(sst_h, cfg, halo, b1["patch"]["size"])
    sst_i = load_sst(region, cfg)
    old = make_data(sst_i, cfg)
    assert np.array_equal(data.val_pairs, old.val_pairs), "pairs must equal the existing validation pairs"
    P = data.n_points
    iss_idx = np.unique(data.val_pairs // P)
    issue = data.dates[iss_idx]
    leads = list(cfg["leads_days"])
    clim_inner = old.clim                       # existing (inner-only) climatology: identical pairs/targets to the existing validation
    ridge_npz = np.load(DERIVED / f"b1_ridge_{region}.npz")
    assert list(ridge_npz["issue"]) == [str(d.date()) for d in issue]
    ridge_info = json.loads((RESULTS / "b1_ridge_val.json").read_text())["regions"][region]
    old_ref = old_val["regions"][region]
    ref, obs_all = {}, {}
    for h in leads:
        tgt = issue + pd.Timedelta(days=h)
        obs, ct, seas = sst_i.sel(time=tgt), clim_at(clim_inner, tgt), season_labels(tgt)
        obs_all[h] = (obs, ct, seas)
        f_r, _, info = ridge_lead(sst_i, clim_inner, cfg, h, period="val")
        f_r = f_r.sel(time=tgt)
        r = {"anomaly_persistence": score_block(anomaly_persistence(sst_i, clim_inner, issue, tgt), obs, ct, seas),
             "ridge": score_block(f_r, obs, ct, seas)}
        for m in ("anomaly_persistence", "ridge"):   # must reproduce the stored numbers
            assert abs(r[m]["overall"]["rmse"] - old_ref["reference"][str(h)][m]["overall"]["rmse"]) < 1e-9, (m, h)
            assert abs(r[m]["overall"]["acc"] - old_ref["reference"][str(h)][m]["overall"]["acc"]) < 1e-9, (m, h)
        for m in ("ridge_doy", "ridge_patch"):
            f = sst_forecast(ct.values, ridge_npz[f"{m}_h{h}"], sst_i, tgt)
            r[m] = score_block(f, obs, ct, seas)
        ref[h] = r
    out = {"n_issue_dates": int(len(issue)), "first_issue": str(issue[0].date()), "last_issue": str(issue[-1].date()),
           "std_deg_c": data.std, "alphas": {"ridge": {str(h): ridge_info_alpha(region, h) for h in leads},
                                             "ridge_doy": {str(h): ridge_info["ridge_doy"][str(h)]["alpha"] for h in leads},
                                             "ridge_patch": {str(h): ridge_info["ridge_patch"][str(h)]["alpha"] for h in leads}},
           "alpha_at_grid_edge": {m: {str(h): ridge_info[m][str(h)]["alpha_at_grid_edge"] for h in leads} for m in ("ridge_doy", "ridge_patch")},
           "reference": {}, "lstm_patch": {}, "lstm_point": {}}
    for h in leads:
        for m in ("ridge_doy", "ridge_patch"):
            ref[h][m]["skill_vs_ridge"] = add_skill(ref[h][m], ref[h]["ridge"])
            ref[h][m]["skill_vs_anomaly_persistence"] = add_skill(ref[h][m], ref[h]["anomaly_persistence"])
        ref[h]["ridge"]["skill_vs_anomaly_persistence"] = add_skill(ref[h]["ridge"], ref[h]["anomaly_persistence"])
        out["reference"][str(h)] = ref[h]
    # LSTM point: stored results, H=64 (not retrained)
    pt = old_ref["lstm"][str(H)]
    out["lstm_point"] = {"source": "results/lstm_val.json (H=64, seeds 0,1,2; not retrained)", "per_seed": pt["per_seed"], "mean": pt["mean"], "std": pt["std"]}
    # LSTM patch
    per_seed = {}
    for s in b1["lstm_patch"]["seeds"]:
        name = f"b1_{region}_patch_H{H}_s{s}"
        model = SSTLSTM(H, n_in=data.n_in)
        model.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt"))
        pred = predict_b1(model, data, data.val_pairs).reshape(len(issue), *data.n_inner, len(leads))
        res = {}
        for k, h in enumerate(leads):
            obs, ct, seas = obs_all[h]
            f = as_da(ct.values + pred[..., k] * data.std, sst_i, issue + pd.Timedelta(days=h))
            blk = score_block(f, obs, ct, seas)
            blk["skill_vs_anomaly_persistence"] = add_skill(blk, ref[h]["anomaly_persistence"])
            blk["skill_vs_ridge"] = add_skill(blk, ref[h]["ridge"])
            blk["skill_vs_ridge_patch"] = add_skill(blk, ref[h]["ridge_patch"])
            blk["skill_vs_ridge_doy"] = add_skill(blk, ref[h]["ridge_doy"])
            res[str(h)] = blk
        per_seed[str(s)] = res
        print(f"[{region} lstm_patch seed={s}] " + "  ".join(f"h={h}: {res[str(h)]['overall']['rmse']:.3f}" for h in leads), flush=True)
    out["lstm_patch"] = {"per_seed": per_seed, **agg(list(per_seed.values()))}
    # learning curves
    out["curves"] = {}
    for tag, pat in (("lstm_patch", f"b1_{region}_patch_H{H}_s{{s}}.json"), ("lstm_point", f"{region}_H{H}_s{{s}}.json")):
        d = RUNS if tag == "lstm_patch" else ROOT / "results" / "lstm_runs"
        out["curves"][tag] = {}
        for s in b1["lstm_patch"]["seeds"]:
            j = json.loads((d / pat.format(s=s)).read_text())
            hv, ht = j["history"]["val_loss"], j["history"]["train_loss"]
            out["curves"][tag][str(s)] = {"best_epoch": j["best_epoch"], "epochs_run": j["epochs_run"], "best_val_loss": j["best_val_loss"],
                                          "last_val_loss": hv[-1], "last_train_loss": ht[-1], "val_loss": hv, "train_loss": ht}
    return out


def ridge_info_alpha(region, h):
    return json.loads((RESULTS / "ridge_alphas.json").read_text())["regions"][region][str(h)]["alpha"]


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"


def report_md(res, cfg, checks):
    leads = list(cfg["leads_days"])
    L = ["# B1 spatial context - validation 2016-2019 (inner 40x40, identical pairs for every method)", "",
         "RMSE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation (primary ACC); skill = 1 - RMSE/RMSE(Ridge). "
         "LSTM entries: mean ± sample std (ddof=1) over seeds 0,1,2 of per-seed scores.", ""]
    for r, R in res["regions"].items():
        L += [f"## {r}  (issue dates {R['first_issue']}..{R['last_issue']}, n={R['n_issue_dates']})", ""]
        for h in leads:
            ref, hs = R["reference"][str(h)], str(h)
            L += [f"### lead {h} d", "", "| method | RMSE | ACC | skill vs Ridge |", "|---|---|---|---|"]
            for m, lab in (("anomaly_persistence", "anomaly persistence"), ("ridge", "Ridge"), ("ridge_doy", "Ridge+doy"), ("ridge_patch", "Ridge-patch")):
                sk = 1 - ref[m]["overall"]["rmse"] / ref["ridge"]["overall"]["rmse"]   # skill vs Ridge (0 for Ridge itself)
                L.append(f"| {lab} | {fmt(ref[m]['overall']['rmse'])} | {fmt(ref[m]['overall']['acc'])} | {sk:+.3f} |")
            for key, lab in (("lstm_point", "LSTM (point)"), ("lstm_patch", "LSTM-patch")):
                v = R[key]
                L.append(f"| {lab} | {fmt(v['mean'][hs]['overall']['rmse'], v['std'][hs]['overall']['rmse'])} | "
                         f"{fmt(v['mean'][hs]['overall']['acc'], v['std'][hs]['overall']['acc'])} | "
                         f"{v['mean'][hs]['skill_vs_ridge']['overall']:+.3f} ± {v['std'][hs]['skill_vs_ridge']['overall']:.3f} |")
            L.append("")
    L += ["## Skill vs Ridge by season (RMSE skill; LSTM = mean over seeds)", "", "| region | lead | method | " + " | ".join(SEASONS) + " |", "|---|---|---|" + "---|" * 4]
    for r, R in res["regions"].items():
        for h in leads:
            hs, ref = str(h), R["reference"][str(h)]
            rows = {"Ridge+doy": ref["ridge_doy"]["skill_vs_ridge"]["by_season"], "Ridge-patch": ref["ridge_patch"]["skill_vs_ridge"]["by_season"],
                    "LSTM (point)": R["lstm_point"]["mean"][hs]["skill_vs_ridge"]["by_season"],
                    "LSTM-patch": R["lstm_patch"]["mean"][hs]["skill_vs_ridge"]["by_season"]}
            for lab, d in rows.items():
                L.append(f"| {r} | {h} | {lab} | " + " | ".join(f"{d[s]:+.3f}" for s in SEASONS) + " |")
    L += ["", "## Ridge alphas chosen on validation (grid 1e-3..1e3; * = grid edge)", "", "| region | model | lead 1 | lead 7 | lead 30 |", "|---|---|---|---|---|"]
    for r, R in res["regions"].items():
        for m in ("ridge", "ridge_doy", "ridge_patch"):
            edge = R["alpha_at_grid_edge"].get(m, {})
            L.append(f"| {r} | {m} | " + " | ".join(f"{R['alphas'][m][str(h)]:g}{'*' if edge.get(str(h)) else ''}" for h in leads) + " |")
    L += ["", "## Learning curves (best epoch / epochs run; last val - best val; train vs val at best epoch)", "", "| region | model | seed | best epoch / run | best val | last val - best val | train loss at best epoch |", "|---|---|---|---|---|---|---|"]
    for r, R in res["regions"].items():
        for tag in ("lstm_point", "lstm_patch"):
            for s, c in R["curves"][tag].items():
                be = c["best_epoch"]
                L.append(f"| {r} | {tag} | {s} | {be}/{c['epochs_run']} | {c['best_val_loss']:.4f} | {c['last_val_loss'] - c['best_val_loss']:+.4f} | {c['train_loss'][be - 1]:.4f} |")
    L += ["", "## Data checks", "```", json.dumps(checks, indent=1), "```"]
    return "\n".join(L)


def curves_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regions = list(res["regions"])
    fig, axes = plt.subplots(1, len(regions), figsize=(10, 4), squeeze=False)
    sty = {"0": "-", "1": "--", "2": ":"}
    for ax, r in zip(axes[0], regions):
        for tag, col in (("lstm_point", "#999999"), ("lstm_patch", "#cc6677")):
            for s, c in res["regions"][r]["curves"][tag].items():
                ep = np.arange(1, len(c["val_loss"]) + 1)
                ax.plot(ep, c["val_loss"], sty[s], color=col, label=f"{tag} val s{s}")
                ax.plot(ep, c["train_loss"], sty[s], color="#4477aa" if tag == "lstm_patch" else "#bbccdd", label=f"{tag} train s{s}")
                ax.plot(c["best_epoch"], c["val_loss"][c["best_epoch"] - 1], "o", color=col, ms=4)
        ax.set_title(r)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    axes[0, 0].set_ylabel("MSE (standardised units)")
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.suptitle("LSTM learning curves: point vs patch (dot = best validation epoch)")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checks", action="store_true")
    a = p.parse_args()
    cfg, b1 = load_config(), load_b1()
    if a.checks:
        checks = {r: inner_checks(r, cfg, b1) for r in cfg["regions"]}
        (RESULTS / "b1_checks.json").write_text(json.dumps(checks, indent=1))
        print(json.dumps(checks, indent=1))
        return
    checks = json.loads((RESULTS / "b1_checks.json").read_text())
    old_val = json.loads((RESULTS / "lstm_val.json").read_text())
    res = {"period": {"train": cfg["split"]["train"], "evaluated": cfg["split"]["val"]}, "leads_days": cfg["leads_days"],
           "regions": {r: evaluate_region(r, cfg, b1, old_val) for r in cfg["regions"]}}
    (RESULTS / "b1_val.json").write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    (RESULTS / "b1_val_report.md").write_text(report_md(res, cfg, checks))
    curves_figure(res, RESULTS / "figures" / "figB1_curves.png")
    print("wrote results/b1_val.json, results/b1_val_report.md, results/figures/figB1_curves.png")


if __name__ == "__main__":
    main()
