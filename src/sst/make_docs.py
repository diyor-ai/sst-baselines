"""Generate README.md, report/report.md and report/report.html from results/*.json (no hand-typed numbers).

Run `python -m sst.make_docs`; the PDF is made from report.html by LibreOffice (see README "Reproduce").
Qualitative statements are guarded by asserts: if the numbers ever contradict the text, generation fails.
"""
import html
import json

from sst.download import ROOT, load_config

R = ROOT / "results"
J = lambda n: json.loads((R / n).read_text())  # noqa: E731
LABEL = {"persistence": "Persistence", "climatology": "Climatology", "anomaly_persistence": "Anomaly persistence",
         "damped_persistence": "Damped persistence", "ocn": "OCN (10 yr)", "ridge": "Ridge"}
LEADS = ("1", "7", "30")


def f3(x):
    return f"{x:.3f}"


def sg(x):
    return f"{x:+.3f}"


def pm(a, b):
    return f"{f3(a)} ± {b:.3f}"


class Data:
    def __init__(self):
        self.cfg = load_config()
        self.val, self.test = J("val_all.json"), J("test_all.json")
        self.tb, self.vb = J("test_bias.json"), J("val_bias.json")
        self.stats = J("region_stats.json")["final"]
        self.alph = J("ridge_alphas.json")
        self.qc = J("qc_kuroshio.json")
        self.psl, self.samples = J("psl_validation.json"), J("psl_version_samples.json")
        self.regions = list(self.test["regions"])
        self.lv, self.lt = J("lstm_val.json"), J("lstm_test.json")   # LSTM validation / test (added after the baselines)
        self.bv, self.bt = J("b1_val.json"), J("b1_test.json")       # B1 spatial context: validation / test (added after the LSTM)
        self.bm, self.bh = J("b1_manifest.json"), J("halo_checks.json")
        self.b2t, self.b2v, self.b2c = J("b2_test.json"), J("b2_val_res.json"), J("b2_test_choice.json")   # B2 whole-region models: test / validation / choice

    def rmse(self, d, r, h, m):
        return d["regions"][r][h]["methods"][m]["overall"]["rmse"]

    def skill(self, d, r, h, m, ref="anomaly_persistence"):
        return d["regions"][r][h]["skill"][m][f"rmse_skill_vs_{ref}"]


def cell(D, d, r, h, m):
    return f"{f3(D.rmse(d, r, h, m))} ({sg(D.skill(d, r, h, m))})"


def results_table(D, region, which=("val", "test")):
    head = ["Method"] + [f"{'Val' if w == 'val' else 'Test'} {h} d" for w in which for h in LEADS]
    rows = [[LABEL[m]] + [cell(D, D.val if w == "val" else D.test, region, h, m) for w in which for h in LEADS]
            for m in D.test["methods"]]
    return head, rows


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def html_table(head, rows):
    h = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table border="1" cellspacing="0" cellpadding="2" width="100%"><tr>{h}</tr>{b}</table>'


def facts(D):
    """All numbers used in prose, taken from JSON, plus asserts that guard the qualitative claims."""
    t, v, K, S = D.test, D.val, "kuroshio", "subtropical"
    n_psl = len(D.psl["pieces"])
    f = {
        "n_days": D.qc["days_present"], "first": D.qc["first_date"], "last": D.qc["last_date"],
        "grid": "x".join(map(str, D.qc["grid_shape"])), "n_pieces": n_psl, "n_pieces_ok": sum(p["month_ok"] for p in D.psl["pieces"]),
        "n_years": len(D.psl["years"]), "n_years_ok": sum(1 for y in D.psl["years"].values() if y["cells_diff_gt_tol"] == 0),
        "max_diff": max(y["max_abs_diff_vs_erddap"] for y in D.psl["years"].values()), "tol": D.psl["tolerance"],
        "n_samples": len(D.samples["rows"]), "samples_max": max(x["max_abs_diff"] for x in D.samples["rows"]),
        "std_pt_K": D.stats[K]["point_anomaly_std_cosw_mean"], "std_pt_S": D.stats[S]["point_anomaly_std_cosw_mean"],
        "n_test": t["regions"][K]["1"]["pairs"]["used"], "n_val": v["regions"][K]["1"]["pairs"]["used"],
        "skipped": sum(t["regions"][r][h]["pairs"]["skipped"] for r in D.regions for h in LEADS),
        "commit": t["run"]["commit"], "run_utc": t["run"]["started_utc"],
    }
    import xarray as xr
    src = xr.open_dataset(ROOT / "data" / "processed" / "kuroshio.nc")["source"].values
    f["n_gap"] = int((src == 1).sum())
    f["n_erddap"] = int((src == 0).sum())
    # ---- guarded claims ----
    for r in D.regions:  # anomaly persistence/ridge << climatology & OCN at 1 day
        assert D.rmse(t, r, "1", "ridge") < 0.5 * min(D.rmse(t, r, "1", "climatology"), D.rmse(t, r, "1", "ocn"))
    assert all(D.skill(t, K, h, "ridge") > 0 for h in LEADS)                  # Ridge beats AP everywhere in Kuroshio
    assert D.skill(t, S, "1", "ridge") < 0 and D.skill(t, S, "7", "ridge") < 0   # but not at 1/7 d in the subtropical box
    assert all(D.skill(t, r, "30", "persistence") < -0.5 for r in D.regions)    # plain persistence is poor at 30 d
    for r in D.regions:  # claims behind the ranking-change finding
        assert all(D.skill(v, r, h, "ridge") > 0 for h in LEADS)
        assert D.skill(t, r, "1", "ridge") < D.skill(v, r, "1", "ridge") and D.skill(t, r, "30", "ridge") < D.skill(v, r, "30", "ridge")
        assert D.tb["regions"][r]["1"]["climatology"]["bias"] < D.vb["bias_by_region"][r]["1"]["climatology"]["bias"] < 0
        assert D.tb["regions"][r]["30"]["damped_persistence"]["bias"] < D.vb["bias_by_region"][r]["30"]["damped_persistence"]["bias"] < 0
    assert all(D.skill(v, K, h, "damped_persistence") > 0 for h in LEADS)
    assert all(D.skill(t, S, h, "damped_persistence") < 0 for h in LEADS)
    assert all(D.skill(t, K, h, "damped_persistence") < D.skill(v, K, h, "damped_persistence") for h in LEADS)
    for r in D.regions:
        bb = D.tb["regions"][r]["30"]
        assert bb["climatology"]["bias"] < -0.5 and bb["ridge"]["bias"] < 0 and bb["damped_persistence"]["bias"] < 0
        assert abs(bb["anomaly_persistence"]["bias"]) < 0.02
        assert D.rmse(t, r, "1", "climatology") > D.rmse(v, r, "1", "climatology")  # clim error grew from val to test
    return f


def findings(D, f):
    t, v, K, S = D.test, D.val, "kuroshio", "subtropical"
    r = lambda d, reg, h, m: D.rmse(d, reg, h, m)  # noqa: E731
    k = lambda reg, h, m: D.skill(t, reg, h, m)  # noqa: E731
    kv = lambda reg, h, m: D.skill(v, reg, h, m)  # noqa: E731
    b = lambda reg, h, m: D.tb["regions"][reg][h][m]["bias"]  # noqa: E731
    bv = lambda reg, h, m: D.vb["bias_by_region"][reg][h][m]["bias"]  # noqa: E731
    sk = lambda fn, reg, m: ", ".join(sg(fn(reg, h, m)) for h in LEADS)  # noqa: E731
    return [
        ("Anomaly persistence is the bar to beat, especially at short range.",
         f"Its 1-day RMSE is {f3(r(t, S, '1', 'anomaly_persistence'))} °C (subtropical) and {f3(r(t, K, '1', 'anomaly_persistence'))} °C (Kuroshio), "
         f"while climatology gives {f3(r(t, S, '1', 'climatology'))} and {f3(r(t, K, '1', 'climatology'))} °C and OCN {f3(r(t, S, '1', 'ocn'))} and "
         f"{f3(r(t, K, '1', 'ocn'))} °C. Ridge's margin over anomaly persistence at 1 day is small "
         f"(skill {sg(k(K, '1', 'ridge'))} Kuroshio, {sg(k(S, '1', 'ridge'))} subtropical)."),
        ("Ridge helps in the high-variability Kuroshio box, not reliably in the subtropical box.",
         f"Kuroshio: Ridge beats anomaly persistence at every lead (skill {sk(k, K, 'ridge')} at 1, 7, 30 d). Subtropical: Ridge is worse at 1 and 7 days "
         f"({sg(k(S, '1', 'ridge'))}, {sg(k(S, '7', 'ridge'))}) and about equal at 30 d ({sg(k(S, '30', 'ridge'))})."),
        ("Plain persistence fails at 30 days, and the warm shift hurts climatology-based forecasts.",
         f"Persistence at 30 d has skill {sg(k(K, '30', 'persistence'))} (Kuroshio) and {sg(k(S, '30', 'persistence'))} (subtropical) relative to anomaly persistence. "
         f"Against the 1982–2015 climatology the test years are warm: mean bias of the climatology forecast is {sg(b(K, '30', 'climatology'))} °C (Kuroshio) and "
         f"{sg(b(S, '30', 'climatology'))} °C (subtropical); Ridge at 30 d is also biased low ({sg(b(K, '30', 'ridge'))}, {sg(b(S, '30', 'ridge'))} °C)."),
        ("The ranking changed between validation and test.",
         f"Methods that relax towards the 1982–2015 climatology (Ridge, damped persistence) beat anomaly persistence on validation but lost much or all of "
         f"that advantage on test. Skill vs anomaly persistence at 1, 7, 30 d, validation → test: Ridge Kuroshio {sk(kv, K, 'ridge')} → {sk(k, K, 'ridge')}; "
         f"Ridge subtropical {sk(kv, S, 'ridge')} → {sk(k, S, 'ridge')}; damped persistence Kuroshio {sk(kv, K, 'damped_persistence')} → {sk(k, K, 'damped_persistence')}; "
         f"damped persistence subtropical {sk(kv, S, 'damped_persistence')} → {sk(k, S, 'damped_persistence')} (negative at every test lead). "
         f"This is consistent with a larger warm shift in the test years: the climatology forecast's bias is {sg(bv(K, '1', 'climatology'))} / {sg(bv(S, '1', 'climatology'))} °C "
         f"on validation but {sg(b(K, '1', 'climatology'))} / {sg(b(S, '1', 'climatology'))} °C on test (Kuroshio / subtropical), and the 30-day bias of damped persistence "
         f"goes from {sg(bv(K, '30', 'damped_persistence'))} / {sg(bv(S, '30', 'damped_persistence'))} to {sg(b(K, '30', 'damped_persistence'))} / "
         f"{sg(b(S, '30', 'damped_persistence'))} °C. It is not shown here that the warm shift is the cause.")]


REFS = ["Huang, B., et al. (2021). Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1. Journal of Climate, 34(8), 2923–2939. doi:10.1175/JCLI-D-20-0166.1",
        "Murphy, A. H. (1992). Climatology, persistence, and their linear combination as standards of reference in skill scores. Weather and Forecasting, 7(4), 692–698. doi:10.1175/1520-0434(1992)007<0692:CPATLC>2.0.CO;2",
        "Huang, J., van den Dool, H. M., and Barnston, A. G. (1996). Long-lead seasonal temperature prediction using optimal climate normals. Journal of Climate, 9(4), 809–817. doi:10.1175/1520-0442(1996)009<0809:LLSTPU>2.0.CO;2",
        "Hao, R., Zhao, Y., Zhang, S., and Deng, X. (2025). Deep learning for ocean forecasting: a comprehensive review of methods, applications, and datasets. IEEE Transactions on Cybernetics, 55(6), 2879–2898. doi:10.1109/TCYB.2025.3539990"]
REFS_MD = "\n".join(f"{i}. {x}" for i, x in enumerate(REFS, 1))

FIGS = [("results/figures/fig1_regions.png", "Figure 1. Mean SST and standard deviation of the daily anomaly (training years) in the two regions."),
        ("results/figures/fig2_val_leads.png", "Figure 2 (validation). RMSE and skill vs anomaly persistence by lead."),
        ("results/figures/fig2_test_leads.png", "Figure 2 (test, 2020–2025). Same layout; values below the axis are listed in the box."),
        ("results/figures/fig3_test_seasons.png", "Figure 3. Test skill vs anomaly persistence by season (mean ± std over years, dots = years)."),
        ("results/figures/figS1_area_mean_anomaly.png", "Figure S1. Area-mean anomaly vs the 1982–2015 climatology; the validation period is shaded.")]


# ---- LSTM experiment (added after the baseline test run; protocol docs/lstm_test_protocol.md, tag pre-lstm-test) ----
def lstm_stat(D, r, h, key, test=True):
    """(mean, std) over seeds of an LSTM score: key in rmse, acc, skill_ap, skill_ridge."""
    blk = D.lt["regions"][r]["lstm"] if test else D.lv["regions"][r]["lstm"]["64"]
    m, s = blk["mean"][h], blk["std"][h]
    if key in ("rmse", "acc"):
        return m["overall"][key], s["overall"][key]
    k = {"skill_ap": "skill_vs_anomaly_persistence", "skill_ridge": "skill_vs_ridge"}[key]
    return m[k]["overall"], s[k]["overall"]


def lstm_table(D):
    head = ["Region, lead", "AP RMSE", "Ridge RMSE", "LSTM RMSE (mean±std)", "ACC AP / Ridge / LSTM", "Skill vs AP", "Skill vs Ridge", "Skill vs Ridge (validation)"]
    rows = []
    for r in D.regions:
        for h in LEADS:
            ref = D.lt["regions"][r]["reference"][h]
            ap, rd = ref["anomaly_persistence"]["overall"], ref["ridge"]["overall"]
            (a, sa), (c, sc), (e, se) = lstm_stat(D, r, h, "rmse"), lstm_stat(D, r, h, "skill_ap"), lstm_stat(D, r, h, "skill_ridge")
            (acc, sacc), (v, sv) = lstm_stat(D, r, h, "acc"), lstm_stat(D, r, h, "skill_ridge", test=False)
            rows.append([f"{r.capitalize()} {h} d", f3(ap["rmse"]), f3(rd["rmse"]), f"{f3(a)} ± {sa:.3f}",
                         f"{ap['acc']:.3f} / {rd['acc']:.3f} / {acc:.3f} ± {sacc:.3f}", f"{sg(c)} ± {sc:.3f}", f"{sg(e)} ± {se:.3f}", f"{sg(v)} ± {sv:.3f}"])
    return head, rows


def lstm_hypotheses(D):
    """Two hypotheses (clearly marked as not tested), each with asserts on the numbers they rest on."""
    K, S = "kuroshio", "subtropical"
    ref = lambda r, m: D.lt["regions"][r]["reference"]["30"][m]["overall"]  # noqa: E731
    for r in D.regions:   # 30 d: lower RMSE than anomaly persistence, but lower ACC than anomaly persistence AND Ridge
        assert lstm_stat(D, r, "30", "rmse")[0] < ref(r, "anomaly_persistence")["rmse"]
        assert lstm_stat(D, r, "30", "acc")[0] < ref(r, "anomaly_persistence")["acc"]
        assert lstm_stat(D, r, "30", "acc")[0] < ref(r, "ridge")["acc"]
    sd = lambda r, h, test: D.lt["regions"][r]["lstm"]["std"][h]["overall"]["rmse"] if test else D.lv["regions"][r]["lstm"]["64"]["std"][h]["overall"]["rmse"]  # noqa: E731
    ratio = lambda r, h: sd(r, h, True) / sd(r, h, False)  # noqa: E731
    assert all(ratio(K, h) > 2.5 for h in LEADS) and ratio(S, "7") > 2 and ratio(S, "1") < 1 and 1 < ratio(S, "30") < 1.5
    R = lambda r: ", ".join(f"{ratio(r, h):.1f}×" for h in LEADS)  # noqa: E731
    return [
        f"**ACC at 30 days (hypothesis, not tested).** At 30 days the LSTM has a lower RMSE than anomaly persistence in both boxes but a *lower* temporal ACC than both anomaly persistence and Ridge "
        f"(ACC {lstm_stat(D, K, '30', 'acc')[0]:.3f} vs {ref(K, 'anomaly_persistence')['acc']:.3f} / {ref(K, 'ridge')['acc']:.3f} Kuroshio; {lstm_stat(D, S, '30', 'acc')[0]:.3f} vs "
        f"{ref(S, 'anomaly_persistence')['acc']:.3f} / {ref(S, 'ridge')['acc']:.3f} subtropical). This is consistent with the RMSE gain coming from a damped forecast amplitude (smaller anomalies "
        "that reduce the squared error) rather than from a better pattern of anomalies; amplitude and pattern were not analysed separately.",
        f"**Seed spread (hypothesis, not tested).** The std of the RMSE over the three seeds is larger on test than on validation in the Kuroshio box at every lead "
        f"(test/validation ratio at 1, 7, 30 d: {R(K)}; e.g. 1 d: {sd(K, '1', False):.4f} → {sd(K, '1', True):.4f} °C) and at 7 d in the subtropical box ({ratio(S, '7'):.1f}×), but not at 1 d there "
        f"({ratio(S, '1'):.1f}×) and only slightly at 30 d ({ratio(S, '30'):.1f}×). A possible explanation is that the test years contain inputs outside the range seen in training (the warm shift), "
        "where the differently-initialised networks extrapolate differently; this was not tested."]


def lstm_facts(D):
    """Prose for the LSTM section; every qualitative claim is guarded by an assert on the numbers."""
    K, S = "kuroshio", "subtropical"
    sr = lambda r, h: lstm_stat(D, r, h, "skill_ridge")[0]  # noqa: E731
    sa = lambda r, h: lstm_stat(D, r, h, "skill_ap")[0]  # noqa: E731
    svr = lambda r, h: lstm_stat(D, r, h, "skill_ridge", test=False)[0]  # noqa: E731
    assert all(sr(K, h) < 0 for h in LEADS)                       # Kuroshio: LSTM never beats Ridge on test
    assert all(sr(S, h) > 0 for h in LEADS)                       # subtropical: LSTM better than Ridge at every lead
    assert sa(S, "1") < 0 and sa(S, "7") < 0 and sa(S, "30") > 0 and sa(K, "30") > 0 and sa(K, "1") < 0 and sa(K, "7") < 0
    assert svr(K, "7") > 0 and svr(K, "30") > 0 and svr(K, "1") < 0 and svr(S, "1") < 0   # validation signs quoted below
    assert all(lstm_stat(D, r, "30", "acc")[0] < D.lt["regions"][r]["reference"]["30"]["anomaly_persistence"]["overall"]["acc"] for r in D.regions)
    L = lambda r: ", ".join(sg(sr(r, h)) for h in LEADS)  # noqa: E731
    V = lambda r: ", ".join(sg(svr(r, h)) for h in LEADS)  # noqa: E731
    A = lambda r: ", ".join(sg(sa(r, h)) for h in LEADS)  # noqa: E731
    seas = lambda r, h, g: D.lt["regions"][r]["lstm"]["mean"][h]["skill_vs_ridge"]["by_season"][g]  # noqa: E731
    assert all(seas(S, h, g) > 0 for h in LEADS for g in ("DJF", "MAM", "JJA", "SON"))
    return [
        f"**Result (test, H = 64, mean ± std over seeds 0, 1, 2).** The LSTM does **not** beat Ridge in the Kuroshio box (skill vs Ridge {L(K)} at 1, 7, 30 d; "
        f"on validation it had been {V(K)}, so the 7- and 30-day gains did not hold). In the subtropical box it beats Ridge at every lead (skill {L(S)}; validation {V(S)}), "
        f"but Ridge itself is worse than anomaly persistence at 1 and 7 days there, and the LSTM does not beat anomaly persistence at those leads either "
        f"(skill vs anomaly persistence {A(S)}; Kuroshio {A(K)}). At 30 days the LSTM is better than anomaly persistence in both boxes. ACC at 30 d is slightly lower than that of anomaly persistence in both boxes.",
        f"**Seasons.** Subtropical: positive vs Ridge in all four seasons at every lead (several within about one std). Kuroshio: mostly negative vs Ridge, "
        f"worst in DJF at 1 day ({sg(seas(K, '1', 'DJF'))}).",
        "**Fairness caveat.** The LSTM gets the full 30-day sequence plus sin/cos of the day of year; Ridge gets lags 0–6 and the 30-day mean and no day of year, so part of any LSTM "
        "gain may come from the extra inputs, not from the nonlinear model (not tested). Validation and test used slightly different pair conventions (see the protocol). "
        "This is a learning experiment, not an ocean forecasting model, and the Kuroshio and subtropical results disagree, so no general claim is made."]


# ---- B1 spatial context (added after the LSTM test; protocols docs/b1_spatial_protocol.md, docs/b1_test_protocol.md, tag pre-b1-test) ----
def b1_ref(D, r, h, m, test=True):
    return (D.bt if test else D.bv)["regions"][r]["reference"][h][m]


def b1_lstm(D, r, h, key, test=True):
    """(mean, std) over seeds of LSTM-patch: key in rmse, acc, skill_ridge, skill_ap, skill_rp (vs Ridge-patch)."""
    blk = (D.bt if test else D.bv)["regions"][r]["lstm_patch"]
    m, s = blk["mean"][h], blk["std"][h]
    if key in ("rmse", "acc"):
        return m["overall"][key], s["overall"][key]
    k = {"skill_ap": "skill_vs_anomaly_persistence", "skill_ridge": "skill_vs_ridge", "skill_rp": "skill_vs_ridge_patch"}[key]
    return m[k]["overall"], s[k]["overall"]


def b1_rp_skill(D, r, h, test=True):
    """Ridge-patch skill vs Ridge (and Ridge+doy vs Ridge) overall RMSE skill."""
    return b1_ref(D, r, h, "ridge_patch", test)["skill_vs_ridge"]["overall"]


def b1_tables(D):
    pm = lambda a, b: f"{f3(a)} ± {b:.3f}"  # noqa: E731
    h1 = ["Region, lead", "Anom. persistence", "Ridge", "Ridge+doy", "Ridge-patch", "LSTM (point)", "LSTM-patch"]
    h2 = ["Region, lead", "Ridge+doy vs Ridge", "Ridge-patch vs Ridge [val]", "LSTM-patch vs Ridge [val]", "LSTM-patch vs Ridge-patch [val]", "LSTM-patch vs AP"]
    h3 = ["Region, lead", "AP", "Ridge", "Ridge+doy", "Ridge-patch", "LSTM (point)", "LSTM-patch"]
    r1, r2, r3 = [], [], []
    for r in D.regions:
        for h in LEADS:
            n = f"{r.capitalize()} {h} d"
            o = D.lt["regions"][r]["lstm"]
            r1.append([n] + [f3(b1_ref(D, r, h, m)["overall"]["rmse"]) for m in ("anomaly_persistence", "ridge", "ridge_doy", "ridge_patch")]
                      + [pm(o["mean"][h]["overall"]["rmse"], o["std"][h]["overall"]["rmse"]), pm(*b1_lstm(D, r, h, "rmse"))])
            sk = lambda key, t=True: b1_lstm(D, r, h, key, t)  # noqa: E731
            r2.append([n, sg(b1_ref(D, r, h, "ridge_doy")["skill_vs_ridge"]["overall"]),
                       f"{sg(b1_rp_skill(D, r, h))} [{sg(b1_rp_skill(D, r, h, False))}]",
                       f"{sg(sk('skill_ridge')[0])} ± {sk('skill_ridge')[1]:.3f} [{sg(sk('skill_ridge', False)[0])}]",
                       f"{sg(sk('skill_rp')[0])} ± {sk('skill_rp')[1]:.3f} [{sg(sk('skill_rp', False)[0])}]",
                       f"{sg(sk('skill_ap')[0])} ± {sk('skill_ap')[1]:.3f}"])
            r3.append([n] + [f"{b1_ref(D, r, h, m)['overall']['acc']:.3f}" for m in ("anomaly_persistence", "ridge", "ridge_doy", "ridge_patch")]
                      + [f"{o['mean'][h]['overall']['acc']:.3f}", pm(*b1_lstm(D, r, h, "acc"))])
    return (h1, r1), (h2, r2), (h3, r3)


def b1_facts(D):
    """Prose for the B1 section; every qualitative claim is guarded by an assert on the numbers."""
    K, S = "kuroshio", "subtropical"
    seas = ("DJF", "MAM", "JJA", "SON")
    # data checks
    for r in D.bh.values():
        assert r["days"] == 16071 and r["grid_shape"] == [48, 48] and r["nan_values_total"] == 0
        assert r["sst_equal_bitwise_incl_nan"] and r["source_equal"] and r["halo_points"] == 704
    assert D.bt["run"]["dry_run"] is False and D.bt["run"]["tag"] == "pre-b1-test"
    # (a) doy alone does nothing; linear patch gain holds
    assert all(abs(b1_ref(D, r, h, "ridge_doy", t)["skill_vs_ridge"]["overall"]) < 5e-4 for r in D.regions for h in LEADS for t in (True, False))
    assert all(b1_rp_skill(D, r, h) > 0 for r in D.regions for h in LEADS)
    assert all(b1_rp_skill(D, r, h) > 0 for r in D.regions for h in ("7", "30")) and all(b1_rp_skill(D, r, h, False) > 0 for r in D.regions for h in ("7", "30"))
    assert all(abs(b1_rp_skill(D, r, h) - b1_rp_skill(D, r, h, False)) < 0.01 for r in D.regions for h in ("7", "30"))   # "similar size"
    ap = lambda r, h, m: 1 - b1_ref(D, r, h, m)["overall"]["rmse"] / b1_ref(D, r, h, "anomaly_persistence")["overall"]["rmse"]  # noqa: E731
    assert all(ap(K, h, "ridge_patch") > 0 for h in LEADS) and ap(S, "1", "ridge_patch") < 0 and ap(S, "7", "ridge_patch") < 0 and ap(S, "30", "ridge_patch") > 0
    # (b) LSTM-patch vs Ridge-patch
    rp = lambda r, h: b1_lstm(D, r, h, "skill_rp")  # noqa: E731
    assert rp(S, "7")[0] > 2 * rp(S, "7")[1] > 0 and rp(S, "30")[0] > 2 * rp(S, "30")[1] > 0
    assert all(D.bt["regions"][S]["lstm_patch"]["per_seed"][k][h]["skill_vs_ridge_patch"]["overall"] > 0 for k in "012" for h in ("7", "30"))
    assert all(rp(K, h)[0] < 0 for h in ("1", "7")) and abs(rp(K, "30")[0]) < rp(K, "30")[1]
    sg30 = [D.bt["regions"][K]["lstm_patch"]["per_seed"][k]["30"]["skill_vs_ridge_patch"]["overall"] for k in "012"]
    assert min(sg30) < 0 < max(sg30)                                                                                    # seeds of both signs
    assert all(D.bt["regions"][r]["lstm_patch"]["per_seed"][k]["1"]["skill_vs_ridge_patch"]["overall"] < 0 for r in D.regions for k in "012")
    # (c) validation vs test
    assert all(rp(S, h, ) is not None for h in LEADS) and all(b1_lstm(D, S, h, "skill_ridge", False)[0] > 0.05 for h in ("7", "30"))
    assert all(abs(b1_lstm(D, S, h, "skill_ridge")[0] - b1_lstm(D, S, h, "skill_ridge", False)[0]) < 0.01 for h in ("7", "30"))
    assert all(b1_lstm(D, K, h, "skill_rp", False)[0] > 0 for h in ("7", "30")) and all(rp(K, h)[0] < 0 for h in ("7", "30"))
    assert b1_lstm(D, K, "1", "skill_ridge")[0] < -0.1 and b1_lstm(D, K, "1", "skill_ridge", False)[0] > -0.02
    # LSTM-patch vs LSTM (point)
    assert all(b1_lstm(D, r, h, "rmse")[0] < D.lt["regions"][r]["lstm"]["mean"][h]["overall"]["rmse"] for r in D.regions for h in ("7", "30"))
    assert all(b1_lstm(D, r, "1", "rmse")[0] > D.lt["regions"][r]["lstm"]["mean"]["1"]["overall"]["rmse"] for r in D.regions)
    # seasons
    sl = lambda r, h: D.bt["regions"][r]["reference"][h]["ridge_patch"]["skill_vs_ridge"]["by_season"]  # noqa: E731
    assert all(sl(r, h)[g] > 0 for r in D.regions for h in ("7", "30") for g in seas if not (r == K and h == "7" and g == "DJF")) and sl(K, "7")["DJF"] < 0
    lp = lambda r, h: D.bt["regions"][r]["lstm_patch"]["mean"][h]["skill_vs_ridge_patch"]["by_season"]  # noqa: E731
    assert all(lp(S, h)[g] > 0 for h in ("7", "30") for g in seas) and sum(lp(K, "30")[g] > 0 for g in seas) == 1 and lp(K, "30")["DJF"] > 0
    sd_k1 = D.bt["regions"][K]["lstm_patch"]["std"]["1"]["overall"]["rmse"]
    sk = lambda r, h, f_=sg: f_(b1_lstm(D, r, h, "skill_rp")[0])  # noqa: E731
    P = lambda r, h: f"{sg(b1_lstm(D, r, h, 'skill_rp')[0])} ± {b1_lstm(D, r, h, 'skill_rp')[1]:.3f}"  # noqa: E731
    V = lambda r, h: sg(b1_lstm(D, r, h, "skill_rp", False)[0])  # noqa: E731
    R = lambda r, test=True: ", ".join(sg(b1_rp_skill(D, r, h, test)) for h in LEADS)  # noqa: E731
    return [
        f"**Inputs and models.** The 40×40 boxes were extended by a 1° halo (4 grid points per side, 48×48 grids; same ERDDAP + PSL pipeline and checks; 16,071 days, no missing/land point, "
        f"the inner 40×40 equals the earlier files bit for bit: `results/halo_checks.json`). Targets and scoring stay on the inner 40×40 and use the same pairs as before; the halo only provides inputs. "
        f"A patch is the 5×5 block of grid points centred on the target point. **Ridge+doy** = the existing Ridge features plus sin/cos of the day of year; **Ridge-patch** = lags 0–6 and the 30-day mean at each of the 25 patch points plus sin/cos of the day of year (202 features); "
        f"**LSTM-patch** = the earlier LSTM (H = 64, same training settings, seeds 0, 1, 2) with the 25 patch anomalies plus sin/cos of the day of year as input of every day (30 × 27). All fitted on 1982–2015, "
        f"ridge alphas and LSTM early stopping chosen on 2016–2019; these models were tested once (commit `{D.bt['run']['commit'][:7]}`, tag `pre-b1-test`, started {D.bt['run']['started_utc']}). "
        f"Anomaly persistence and Ridge were recomputed on the test pairs and equal the frozen `results/test_all.json`; the weights hashes equal `results/b1_manifest.json`. LSTM (point) is the frozen test result of the previous section.",
        f"**Did the linear patch gain hold?** Yes. The day of year alone does nothing for Ridge (skill of Ridge+doy vs Ridge within ±0.0005 in all six cells, on validation and on test). Ridge-patch beats Ridge in all six test cells "
        f"(skill vs Ridge Kuroshio {R(K)}, subtropical {R(S)} at 1, 7, 30 d; validation {R(K, False)} and {R(S, False)}), and at 7 and 30 d its validation gain kept its sign and a similar size (differences below 0.01). "
        f"Ridge-patch is better than anomaly persistence at every lead in the Kuroshio box, but in the subtropical box it is still worse than anomaly persistence at 1 and 7 days (skill {sg(ap(S, '1', 'ridge_patch'))}, {sg(ap(S, '7', 'ridge_patch'))}) and better only at 30 days ({sg(ap(S, '30', 'ridge_patch'))}).",
        f"**Did LSTM-patch beat Ridge-patch (nonlinear gain, inputs held equal)?** In the subtropical box at 7 and 30 days, yes: skill vs Ridge-patch {P(S, '7')} and {P(S, '30')} (all three seeds positive). "
        f"In the Kuroshio box no: {P(K, '1')} at 1 d, {P(K, '7')} at 7 d, and at 30 d it is a tie ({P(K, '30')}, seeds of both signs). At 1 day LSTM-patch is worse than Ridge-patch in both boxes in every seed "
        f"({P(S, '1')} subtropical, where the seed spread is large). The patch also lowers the LSTM's RMSE at 7 and 30 days in both boxes relative to LSTM (point), but raises it at 1 day.",
        f"**Did the validation gains hold?** Partly. Subtropical LSTM-patch vs Ridge held almost exactly at 7 and 30 days ({sg(b1_lstm(D, S, '7', 'skill_ridge')[0])}, {sg(b1_lstm(D, S, '30', 'skill_ridge')[0])} on test vs "
        f"{sg(b1_lstm(D, S, '7', 'skill_ridge', False)[0])}, {sg(b1_lstm(D, S, '30', 'skill_ridge', False)[0])} on validation). In the Kuroshio box the nonlinear gain did not: LSTM-patch vs Ridge-patch was "
        f"{V(K, '7')} and {V(K, '30')} at 7 and 30 d on validation and is {sk(K, '7')} and {sk(K, '30')} on test; at 1 day the Kuroshio LSTM-patch got much worse on test (skill vs Ridge {sg(b1_lstm(D, K, '1', 'skill_ridge')[0])} vs "
        f"{sg(b1_lstm(D, K, '1', 'skill_ridge', False)[0])} on validation; seed std of the RMSE {sd_k1:.3f} °C). As with the earlier LSTM, the Kuroshio results are the less stable ones.",
        f"**Seasons.** Ridge-patch beats Ridge in all four seasons at 7 and 30 d in both boxes except Kuroshio DJF at 7 d ({sg(sl(K, '7')['DJF'])}). LSTM-patch beats Ridge-patch in all four seasons at 7 and 30 d in the subtropical box; "
        f"in the Kuroshio box at 30 d only in DJF ({sg(lp(K, '30')['DJF'])}).",
        "**Caveats (stated before the test).** Alphas of the point Ridge and Ridge+doy sit at the lower edge of the grid at 7 and 30 days (validation curve flat there, not changed); the Ridge-patch alphas are interior. "
        "Validation numbers are slightly optimistic: the new ridge alphas and the LSTM early stopping were chosen on the same validation years. The earlier point-LSTM gains shrank or vanished on test, and the Kuroshio gains of LSTM-patch over Ridge shrank or vanished in the same way. "
        "Validation and test use different pair conventions (see the protocols). Two boxes that disagree, three seeds and one test run: no general claim is made; this is a learning experiment, not an ocean forecasting model."]


def b1_short(D):
    b1_facts(D)   # runs the asserts that guard the statements below
    K, S = "kuroshio", "subtropical"
    P = lambda r, h: f"{sg(b1_lstm(D, r, h, 'skill_rp')[0])}"  # noqa: E731
    return (f"The day of year alone does not help Ridge; the patch does (Ridge-patch beats Ridge in all six cells). LSTM-patch beats Ridge-patch in the subtropical box at 7 and 30 d "
            f"({P(S, '7')}, {P(S, '30')} skill vs Ridge-patch; its validation gain vs Ridge held) but not in the Kuroshio box ({P(K, '1')}, {P(K, '7')}, {P(K, '30')}; validation gains did not hold). "
            "Two boxes that disagree, three seeds, one test run: no general claim.")


# ---- B2 whole-region models (added after B1; protocols docs/b2_protocol.md, docs/b2_test_protocol.md, tag pre-b2-test) ----
FAMS = (("unet", "U-Net"), ("convlstm_ed", "ConvLSTM-ED"))


def b2m(D, r, fam, h, key="skill_vs_ridge_patch"):
    M = D.b2t["regions"][r]["models"][fam]
    if key == "rmse":
        return M["mean"][h]["overall"]["rmse"], M["std"][h]["overall"]["rmse"]
    return M["mean"][h][key]["overall"], M["std"][h][key]["overall"]


def b2_val_skill(D, r, fam, h):
    v = D.b2c["regions"][r][fam]
    return D.b2v["regions"][r]["models"][v["chosen"]][str(v["C"])]["mean"][h]["skill_vs_ridge_patch"]["overall"]


def b2_tables(D):
    head = ["region", "lead", "Ridge-patch", "LSTM-patch", "U-Net", "ConvLSTM-ED", "U-Net skill vs Ridge-patch [val]", "ConvLSTM-ED skill vs Ridge-patch [val]"]
    rows = []
    for r in D.regions:
        R = D.b2t["regions"][r]
        for h in LEADS:
            lp = R["lstm_patch"]
            row = [r, f"{h} d", f3(R["reference"][h]["ridge_patch"]["overall"]["rmse"]), pm(lp["mean"][h]["overall"]["rmse"], lp["std"][h]["overall"]["rmse"])]
            row += [pm(*b2m(D, r, f, h, "rmse")) for f, _ in FAMS]
            row += [f"{pm(*b2m(D, r, f, h))} [{sg(b2_val_skill(D, r, f, h))}]" for f, _ in FAMS]
            rows.append(row)
    return head, rows


def b2_facts(D):
    """Prose for the B2 section; every qualitative claim is guarded by an assert on the numbers."""
    K, S = "kuroshio", "subtropical"
    run = D.b2t["run"]
    assert run["dry_run"] is False and run["tag"] == "pre-b2-test"
    assert all(c["chosen"].startswith("res") for r in D.regions for c in D.b2c["regions"][r].values())
    assert all(b2m(D, r, f, h)[0] - b2m(D, r, f, h)[1] > 0 for r in D.regions for f, _ in FAMS for h in ("7", "30"))
    assert all(b2m(D, r, f, "1")[0] < 0 for r in D.regions for f, _ in FAMS)
    assert all(b2m(D, r, f, "1", "skill_vs_anomaly_persistence")[0] < 0 for r in D.regions for f, _ in FAMS)
    assert all(b2m(D, r, f, h, "skill_vs_lstm_patch")[0] > 0 for r in D.regions for f, _ in FAMS for h in LEADS)
    assert all(b2m(D, r, "unet", h, "rmse")[0] < b2m(D, r, "convlstm_ed", h, "rmse")[0] for r in D.regions for h in LEADS)
    assert all(b2_val_skill(D, r, f, h) > 0 for r in D.regions for f, _ in FAMS for h in ("7", "30"))
    assert all(b2m(D, r, f, h)[0] > b2_val_skill(D, r, f, h) for r in D.regions for f, _ in FAMS for h in ("7", "30"))
    assert b2_val_skill(D, S, "unet", "1") > 0 and b2m(D, S, "unet", "1")[0] < 0 and b2_val_skill(D, K, "unet", "1") < 0
    sk = lambda r, f, h, k="skill_vs_ridge_patch": f"{sg(b2m(D, r, f, h, k)[0])} ± {b2m(D, r, f, h, k)[1]:.3f}"  # noqa: E731
    rng = lambda f, k: f"{sg(min(b2m(D, r, f, h, k)[0] for r in D.regions for h in LEADS))} to {sg(max(b2m(D, r, f, h, k)[0] for r in D.regions for h in LEADS))}"  # noqa: E731
    return [
        f"**Models and choice.** One model sees the whole 48×48 box (30 days of anomaly + sin/cos day of year) and forecasts all 3 leads at once for the inner 40×40: a small **U-Net** (C = 32, 3 downsamplings, GroupNorm) and a **ConvLSTM encoder-forecaster** "
        f"(C = 16; strided conv, 2-layer ConvLSTM at 24×24 on every 3rd day, 10 steps; a deviation from the first spec for compute reasons, documented in `docs/b2_protocol.md`). "
        f"Each also has a **residual** variant (output = last input anomaly + network). Per region and family the variant with the lowest mean validation RMSE (seeds 0, 1, 2 × leads 1, 7, 30) went to the test, chosen by code "
        f"(`results/b2_test_choice.json`): the residual one in all four cases (validation mean RMSE plain → residual, Kuroshio U-Net {D.b2c['regions'][K]['unet']['mean_rmse_plain']:.4f} → {D.b2c['regions'][K]['unet']['mean_rmse_residual']:.4f}, "
        f"subtropical U-Net {D.b2c['regions'][S]['unet']['mean_rmse_plain']:.4f} → {D.b2c['regions'][S]['unet']['mean_rmse_residual']:.4f}). Trained on 1982–2015 only (~12k samples), early stopping on 2016–2019, seeds 0, 1, 2; "
        f"the 12 existing checkpoints were tested once (commit `{run['commit'][:7]}`, tag `pre-b2-test`, started {run['started_utc']}), hashes checked against the manifests, and the baselines recomputed on the test pairs equal the frozen B1/baseline test results.",
        f"**Beat the patch models?** At 7 and 30 days yes, in both boxes and for both models, by more than one seed-std: skill vs Ridge-patch U-Net Kuroshio {sk(K, 'unet', '7')} / {sk(K, 'unet', '30')}, subtropical {sk(S, 'unet', '7')} / {sk(S, 'unet', '30')}; "
        f"ConvLSTM-ED Kuroshio {sk(K, 'convlstm_ed', '7')} / {sk(K, 'convlstm_ed', '30')}, subtropical {sk(S, 'convlstm_ed', '7')} / {sk(S, 'convlstm_ed', '30')}. "
        f"At 1 day no: both models are below Ridge-patch (U-Net {sk(K, 'unet', '1')} Kuroshio, {sk(S, 'unet', '1')} subtropical; ConvLSTM-ED {sk(K, 'convlstm_ed', '1')}, {sk(S, 'convlstm_ed', '1')}) and also worse than anomaly persistence in all four cells. "
        f"Against LSTM-patch the means are positive in all 12 model-region-lead cells (U-Net {rng('unet', 'skill_vs_lstm_patch')}, ConvLSTM-ED {rng('convlstm_ed', 'skill_vs_lstm_patch')}); the 1-day gain is partly because LSTM-patch was poor at 1 day on test (B1).",
        f"**U-Net vs ConvLSTM-ED, regions, validation.** The U-Net has the lower test RMSE than the ConvLSTM-ED in all six region-lead cells (differences small, e.g. Kuroshio 30 d {b2m(D, K, 'unet', '30', 'rmse')[0]:.3f} vs {b2m(D, K, 'convlstm_ed', '30', 'rmse')[0]:.3f} °C), "
        f"with a ~50× larger parameter count. The gain over Ridge-patch is larger in the subtropical box than in the Kuroshio box at 7 and 30 d. Validation conclusions held where they were positive: the 7 and 30 d gains over Ridge-patch were positive on validation and larger on test; "
        f"at 1 day the subtropical U-Net was slightly positive on validation ({sg(b2_val_skill(D, S, 'unet', '1'))}) and is negative on test.",
        "**Caveats.** C, the variant and early stopping were chosen on the validation years; ~12k highly autocorrelated training samples for 4×10⁴–2×10⁶ parameters, validation best epochs 2–8; three seeds measure training noise only, not the sampling uncertainty of six test years; "
        "the residual variant mainly fixes the 1-day forecast and is nearly neutral at 7 and 30 d (validation only: `results/b2_val_res_report.md`); the unchosen variant was not run on test. Learning experiment, not an ocean forecasting model."]


def b2_short(D):
    b2_facts(D)   # runs the asserts that guard the statements below
    K, S = "kuroshio", "subtropical"
    g = lambda r, f, h: sg(b2m(D, r, f, h)[0])  # noqa: E731
    return (f"Whole-region models (U-Net, ConvLSTM-ED; residual variants chosen on validation) beat Ridge-patch at 7 and 30 d in both boxes (U-Net skill vs Ridge-patch Kuroshio {g(K, 'unet', '7')}, {g(K, 'unet', '30')}; "
            f"subtropical {g(S, 'unet', '7')}, {g(S, 'unet', '30')}) and LSTM-patch in all cells, but not at 1 d (U-Net {g(K, 'unet', '1')}, {g(S, 'unet', '1')}; also below anomaly persistence). Three seeds, one test run.")


def readme(D, f):
    cfg = D.cfg
    t, K, S = D.test, "kuroshio", "subtropical"
    L = []
    A = L.append
    A("# sst-baselines\n")
    A("**Baselines and a reproducible evaluation for short-range sea-surface-temperature (SST) forecasts in two small "
      "North Pacific regions.** Six simple methods (persistence, climatology, anomaly persistence, damped persistence, "
      "optimal climate normals, per-grid-point Ridge regression) are compared at 1, 7 and 30 days with a fixed, "
      "pre-registered protocol: training 1982–2015, validation 2016–2019, test 2020–2025 evaluated once. "
      "This is **not** an ocean forecasting model; it is a study of how hard simple baselines are to beat.\n")
    A("## Purpose")
    b = cfg["regions"]
    A(f"Know the strength of simple baselines before any learned model is compared with them. Two 10°×10° boxes: a high-variability "
      f"Kuroshio-extension box ({b[K]['lat_min']}–{b[K]['lat_max']}°N, {b[K]['lon_min']}–{b[K]['lon_max']}°E; per-point anomaly std "
      f"{f['std_pt_K']:.2f} °C) and a low-variability subtropical box ({b[S]['lat_min']}–{b[S]['lat_max']}°N, "
      f"{b[S]['lon_min']}–{b[S]['lon_max']}°E; {f['std_pt_S']:.2f} °C), per-point std of the daily anomaly vs the final training climatology "
      f"(1982–2015, `results/region_stats.json`).\n")
    A("## Data")
    A(f"- **Primary: ERDDAP** `{cfg['dataset']['dataset_id']}` (NOAA OISST v2.1 final, daily, 0.25°, variable `sst`) – {f['n_erddap']:,} days per region.")
    A(f"- **Gap fill: NOAA PSL OPeNDAP** (`{cfg['dataset']['psl_base'].split('/Datasets/')[1]}`, `{cfg['dataset']['psl_file_pattern']}`) – only for the "
      f"{f['n_gap']:,} days per region that the ERDDAP aggregation lacks (mostly 1992–1998; visible in the server's own time axis, "
      f"copied to `results/erddap_time_axis.csv`). Checks: every gap month fetched twice and identical, in range, plausible day-to-day, and matching ERDDAP on all "
      f"overlap days ({f['n_pieces_ok']}/{f['n_pieces']} pieces, {f['n_years_ok']}/{f['n_years']} region-years, max |difference| "
      f"{f['max_diff']:.1e} °C, tolerance {f['tol']:.0e}). Nothing is interpolated; the source of every day is stored in `data/processed/<region>.nc` (`source`).")
    A("- **Data are not included in this repository.** `data/` (raw downloads, the merged `data/processed/*.nc`) is gitignored and is regenerated by the scripts under *Reproduce*; "
      "only the small time-axis file and the QC / validation JSON files are kept in `results/`.")
    A(f"- Result: **{f['n_days']:,} daily fields per region** ({f['first']} … {f['last']}), {f['grid']} grid, no missing day, no land point.")
    A("- Downloaded on 2026-11-02 (ERDDAP primary; PSL for gap days).")
    A("- Split (`config.yaml`): training 1982–2015, validation 2016–2019, test 2020–2025.\n")
    A("## Data notes")
    A("- **1996 cold event (kept unchanged):** the record minimum of the Kuroshio box, −1.14 °C, is on 1996-03-30 at 39.875°N, 146.375°E "
      "(PSL gap fill; the neighbouring days come from ERDDAP and agree). Not a download or processing artefact (two sources agree, smooth for ~10 days, "
      "spatially coherent), but extreme for the region in March (~8 °C below the local mean) and possibly a feature of the AVHRR-only analysis of that era. "
      "It lies in the training period.")
    A(f"- **v2 / v2.1 label at 2016:** PSL labels its files OISST v2 up to 2015 and v2.1 from 2016 – the year the validation period starts. "
      f"{f['n_samples']} single-day samples from both label periods match ERDDAP v2.1 (max |difference| {f['samples_max']:.1e} °C, `results/psl_version_samples.json`); "
      "no numerical difference was seen at 2016.\n")
    A("## Protocol (fixed before looking at validation or test scores)")
    A("- Leads h = 1, 7, 30 days; target = SST at t+h, issued on day t using data at or before t.\n"
      "- Climatology: day-of-year mean over training years only, smoothed with 3 annual harmonics (Feb 29 uses Feb 28); anomaly = SST − climatology.\n"
      "- Scores with cos(latitude) weights: RMSE, MAE, ACC (primary: temporal per grid point; secondary: daily spatial), skill = 1 − RMSE/RMSE_reference "
      "(references: anomaly persistence = primary, climatology), by season and by year. Pairs with a missing input/target day are skipped and counted "
      f"({f['skipped']} skipped in the test period).\n"
      "- Ridge alpha chosen on validation RMSE only; the final Ridge model is fitted on training years only with that alpha (no refit on validation).\n"
      "- The test period was evaluated **once**, after the configuration was frozen and tagged (`pre-test-freeze`); `sst.run_test` refuses to run otherwise.\n")
    A("## Baselines")
    A("| name | forecast for t+h |\n|---|---|\n| persistence | SST(t) |\n| climatology | climatology(t+h) |\n| anomaly persistence | climatology(t+h) + anomaly(t) |\n"
      "| damped persistence | climatology(t+h) + r(h)·anomaly(t), r(h) per grid point from training years |\n"
      "| OCN | mean of the same calendar day over the previous 10 years (all before the issue date) |\n"
      "| Ridge | climatology(t+h) + Ridge(anomaly lags 0–6 and mean anomaly t−29..t); one model per grid point and lead; features standardised with training statistics; alpha from a 13-value log grid 1e-3…1e3, one alpha per region and lead |\n")
    A("Chosen Ridge alphas (validation): " + "; ".join(
        f"{r}: " + ", ".join(f"{h} d = {D.alph['regions'][r][h]['alpha']:.3g}" for h in LEADS) for r in D.regions) + ".\n")
    A("**How each baseline is calculated:** [`report/methods_note.pdf`](report/methods_note.pdf) (exact formulas, every choice explained, and a worked numeric example from the real data; source `report/methods_note.md`).\n")
    A("## Results")
    A(f"Cells: RMSE in °C (skill vs anomaly persistence, unitless; 0 for anomaly persistence by definition). Validation = 2016–2019 "
      f"({f['n_val']} days per lead), test = 2020–2025 ({f['n_test']} days per lead). Full scores (MAE, ACC, skill vs climatology, by season and year) "
      "are in `results/val_all.json` and `results/test_all.json`; bias in `results/val_bias.json` and `results/test_bias.json`.\n")
    for r in D.regions:
        A(f"**{r}**\n")
        A(md_table(*results_table(D, r)) + "\n")
    A("**Key findings (test)**\n")
    for i, (h, body) in enumerate(findings(D, f), 1):
        A(f"{i}. **{h}** {body}")
    A("")
    A("### Figures\n")
    for path, cap in FIGS:
        A(f"![{cap}]({path})\n\n*{cap}*\n")
    A("## LSTM experiment (added after the baselines; the findings above are unchanged)")
    A("Question: with the **same information as Ridge** (the last 30 days of anomaly at one grid point), does a small LSTM forecast better than Ridge and anomaly persistence? "
      "One LSTM per region (1 layer, hidden size chosen on validation: 64 in both regions, 3 seeds), trained on 1982–2015 only, early stopping on 2016–2019. "
      "The protocol was written and tagged (`pre-lstm-test`) before the test period was loaded; the test was run once: [`docs/lstm_test_protocol.md`](docs/lstm_test_protocol.md). "
      "Cells: RMSE in °C and skill (1 − RMSE/RMSE_reference), mean ± std over the three seeds (no seed picking, no ensemble). Full scores incl. seasons: `results/lstm_test.json`, `results/lstm_val.json`; run provenance: `results/lstm_manifest.json`.\n")
    A(md_table(*lstm_table(D)) + "\n")
    for para in lstm_facts(D)[:2] + lstm_hypotheses(D) + lstm_facts(D)[2:]:
        A(para + "\n")
    A("![LSTM learning curves](results/figures/figL1_lstm_curves.png)\n\n*Figure L1. LSTM learning curves (validation period; dot = best epoch).*\n")
    A("## B1: spatial context (added after the LSTM experiment; everything above is unchanged)")
    A("Question: does adding the neighbouring grid points help, and how much of an LSTM gain is the extra inputs rather than the model? "
      "Protocols, written and committed before any model was run on the test period: [`docs/b1_spatial_protocol.md`](docs/b1_spatial_protocol.md) (halo data, models, validation) and "
      "[`docs/b1_test_protocol.md`](docs/b1_test_protocol.md) (test; tag `pre-b1-test`; run once). Full scores incl. seasons: `results/b1_test.json`, `results/b1_val.json`, `results/b1_val_report.md`; "
      "provenance: `results/b1_manifest.json`, `results/b1_checks.json`, `results/halo_checks.json`.\n")
    (t1, t2, t3) = b1_tables(D)
    A("Test RMSE in °C (LSTM: mean ± std over seeds 0, 1, 2):\n")
    A(md_table(*t1) + "\n")
    A("Skill = 1 − RMSE/RMSE_reference (RMSE skill; validation in brackets; LSTM mean ± std over seeds):\n")
    A(md_table(*t2) + "\n")
    A("Temporal ACC (test):\n")
    A(md_table(*t3) + "\n")
    for para in b1_facts(D):
        A(para + "\n")
    A("![B1 learning curves](results/figures/figB1_curves.png)\n\n*Figure B1. LSTM learning curves, point vs patch (validation period; dot = best epoch).*\n")
    A("## B2: whole-region models (added after B1; everything above is unchanged)")
    A("Question: do models that see the whole 48×48 box at once (U-Net, ConvLSTM encoder-forecaster) beat the patch models? "
      "Protocols, written and committed before the models were trained / before the test period was touched: [`docs/b2_protocol.md`](docs/b2_protocol.md) (models, validation, residual variant) and "
      "[`docs/b2_test_protocol.md`](docs/b2_test_protocol.md) (test; tag `pre-b2-test`; run once). Full scores incl. seasons and MAE: `results/b2_test.json`, `results/b2_test_report.md`; validation: `results/b2_val_report.md`, "
      "`results/b2_val_res_report.md` (plain vs residual); provenance: `results/b2_manifest.json`, `results/b2_manifest_res.json`.\n")
    A("Test RMSE in °C (LSTM-patch, U-Net, ConvLSTM-ED: mean ± std over seeds 0, 1, 2) and RMSE skill vs Ridge-patch (validation in brackets):\n")
    A(md_table(*b2_tables(D)) + "\n")
    for para in b2_facts(D):
        A(para + "\n")
    for r in D.regions:   # the captions below describe a 4-model x 2-region x 3-seed layout; fail loudly if the data changed
        assert set(D.b2v["regions"][r]["models"]) == {"unet", "resunet", "convlstm_ed", "resconvlstm_ed"}
    A("![B2 validation maps](results/figures/figB2_maps_res.png)\n\n"
      "*Figure B2 (maps; validation 2016–2019, not test). One row per region and model (Kuroshio then subtropical; U-Net C=32, U-Net residual C=32, ConvLSTM C=16, ConvLSTM residual C=16), "
      "mean over seeds 0, 1, 2. Left and middle columns: per-grid-point RMSE skill vs Ridge-patch (1 − RMSE_model / RMSE_Ridge-patch) over the 40×40 evaluated points at 7 d and 30 d lead; "
      "blue = lower RMSE than Ridge-patch, red = higher, white = equal, colour scale fixed at −0.3 to +0.3 (the colour-bar label is partly overprinted by the panels' axes). "
      "The black lines mark the outer 4-point ring. Right column: the same skill averaged over rings of equal distance from the box edge (0 = outermost point), 7 d in blue and 30 d in red, with a zero line.*\n")
    A("![B2 learning curves](results/figures/figB2_curves_res.png)\n\n"
      "*Figure B2 (learning curves; training 1982–2015, validation 2016–2019). One panel per region (rows) and model variant (columns: U-Net C=32, U-Net residual C=32, ConvLSTM C=16, ConvLSTM residual C=16). "
      "Blue = training loss, red = validation loss, both MSE on standardised data per epoch; line styles solid, dashed, dotted are seeds 0, 1, 2, and the dot marks the best-validation epoch of each seed "
      "(the epoch whose checkpoint was kept). The panels have their own y-axes and different numbers of epochs because of early stopping.*\n")
    A("## Status")
    A(f"Validation and test evaluation complete. The test period was evaluated once (commit `{f['commit'][:7]}`, tag `pre-test-freeze`, "
      f"started {f['run_utc']}). **Post-test changes:** none to methods or parameters. After the test run only figure and documentation code "
      "was added (fig 2 / fig 3 layout, this README, the report). The LSTM test was then run once under its own pre-registered protocol (commit `" + D.lt["run"]["commit"][:7] + "`, tag `pre-lstm-test`, started " + D.lt["run"]["started_utc"] + "); the baseline results were not changed. The B1 spatial-context models were then tested once under their own pre-registered protocol (commit `" + D.bt["run"]["commit"][:7] + "`, tag `pre-b1-test`, started " + D.bt["run"]["started_utc"] + "); neither the baseline nor the LSTM results were changed, and this README was updated once after that test. The B2 whole-region models were then tested once under their own pre-registered protocol (commit `" + D.b2t["run"]["commit"][:7] + "`, tag `pre-b2-test`, started " + D.b2t["run"]["started_utc"] + "); no earlier result was changed. **Post-test documentation-only changes:** corrected key finding 1 (damped persistence at 1 day in the Kuroshio box "
      "is 0.262 °C, so \"all anomaly-based methods\" was wrong), added the validation-to-test ranking-change finding, the data-availability note, the references and the report header; "
      "no method, parameter or score changed.\n")
    A("## Limitations")
    A("- Two small regions, one product (OISST); results may not transfer elsewhere.\n"
      "- Simple statistical baselines only; no dynamical model, no other predictors, no probabilistic scores or uncertainty intervals.\n"
      "- OISST is an interpolated analysis (smooth in time), so 1-day persistence is very strong and the 1-day margins between methods are small.\n"
      "- Hybrid data sources (ERDDAP + PSL for the gap days), verified identical on all overlap days, but still two sources.\n"
      "- PSL labels the product v2 up to 2015 and v2.1 from 2016 – the same year the validation period starts (no numerical difference found).\n"
      "- The 1996 cold event is kept unchanged.\n"
      "- Warm shift: validation and test years are warmer than the 1982–2015 climatology; no detrending was applied (OCN is included instead). "
      "Methods that revert towards the training climatology carry a negative bias that grows with lead.\n"
      "- Ridge is fitted per grid point and ignores spatial structure.\n"
      "- Ridge alpha was chosen at the lower edge of the pre-fixed grid for several region/lead cases (validation curve flat there, `results/ridge_alphas.json`); the grid was not extended.\n"
      "- B1: only a 5×5 patch of two boxes, three seeds, one test run; the halo is inputs only. The point-Ridge and Ridge+doy alphas sit at the lower grid edge at 7 and 30 days (flat curve, not changed); validation numbers are slightly optimistic because alphas and early stopping used the validation years.\n"
      "- B2: ~12k training samples for the whole-region networks, three seeds, one test run; the U-Net/ConvLSTM-ED margins over Ridge-patch at 7 and 30 d are measured on six test years in two boxes only.\n"
      "- One fixed split (no cross-validation over time blocks); six test years are a short record.\n"
      "- I am still learning oceanography; interpretations here are descriptive, not physical explanations.\n")
    A("## Reproduce")
    A("```bash\npython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt\nexport PYTHONPATH=src\n"
      "python -m sst.download --all            # ERDDAP, validated per file\npython -m sst.fill                      # PSL gap days, validated pieces only\n"
      "python -m sst.fill --samples            # v2 / v2.1 label check\npython -m sst.data                      # merge + QC -> data/processed, results/qc_*.json\n"
      "python -m sst.figures                   # figure 1 + results/region_stats.json\npython -m sst.run_val_all               # validation: all baselines + Ridge, bias, alphas\n"
      "python -m sst.figures2                  # validation fig 2 + fig S1\n# freeze: commit config.yaml + src, then: git tag pre-test-freeze\n"
      "python -m sst.run_test                  # test period, ONCE (guarded)\npython -m sst.figures2 test             # test fig 2 + fig 3\n"
      "python -m sst.lstm_train --all --jobs 4   # LSTM: 12 runs (validation only)\npython -m sst.lstm_manifest             # run provenance\npython -m sst.lstm_eval                 # LSTM validation scores + learning curves\n# commit, then: git tag pre-lstm-test\npython -m sst.lstm_test                 # LSTM test period, ONCE (guarded)\n"
      "python -m sst.halo all                  # B1: 1 deg halo data (new files only) + checks\npython -m sst.b1_eval --checks          # inner block / pair / std equality\n"
      "python -m sst.b1_ridge                  # B1: Ridge+doy, Ridge-patch (validation)\npython -m sst.b1_train --all --jobs 4   # B1: LSTM-patch, 6 runs (validation only)\npython -m sst.b1_manifest               # B1 provenance\npython -m sst.b1_eval                   # B1 validation report\n"
      "# commit, then: git tag pre-b1-test\npython -m sst.b1_test                   # B1 test period, ONCE (guarded)\n"
      "python -m sst.b2_train ...; python -m sst.b2_check; python -m sst.b2_manifest; python -m sst.b2_eval   # B2 (see scripts/b2_chain.sh, scripts/b2_chain_res.sh); validation only\n"
      "python -m sst.b2_eval_res; python -m sst.b2_test_choice   # plain vs residual (validation) -> results/b2_test_choice.json\n"
      "# commit, then: git tag pre-b2-test\npython -m sst.b2_test                   # B2 test period, ONCE (guarded); python -m sst.b2_test_report\n"
      "python -m sst.make_docs                 # README.md + report/report.md/.html from results/*.json\n"
      "python report/build_report_pdf.py   # report.pdf (embeds the figure, then soffice)\n"
      "python report/methods_example.py && python report/build_methods_note.py   # methods note (then soffice as above on report/methods_note.html)\npytest                                  # synthetic tests\n```\n")
    A("## References")
    A(REFS_MD + "\n")
    A("## How this was built")
    A("The code was written with Claude Code (an AI coding assistant); the methodology, decisions and verification are mine.\n")
    return "\n".join(L)


def report_blocks(D, f):
    t, K, S = D.test, "kuroshio", "subtropical"
    cols = [(r, h) for r in D.regions for h in LEADS]
    head = ["Method"] + [f"{r.capitalize()} {h} d" for r, h in cols]
    rows = [[LABEL[m]] + [cell(D, t, r, h, m) for r, h in cols] for m in t["methods"]]
    fi = findings(D, f)
    B = [("h1", "Simple SST forecast baselines in two North Pacific boxes: a test-period evaluation"),
         ("auth", "Mukhammaddiyor Abdulazimov · Harbin Engineering University · November 2026"),
         ("p", "**Question.** How hard are simple baselines to beat for 1-, 7- and 30-day SST forecasts, and does a per-grid-point Ridge regression help? "
               "This is a baselines study, not an ocean forecasting model."),
         ("p", f"**Data.** NOAA OISST v2.1 daily 0.25° SST, two {f['grid']}-point boxes (Kuroshio extension, per-point anomaly std {f['std_pt_K']:.2f} °C; "
               f"subtropical, {f['std_pt_S']:.2f} °C), {f['first']}–{f['last']} ({f['n_days']:,} days). ERDDAP is the primary source; {f['n_gap']:,} days it lacks were filled from "
               f"NOAA PSL after checks (double fetch, range, step, exact match on overlaps: max |difference| {f['max_diff']:.1e} °C)."),
         ("p", f"**Protocol.** Training 1982–2015 (climatology: 3 harmonics; Ridge fit), validation 2016–2019 (Ridge alpha), test 2020–2025 "
               f"({f['n_test']} days per lead) evaluated once after a frozen, tagged configuration. Score: cos(lat)-weighted RMSE; skill = 1 − RMSE/RMSE of anomaly persistence."),
         ("p", "**Baselines.** persistence; climatology; anomaly persistence; damped persistence (training-year lag correlation); OCN (mean of the same day over the previous 10 years); "
               "Ridge (anomaly lags 0–6 and 30-day mean anomaly, one model per grid point and lead)."),
         ("cap", "Test results: RMSE in °C (skill vs anomaly persistence)."),
         ("table", head, rows),
         ("img", "results/figures/fig2_test_leads.png", "Figure: test RMSE and skill by lead."),
         ("p", "**Key findings.**"),
         ("ol", [f"**{h}** {b}" for h, b in fi]),
         ("p", "**LSTM (added after the baselines).** Same inputs as Ridge (30 days of anomaly at one grid point); one LSTM per region, 3 seeds, protocol tagged before the test and run once. "
               "RMSE in °C (skill vs Ridge, mean ± std over seeds; validation in brackets)."),
         ("table", ["LSTM", "Kuroshio 1 d", "Kuroshio 7 d", "Kuroshio 30 d", "Subtropical 1 d", "Subtropical 7 d", "Subtropical 30 d"],
          [["RMSE (skill vs Ridge)"] + [f"{lstm_stat(D, r, h, 'rmse')[0]:.3f} ({sg(lstm_stat(D, r, h, 'skill_ridge')[0])}) [{sg(lstm_stat(D, r, h, 'skill_ridge', test=False)[0])}]" for r in D.regions for h in LEADS]]),
         ("p", lstm_facts(D)[0]),
         ("p", lstm_facts(D)[2]),
         ("p", "**B1: spatial context (added after the LSTM; own protocol, tag `pre-b1-test`, run once).** Same pairs, inner 40×40 scored; inputs from a 5×5 patch of neighbouring points (1° halo data). "
               "RMSE skill vs Ridge, test [validation]; LSTM-patch mean ± std over 3 seeds."),
         ("table", ["B1", "Kuroshio 1 d", "Kuroshio 7 d", "Kuroshio 30 d", "Subtropical 1 d", "Subtropical 7 d", "Subtropical 30 d"],
          [["Ridge-patch"] + [f"{sg(b1_rp_skill(D, r, h))} [{sg(b1_rp_skill(D, r, h, False))}]" for r in D.regions for h in LEADS],
           ["LSTM-patch"] + [f"{sg(b1_lstm(D, r, h, 'skill_ridge')[0])} [{sg(b1_lstm(D, r, h, 'skill_ridge', False)[0])}]" for r in D.regions for h in LEADS]]),
         ("p", b1_short(D)),
         ("p", "**B2: whole-region models (U-Net, ConvLSTM-ED; own protocol, tag `pre-b2-test`, run once).** Test RMSE skill vs Ridge-patch, mean ± std over 3 seeds [validation]; variant (plain/residual) chosen on validation by code."),
         ("table", ["B2", "Kuroshio 1 d", "Kuroshio 7 d", "Kuroshio 30 d", "Subtropical 1 d", "Subtropical 7 d", "Subtropical 30 d"],
          [[lab] + [f"{pm(*b2m(D, r, f, h))} [{sg(b2_val_skill(D, r, f, h))}]" for r in D.regions for h in LEADS] for f, lab in FAMS]),
         ("p", b2_short(D)),
         ("p", "**Limitations.** Two small regions; simple statistical baselines; OISST is a smooth interpolated analysis; hybrid ERDDAP + PSL data (verified on overlaps); "
               "v2/v2.1 label change at 2016; warm shift vs the 1982–2015 climatology, not detrended; Ridge ignores spatial structure; its alpha sits at the grid's lower edge in several cases; one fixed split."),
         ("p", "Code, protocols (`docs/lstm_test_protocol.md`, `docs/b1_spatial_protocol.md`, `docs/b1_test_protocol.md`, `docs/b2_protocol.md`, `docs/b2_test_protocol.md`) and data checks: `README.md`, `results/*.json`. Reference for deep learning in ocean forecasting: Hao et al. 2025 (review)."),
         ("refs", REFS),
         ("p", "*Code written with Claude Code; methodology, decisions and verification by the author.*")]
    return B


def render_report(blocks):
    import re
    md, ht = [], []
    inline = lambda s: re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s, quote=False))).replace("`", "")  # noqa: E731
    for b in blocks:
        k = b[0]
        if k == "h1":
            md.append(f"# {b[1]}\n"); ht.append(f"<h1>{html.escape(b[1])}</h1>")
        elif k == "p":
            md.append(b[1] + "\n"); ht.append(f"<p>{inline(b[1])}</p>")
        elif k == "auth":
            md.append(f"*{b[1]}*\n"); ht.append(f"<p><i>{html.escape(b[1])}</i></p>")
        elif k == "refs":
            md.append("**References**\n\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(b[1], 1)) + "\n")
            ht.append("<p><b>References</b></p><ol class='r'>" + "".join(f"<li>{html.escape(x)}</li>" for x in b[1]) + "</ol>")
        elif k == "cap":
            md.append(f"*{b[1]}*\n"); ht.append(f"<p><i>{html.escape(b[1])}</i></p>")
        elif k == "table":
            md.append(md_table(b[1], b[2]) + "\n"); ht.append(html_table(b[1], b[2]))
        elif k == "img":
            md.append(f"![{b[2]}](../{b[1]})\n"); ht.append(f'<p align="center"><img src="../{b[1]}" width="{IMG_PX}" height="{int(IMG_PX * 0.764)}"></p>')
        elif k == "ol":
            md.append("\n".join(f"{i}. {x}" for i, x in enumerate(b[1], 1)) + "\n")
            ht.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in b[1]) + "</ol>")
    css = ("<style>@page{size:21cm 29.7cm;margin:1.0cm}body{font-family:'DejaVu Sans',Arial;font-size:6.9pt}"
           "h1{font-size:11pt;margin:0 0 3pt 0}p{margin:0 0 3pt 0}td,th{font-size:6pt}ol{margin:0 0 3pt 0}li{margin:0}ol.r li{font-size:6pt}</style>")
    return "\n".join(md), f"<html><head><meta charset='utf-8'>{css}</head><body>" + "\n".join(ht) + "</body></html>"


IMG_PX = 240


def main():
    D = Data()
    f = facts(D)
    (ROOT / "README.md").write_text(readme(D, f))
    md, ht = render_report(report_blocks(D, f))
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "report.md").write_text(md)
    (ROOT / "report" / "report.html").write_text(ht)
    print("wrote README.md, report/report.md, report/report.html")


if __name__ == "__main__":
    main()
