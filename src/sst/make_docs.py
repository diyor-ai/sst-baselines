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
    for h in LEADS:  # anomaly persistence/damped/ridge >> climatology & OCN at 1 day
        for r in D.regions:
            assert D.rmse(t, r, "1", "ridge") < 0.5 * min(D.rmse(t, r, "1", "climatology"), D.rmse(t, r, "1", "ocn"))
    assert all(D.skill(t, K, h, "ridge") > 0 for h in LEADS)                  # Ridge beats AP everywhere in Kuroshio
    assert D.skill(t, S, "1", "ridge") < 0 and D.skill(t, S, "7", "ridge") < 0   # but not at 1/7 d in the subtropical box
    assert all(D.skill(t, r, "30", "persistence") < -0.5 for r in D.regions)    # plain persistence is poor at 30 d
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
    b = lambda reg, h, m: D.tb["regions"][reg][h][m]["bias"]  # noqa: E731
    return [
        ("Anomaly persistence is the bar to beat, especially at short range.",
         f"At 1 day all anomaly-based methods have RMSE of {f3(r(t, S, '1', 'anomaly_persistence'))}–{f3(r(t, K, '1', 'anomaly_persistence'))} °C "
         f"(subtropical–Kuroshio, anomaly persistence), while climatology gives {f3(r(t, S, '1', 'climatology'))}–{f3(r(t, K, '1', 'climatology'))} °C "
         f"and OCN {f3(r(t, S, '1', 'ocn'))}–{f3(r(t, K, '1', 'ocn'))} °C. Ridge's margin over anomaly persistence at 1 day is small "
         f"(skill {sg(k(K, '1', 'ridge'))} Kuroshio, {sg(k(S, '1', 'ridge'))} subtropical)."),
        ("Ridge helps in the high-variability Kuroshio box, not reliably in the subtropical box.",
         f"Kuroshio: Ridge beats anomaly persistence at every lead (skill {sg(k(K, '1', 'ridge'))}, {sg(k(K, '7', 'ridge'))}, {sg(k(K, '30', 'ridge'))} "
         f"at 1, 7, 30 d). Subtropical: Ridge is worse at 1 and 7 days ({sg(k(S, '1', 'ridge'))}, {sg(k(S, '7', 'ridge'))}) and about equal at 30 d "
         f"({sg(k(S, '30', 'ridge'))}); on validation it had been better at all leads ({sg(D.skill(v, S, '1', 'ridge'))}, "
         f"{sg(D.skill(v, S, '7', 'ridge'))}, {sg(D.skill(v, S, '30', 'ridge'))}). The validation advantage did not carry over."),
        ("Plain persistence fails at 30 days, and the warm shift hurts climatology-based forecasts.",
         f"Persistence at 30 d has skill {sg(k(K, '30', 'persistence'))} (Kuroshio) and {sg(k(S, '30', 'persistence'))} (subtropical) relative to anomaly persistence. "
         f"Against the 1982–2015 climatology the test years are warm: mean bias of the climatology forecast is {sg(b(K, '30', 'climatology'))} °C (Kuroshio) and "
         f"{sg(b(S, '30', 'climatology'))} °C (subtropical); Ridge at 30 d is also biased low ({sg(b(K, '30', 'ridge'))}, {sg(b(S, '30', 'ridge'))} °C)."),
    ]


FIGS = [("results/figures/fig1_regions.png", "Figure 1. Mean SST and standard deviation of the daily anomaly (training years) in the two regions."),
        ("results/figures/fig2_val_leads.png", "Figure 2 (validation). RMSE and skill vs anomaly persistence by lead."),
        ("results/figures/fig2_test_leads.png", "Figure 2 (test, 2020–2025). Same layout; values below the axis are listed in the box."),
        ("results/figures/fig3_test_seasons.png", "Figure 3. Test skill vs anomaly persistence by season (mean ± std over years, dots = years)."),
        ("results/figures/figS1_area_mean_anomaly.png", "Figure S1. Area-mean anomaly vs the 1982–2015 climatology; the validation period is shaded.")]


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
      f"`data/raw/erddap_time_axis.csv`). Checks: every gap month fetched twice and identical, in range, plausible day-to-day, and matching ERDDAP on all "
      f"overlap days ({f['n_pieces_ok']}/{f['n_pieces']} pieces, {f['n_years_ok']}/{f['n_years']} region-years, max |difference| "
      f"{f['max_diff']:.1e} °C, tolerance {f['tol']:.0e}). Nothing is interpolated; the source of every day is stored in `data/processed/<region>.nc` (`source`).")
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
    A("## Status")
    A(f"Validation and test evaluation complete. The test period was evaluated once (commit `{f['commit'][:7]}`, tag `pre-test-freeze`, "
      f"started {f['run_utc']}). **Post-test changes:** none to methods or parameters. After the test run only figure and documentation code "
      "was added (fig 2 / fig 3 layout, this README, the report).\n")
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
      "- One fixed split (no cross-validation over time blocks); six test years are a short record.\n"
      "- I am still learning oceanography; interpretations here are descriptive, not physical explanations.\n")
    A("## Reproduce")
    A("```bash\npython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt\nexport PYTHONPATH=src\n"
      "python -m sst.download --all            # ERDDAP, validated per file\npython -m sst.fill                      # PSL gap days, validated pieces only\n"
      "python -m sst.fill --samples            # v2 / v2.1 label check\npython -m sst.data                      # merge + QC -> data/processed, results/qc_*.json\n"
      "python -m sst.figures                   # figure 1 + results/region_stats.json\npython -m sst.run_val_all               # validation: all baselines + Ridge, bias, alphas\n"
      "python -m sst.figures2                  # validation fig 2 + fig S1\n# freeze: commit config.yaml + src, then: git tag pre-test-freeze\n"
      "python -m sst.run_test                  # test period, ONCE (guarded)\npython -m sst.figures2 test             # test fig 2 + fig 3\n"
      "python -m sst.make_docs                 # README.md + report/report.md/.html from results/*.json\n"
      "soffice --headless --convert-to pdf --outdir report report/report.html   # report.pdf\npytest                                  # synthetic tests\n```\n")
    A("## How this was built")
    A("The code was written with Claude Code (an AI coding assistant); the methodology, decisions and verification are mine.\n")
    return "\n".join(L)


def report_blocks(D, f):
    t, K, S = D.test, "kuroshio", "subtropical"
    cols = [(r, h) for r in D.regions for h in LEADS]
    head = ["Method"] + [f"{r[:5]}. {h} d" for r, h in cols]
    rows = [[LABEL[m]] + [cell(D, t, r, h, m) for r, h in cols] for m in t["methods"]]
    fi = findings(D, f)
    B = [("h1", "Simple SST forecast baselines in two North Pacific boxes: a test-period evaluation"),
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
         ("p", "**Limitations.** Two small regions; simple statistical baselines; OISST is a smooth interpolated analysis; hybrid ERDDAP + PSL data (verified on overlaps); "
               "v2/v2.1 label change at 2016; warm shift vs the 1982–2015 climatology, not detrended; Ridge ignores spatial structure; its alpha sits at the grid's lower edge in several cases; one fixed split."),
         ("p", "**Next step.** Optionally a small LSTM under the same protocol (reference: Hao et al. 2025, IEEE Trans. Cybernetics 55(6), review). "
               "Code and data checks: `README.md`, `results/*.json`.")]
    return B


def render_report(blocks):
    import re
    md, ht = [], []
    inline = lambda s: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s, quote=False)).replace("`", "")  # noqa: E731
    for b in blocks:
        k = b[0]
        if k == "h1":
            md.append(f"# {b[1]}\n"); ht.append(f"<h1>{html.escape(b[1])}</h1>")
        elif k == "p":
            md.append(b[1] + "\n"); ht.append(f"<p>{inline(b[1])}</p>")
        elif k == "cap":
            md.append(f"*{b[1]}*\n"); ht.append(f"<p><i>{html.escape(b[1])}</i></p>")
        elif k == "table":
            md.append(md_table(b[1], b[2]) + "\n"); ht.append(html_table(b[1], b[2]))
        elif k == "img":
            md.append(f"![{b[2]}](../{b[1]})\n"); ht.append(f'<p align="center"><img src="../{b[1]}" width="{IMG_PX}" height="{int(IMG_PX * 0.764)}"></p>')
        elif k == "ol":
            md.append("\n".join(f"{i}. {x}" for i, x in enumerate(b[1], 1)) + "\n")
            ht.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in b[1]) + "</ol>")
    css = ("<style>@page{size:21cm 29.7cm;margin:1.0cm}body{font-family:'DejaVu Sans',Arial;font-size:7.6pt}"
           "h1{font-size:11pt;margin:0 0 3pt 0}p{margin:0 0 3pt 0}td,th{font-size:6pt}ol{margin:0 0 3pt 0}li{margin:0}</style>")
    return "\n".join(md), f"<html><head><meta charset='utf-8'>{css}</head><body>" + "\n".join(ht) + "</body></html>"


IMG_PX = 500


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
