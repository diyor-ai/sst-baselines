"""Build report/methods_note.md (+ .html for LibreOffice -> methods_note.pdf) from results/methods_example.json
and config.yaml. Prose is fixed text; every number in the worked example comes from the JSON file.

Run:  python report/methods_example.py && python report/build_methods_note.py
      soffice --headless --convert-to pdf --outdir report report/methods_note.html
"""
import json
import re

from sst.download import ROOT, load_config

E = json.loads((ROOT / "results" / "methods_example.json").read_text())
cfg = load_config()
A = cfg["ridge"]["alphas"]
g = lambda x, n=3: f"{x:.{n}f}"  # noqa: E731
sg = lambda x, n=3: f"{x:+.{n}f}"  # noqa: E731
C, R, O, RG = E["clim"], E["r"], E["ocn"], E["ridge"]
F, ER = E["forecast"], E["error"]
b = C["coef"]
H = E["lead"]
fmt_a = lambda v: f"{v:g}"  # noqa: E731

MD = f"""# Methods note: how the six baselines are calculated

*Mukhammaddiyor Abdulazimov · Harbin Engineering University · November 2026*

Documentation only: this note describes the code as frozen (git tag `pre-test-freeze`); no method, parameter or result is changed. Function names refer to `src/sst/`.

## 1. Notation

- **s(t, i)**: daily OISST v2.1 sea-surface temperature (°C) on day *t* at grid point *i* (0.25° grid; two 40×40 regions, Kuroshio 30.125–39.875°N, 145.125–154.875°E and subtropical 15.125–24.875°N, 150.125–159.875°E). Both boxes are open ocean (no NaN points; the code would skip NaN points).
- **t** is the issue date (the last day whose SST is known), **h** the lead (1, 7, 30 days), **t+h** the target date. A forecast issued at *t* uses SST only on days ≤ *t*.
- **d(t)** ∈ {{1,…,365}} is the calendar day in a 365-day calendar: 1 Jan = 1, 31 Dec = 365, and 29 Feb is mapped to 28 Feb (`climatology.doy365`).
- Periods: training 1982–2015, validation 2016–2019, test 2020–2025. Everything fitted (climatology, *r*(h), Ridge) uses training years only.
- Every baseline is computed separately at each grid point; the subscript *i* is dropped below when it is not needed.

## 2. Climatology and anomaly

**Raw day-of-year mean** (`climatology.raw_doy_mean`): for each calendar day *d*, the average of s over all training days with d(t) = *d* (34 years; 29 Feb observations are pooled into 28 Feb, so 28 Feb averages 42 values (34 + 8 leap days), other days 34; the regions have no missing days, and the code skips them if present):
c<sub>raw</sub>(d, i) = mean{{ s(t, i) : t in 1982–2015, d(t) = d }}.

**Smoothed climatology** (`climatology.smooth_harmonics`, `build_climatology`, config `smoothing: harmonics3`): at each grid point, c(·, i) is the least-squares fit (`numpy.linalg.lstsq`) of the 365 raw values by a constant and three annual harmonics,

c(d, i) = β<sub>0</sub> + Σ<sub>k=1..3</sub> [ β<sub>k,c</sub> cos(2πk(d−1)/365) + β<sub>k,s</sub> sin(2πk(d−1)/365) ].

Seven coefficients per grid point, fitted on the training raw means only (equal weights; the angle uses d−1, i.e. the 0-based index used in the code). For the grid point of §5 ({g(E['lat'],3)}°N, {g(E['lon'],3)}°E): β<sub>0</sub> = {g(b[0],4)}, (β<sub>1,c</sub>, β<sub>1,s</sub>) = ({g(b[1],4)}, {g(b[2],4)}), (β<sub>2,c</sub>, β<sub>2,s</sub>) = ({g(b[3],4)}, {g(b[4],4)}), (β<sub>3,c</sub>, β<sub>3,s</sub>) = ({g(b[5],4)}, {g(b[6],4)}) °C; the RMS difference between raw and fitted values over the 365 days is {g(C['rms_raw_minus_fit'])} °C. The raw mean is only an intermediate; every forecast and anomaly uses the smoothed c. `clim_at` looks c up for any date (any year, including validation and test) through d(t).

**Anomaly** (`climatology.anomaly`): a(t, i) = s(t, i) − c(d(t), i). The climatology is never refitted on later years, so a warm shift after 2015 stays in the anomaly.

## 3. The six baselines

Forecasts for target *t+h*, written ŝ(t+h). All are evaluated on the same (issue, target) pairs (`metrics.valid_pairs`; a pair is skipped and counted if the issue or target day is missing; `run_val_all.run_region` asserts that Ridge uses identical pairs).

**(1) Persistence** (`baselines.persistence`): ŝ(t+h) = s(t). No climatology, no parameters.

**(2) Fixed smoothed climatology** (`baselines.climatology_forecast`; called "Climatology" in tables and figures): ŝ(t+h) = c(d(t+h)), the 1982–2015 3-harmonic fit of §2. It ignores s(t) and gives the same value for every year and every lead at a given target day.

**(3) Anomaly persistence** (`baselines.anomaly_persistence`): ŝ(t+h) = c(d(t+h)) + a(t). Today's anomaly is kept unchanged (equivalent to s(t) + [c(d(t+h)) − c(d(t))]).

**(4) Damped persistence** (`baselines.lag_corr`, `damped_persistence`): ŝ(t+h) = c(d(t+h)) + r<sub>i</sub>(h)·a(t). One coefficient per grid point and lead:
r<sub>i</sub>(h) = Σ a(t,i)·a(t+h,i) / √( Σ a(t,i)² · Σ a(t+h,i)² ),
summed over all pairs (t, t+h) in which **both** days lie in the training years 1982–2015 (all seasons pooled; for this example {R['n_pairs']:,} pairs). It is **not centred**: no mean is subtracted (the training-anomaly means are ≈ 0, e.g. {sg(R['mean_x'],4)} °C here, so it is almost the Pearson correlation). Validation and test data are never used; the same r<sub>i</sub>(h) is applied in every later year. It is a correlation, not the regression slope Σ a(t)a(t+h) / Σ a(t)²; the two are close because the variances of a(t) and a(t+h) are nearly equal (here {g(R['sxx'],0)} vs {g(R['syy'],0)}).

**(5) 10-year daily climatology, "OCN"** (`baselines.ocn`, `ocn_years: 10`): for target date T = t+h,
ŝ(t+h) = (1/10) Σ<sub>k=1..10</sub> s(T − k years),
i.e. the mean of the **same calendar day as the target** in the 10 years before the **target** year (T minus 1, 2, …, 10 years; the window is anchored on the target, not on the issue date or the end of training). If T is 29 Feb, 28 Feb is used first and then shifted back, so 29 Feb of earlier leap years is never read. The single-day values are averaged as they are: no smoothing, no anomalies, no climatology, no training split; missing days are skipped by `nanmean`. All ten dates lie at least one year before T; the code raises a `ValueError` unless every date is strictly before the issue date (true for h < 365 d), so the baseline uses only information available at *t*. The window moves with the target (e.g. for 2021 targets it covers 2011–2020), so it can contain validation/test years that precede the issue date.

**(6) Ridge** (`ridge.make_features`, `fit_ridge`, `predict_ridge`, `ridge_lead`): one linear model per grid point and per lead.
- Features at issue date t (8): the anomalies a(t), a(t−1), …, a(t−6) (lags 0–6) and the mean anomaly ā<sub>30</sub>(t) = (1/30) Σ<sub>j=0..29</sub> a(t−j) over t−29..t (anomalies relative to the training climatology). A pair needs all 30 days present.
- Target: the anomaly y = a(t+h) = s(t+h) − c(d(t+h)).
- Training sample: all issue dates whose target lies in 1982–2015 and whose 30-day history is complete (`_pairs`; {fmt_a(RG['n_train_rows'])} rows for h = {H} at the §5 point). Features read only days ≤ t.
- Standardisation: each feature is z = (x − μ)/σ, where μ and σ (population std) are the mean and standard deviation of that feature over the training rows of that grid point and lead; the target is centred by its training mean ȳ (the intercept, not penalised).
- Fit: w = argmin (1/n) Σ (y − ȳ − zᵀw)² + α‖w‖², solved in closed form as w = (ZᵀZ/n + αI)⁻¹ Zᵀ(y − ȳ)/n.
- α: grid 10<sup>−3</sup>, 10<sup>−2.5</sup>, …, 10<sup>3</sup> (13 values). For each region and lead, one α (shared by all grid points of the region) is chosen as the value with the lowest cos(lat)-weighted RMSE of the forecast SST on validation 2016–2019 (`ridge_lead`, validation branch). Chosen values (config.yaml, `results/ridge_alphas.json`): Kuroshio {fmt_a(A['kuroshio'][1])} / {fmt_a(A['kuroshio'][7])} / {fmt_a(A['kuroshio'][30])}, subtropical {fmt_a(A['subtropical'][1])} / {fmt_a(A['subtropical'][7])} / {fmt_a(A['subtropical'][30])} for h = 1 / 7 / 30 d; several lie at the lower edge of the grid. The final model is fitted on **training years only** with this α (no refit on validation) and applied unchanged to the test period.
- Forecast: ŝ(t+h) = c(d(t+h)) + ŷ, with ŷ = ȳ + Σ<sub>m=1..8</sub> w<sub>m</sub> z<sub>m</sub>.

## 4. Scores

With weights w<sub>j</sub> = cos φ<sub>j</sub> / mean<sub>j</sub>(cos φ<sub>j</sub>) for latitude φ<sub>j</sub> (`metrics.lat_weights`), e = ŝ − s over all evaluated pairs and valid grid points:
- **RMSE** = √( Σ w·e² / Σ w ); **MAE** = Σ w·|e| / Σ w (`rmse`, `mae`; one pooled number per region, lead and method).
- **ACC** (temporal, primary; `metrics.acc`): with forecast anomaly f′ = ŝ − c(d(t+h)) and observed anomaly o′ = s(t+h) − c(d(t+h)), each centred by its own mean over the evaluated target dates at that grid point, r<sub>i</sub> = Σ<sub>t</sub> f′o′ / √(Σ f′² Σ o′²); then ACC = Σ<sub>i</sub> w<sub>i</sub> r<sub>i</sub> / Σ<sub>i</sub> w<sub>i</sub> over points where r<sub>i</sub> is defined (zero-variance points, e.g. the climatology forecast, are skipped, so ACC is undefined for it). A secondary daily spatial ACC (`acc_spatial`) is also stored.
- **Skill** (`metrics.skill_score`) = 1 − RMSE<sub>model</sub> / RMSE<sub>ref</sub>; primary reference anomaly persistence, also computed against the fixed climatology (0 = no better than the reference, 1 = perfect, negative = worse).

## 5. Worked example (real data, lead h = {H} d)

Grid point {g(E['lat'],3)}°N, {g(E['lon'],3)}°E (Kuroshio region). Issue date t = {E['issue']} (test period), target t+h = {E['target']}. Values are computed at full precision and displayed to 3 decimals; errors are forecast − observed. Nothing is refitted: the climatology, r(h) and Ridge model are the training-year objects used in the test run.

| Quantity | Value |
|---|---|
| s(t), {E['issue']} | {g(E['s_t'])} °C |
| d(t), d(t+h) | {C['d_issue']}, {C['d_target']} |
| c(d(t)) (raw mean {g(C['raw_issue'])}) | {g(C['c_issue'])} °C |
| c(d(t+h)) (raw mean {g(C['raw_target'])}) | {g(C['c_target'])} °C |
| a(t) = s(t) − c(d(t)) | {sg(E['a_t'])} °C |
| r(h) = {g(R['num'],1)} / √({g(R['sxx'],1)} · {g(R['syy'],1)}), {R['n_pairs']:,} training pairs | {g(R['r'],4)} |
| **Observed s(t+h)**, {E['target']} | **{g(E['s_obs'])} °C** |

**OCN** (same calendar day as the target, 22 March, 10 years before 2021; latest date {O['latest_date']}, {O['gap_to_issue_days']} days before the issue date):

| Year | {" | ".join(d[:4] for d in O['dates'])} | Mean |
|---|{"---|" * 11}
| s (°C) | {" | ".join(g(v, 2) for v in O['values'])} | **{g(O['mean'])}** |

**Ridge** (α = {fmt_a(RG['alpha'])}, intercept ȳ = {g(RG['intercept'],4)} °C):

| Feature | {" | ".join(RG['features'])} |
|---|{"---|" * 8}
| x (°C) | {" | ".join(g(v) for v in RG['x'])} |
| μ (train) | {" | ".join(g(v,4) for v in RG["mu"])} |
| σ (train) | {" | ".join(g(v) for v in RG['sd'])} |
| z = (x−μ)/σ | {" | ".join(g(v) for v in RG['z'])} |
| w | {" | ".join(g(v) for v in RG['w'])} |
| w·z | {" | ".join(g(v) for v in RG['terms'])} |

ŷ = ȳ + Σ w·z = {g(RG['intercept'],4)} + {g(sum(RG['terms']),4)} = {sg(RG['a_hat'])} °C.

**The six forecasts** (target observed {g(E['s_obs'])} °C):

| Baseline | Formula with numbers | Forecast (°C) | Error (°C) |
|---|---|---|---|
| Persistence | s(t) | {g(F['persistence'])} | {sg(ER['persistence'])} |
| Fixed smoothed climatology | c(d(t+h)) = {g(C['c_target'])} | {g(F['climatology'])} | {sg(ER['climatology'])} |
| Anomaly persistence | {g(C['c_target'])} + {g(E['a_t'])} | {g(F['anomaly_persistence'])} | {sg(ER['anomaly_persistence'])} |
| Damped persistence | {g(C['c_target'])} + {g(R['r'],4)} · {g(E['a_t'])} | {g(F['damped_persistence'])} | {sg(ER['damped_persistence'])} |
| OCN (10-yr daily climatology) | mean of the 10 values above | {g(F['ocn'])} | {sg(ER['ocn'])} |
| Ridge | {g(C['c_target'])} + {g(RG['a_hat'],4)} | {g(F['ridge'])} | {sg(ER['ridge'])} |

On this date SST fell by {g(E['s_t'] - E['s_obs'], 2)} °C within the week (a cold event), so every method overshoots; the methods that ignore today's warm anomaly (fixed climatology, OCN) happen to be closest. This is a single illustrative case, not a measure of skill. Reproduce: `python report/methods_example.py` recomputes each step by hand-written arithmetic, asserts agreement with the project functions (and with the full-region `ridge_lead` output) and writes `results/methods_example.json`; this note is generated from that file by `report/build_methods_note.py`.

## 6. Naming note

In oceanography a "daily climatology" normally means the mean of each calendar day over a set of years. In this project, **OCN** is such a daily climatology computed over a **moving 10-year window** ending the year before the target (raw, unsmoothed, anchored on the target date), whereas the **fixed smoothed climatology** (called "Climatology" in the tables, figures and README) is the 3-harmonic fit to the 1982–2015 day-of-year means. The two differ in window (moving 10 years vs fixed 34 training years) and in smoothing (none vs three harmonics). This note uses these two names consistently; no names have been changed in the code or README.
"""


def inline(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", s)
    return re.sub(r"`([^`]+)`", r"<tt>\1</tt>", s)


def to_html(md):
    out, rows, items = [], [], []
    flush_t = lambda: (out.append("<table border='1' cellspacing='0' cellpadding='2'>" + "".join(rows) + "</table>"), rows.clear()) if rows else None  # noqa: E731
    flush_l = lambda: (out.append("<ul>" + "".join(items) + "</ul>"), items.clear()) if items else None  # noqa: E731
    for line in md.split("\n"):
        if line.startswith("|"):
            flush_l()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            tag = "th" if not rows else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        flush_t()
        if line.startswith("- "):
            items.append(f"<li>{inline(line[2:])}</li>")
            continue
        flush_l()
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush_t(); flush_l()
    css = ("<style>@page{size:21cm 29.7cm;margin:1.3cm}body{font-family:'DejaVu Sans',Arial;font-size:8.4pt}"
           "h1{font-size:13pt;margin:0 0 2pt 0}h2{font-size:10pt;margin:5pt 0 2pt 0}p{margin:0 0 3pt 0}"
           "ul{margin:0 0 3pt 0}li{margin:0}td,th{font-size:7.6pt}tt{font-family:'DejaVu Sans Mono';font-size:8pt}</style>")
    return f"<html><head><meta charset='utf-8'>{css}</head><body>" + "\n".join(out) + "</body></html>"


if __name__ == "__main__":
    (ROOT / "report" / "methods_note.md").write_text(MD)
    (ROOT / "report" / "methods_note.html").write_text(to_html(MD))
    print("wrote report/methods_note.md, report/methods_note.html")
