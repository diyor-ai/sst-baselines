# sst-baselines

**DRAFT — not public yet.** Status: validation done (2016–2019); the test period has not been evaluated.

## Purpose
Reproducible evaluation of simple SST forecast baselines (persistence, anomaly persistence, damped persistence,
climatology, optimal climate normals, per-grid-point Ridge regression) at leads of 1, 7 and 30 days for two small
North Pacific boxes, one with high variability (Kuroshio extension, 30.125–39.875°N, 145.125–154.875°E) and one with
low variability (subtropical, 15.125–24.875°N, 150.125–159.875°E). It is a baselines study, not an ocean forecasting
model: the point is to know how hard a baseline is to beat before any learned model is compared with it.

## Data
- **Primary: ERDDAP**, dataset `ncdcOisst21Agg` (NOAA OISST v2.1 final, daily, 0.25°, variable `sst`).
  The aggregation lacks 1,194 days in 1982–2025 (1,177 in 1992–1998, 17 scattered in 2021–2025), visible in the
  server's own time axis (`data/raw/erddap_time_axis.csv`). An earlier download check ("file opens") accepted short
  files; the downloader now checks the exact day count against the server axis.
- **Gap fill: NOAA PSL OPeNDAP** (`.../Datasets/noaa.oisst.v2.highres`, `sst.day.mean.{year}.nc`), used only for
  those 1,194 days. Every gap month is fetched twice (must be identical), must be in range, plausible day to day
  (max step below the threshold in `results/psl_validation.json`) and must match ERDDAP on all overlap days
  (154/154 pieces, 24/24 region-years, max |difference| 1.9e-06 °C). Nothing is interpolated; the source of every
  day is stored in `data/processed/<region>.nc` (`source`) and `data/raw/manifest.json`.
- Result: 16,071 daily fields per region (1982-01-01 … 2025-12-31), 40×40 grid, no missing day, no land points.
- Downloaded on 2026-11-02.
- Variability (training years, anomaly vs the final harmonics3 climatology, cos(lat)-weighted; `results/region_stats.json`, key `final`):
  per-point anomaly std 1.13 °C (Kuroshio) vs 0.69 °C (subtropical); area-mean anomaly std 0.66 vs 0.52 °C;
  lag-1 autocorrelation of the area mean 0.984 vs 0.978. The earlier "preliminary" numbers (unsmoothed day-of-year mean) are kept under `preliminary`.
- Split: training 1982–2015, validation 2016–2019, test 2020–2025 (the test period is not loaded in any validation code).

## Data notes
- **1996 cold event (kept unchanged):** the record minimum of the Kuroshio box, −1.14 °C, is on 1996-03-30 at
  39.875°N, 146.375°E (PSL gap fill; the neighbouring days come from ERDDAP and agree). Not a download or processing
  artefact (two sources agree, smooth ~10 days, spatially coherent), but extreme for the region in March
  (~8 °C below the local mean) and possibly a feature of the AVHRR-only analysis of that era. It lies in the training period.
- **v2 / v2.1 label at 2016:** PSL labels its files OISST v2 up to 2015 and v2.1 from 2016, the year the validation
  period starts. Single-day samples from both label periods (1999, 2005, 2010, 2015, 2016, 2020, 2024; 8 days each)
  matched ERDDAP v2.1 to 1.9e-06 °C (`results/psl_version_samples.json`); no numerical difference was seen at 2016.

## Protocol (fixed before looking at validation scores)
- Leads h = 1, 7, 30 days; target = SST at t+h, issued on day t using data at or before t.
- Climatology: day-of-year mean over training years only, smoothed with 3 annual harmonics (Feb 29 uses Feb 28).
  Anomaly = SST − that climatology.
- Scores on cos(latitude)-weighted grid points: RMSE, MAE, ACC (primary: temporal, per grid point, then weighted
  mean; secondary: daily spatial ACC), skill = 1 − RMSE/RMSE_reference with references climatology and anomaly persistence.
  Also by season (DJF/MAM/JJA/SON) and per year. Pairs with a missing input or target day are skipped and counted.
- Ridge hyper-parameter (alpha) chosen on validation RMSE only; the final model for the test run is fitted on training
  years only with that alpha.
- Bias analysis (validation) is descriptive only; no method is changed because of it.
- The test period (2020–2025) is evaluated once, after the configuration is frozen and committed.

## Baselines
| name | forecast for t+h |
|---|---|
| persistence | SST(t) |
| climatology | climatology(t+h) |
| anomaly persistence | climatology(t+h) + anomaly(t) |
| damped persistence | climatology(t+h) + r(h)·anomaly(t), r(h) per grid point from training years |
| OCN | mean of the same calendar day over the previous 10 years (all before the issue date) |
| Ridge | climatology(t+h) + Ridge(anomaly lags 0–6, mean anomaly t−29..t); one model per grid point and lead; features standardised with training statistics; alpha from a 13-value log grid 1e-3 … 1e3, one alpha per region and lead |

Validation results: `results/val_all.json` (scores), `results/val_bias.json` (bias / MSE split, area-mean anomaly),
`results/ridge_alphas.json` (alpha curves), figures in `results/figures/`.

## Status
Validation (2016–2019) complete for all baselines and Ridge. **Test period not yet evaluated.**

## Limitations (so far)
- Two small regions, one product (OISST); results may not transfer elsewhere.
- The validation years are warmer than the 1982–2015 climatology (see `results/val_bias.json`); methods that revert
  towards the training climatology carry a negative bias there. Not corrected, by protocol.
- The Ridge alpha lies on the lower edge of the grid for several region/lead cases (the validation curve is almost flat
  there, see `results/ridge_alphas.json`); the grid was not extended.
- OISST is an analysis (interpolated from satellites and in situ data), not raw observations; its character changes over time.
- 1,194 days per region come from a second source (PSL), validated but with a v2/v2.1 label change at 2016.
- Only SST is used; no other predictors, no probabilistic scores, no uncertainty intervals yet.

## How this was built
The code was written with Claude Code (an AI coding assistant); the methodology, decisions and verification are mine.
