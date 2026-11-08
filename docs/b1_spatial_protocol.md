# B1 "spatial context" (approved by Teacher Lu) - Part 1: halo data + models + VALIDATION ONLY
Written 2026-11-08 BEFORE any halo download, any new model code or any new result. Test period (2020-2025) is not touched by
any model, statistic or fit; STOP after the validation report. Authoritative copy committed as docs/b1_spatial_protocol.md.

## Question
Until now each forecast used only its own grid point's history. Does adding neighbouring points help, and how much of any LSTM gain
is the extra inputs vs. the nonlinear model?

## 1. Halo data
- 1 deg halo (4 points per side) around both boxes -> 48x48. kuroshio_halo: lat 29.125..40.875, lon 144.125..155.875;
  subtropical_halo: lat 14.125..25.875, lon 149.125..160.875 (config_b1.yaml; config.yaml is NOT edited).
- Same hybrid pipeline and checks as the 40x40 data (ERDDAP primary; PSL gap fill, each gap month fetched twice sequentially, one worker,
  range / day-to-day step / NaN-only-on-land / exact match with ERDDAP on overlap days; year-level gate). New files only
  (data/raw/{erddap,psl}/<region>_halo, data/processed/<region>_halo.nc, results/qc_<region>_halo.json, results/halo_psl_validation.json,
  data/raw/manifest_halo.json). Existing 40x40 files and data/raw/manifest.json are not written.
- Day count must be 16,071 per region. Assert: the inner 40x40 of the halo files == existing processed files exactly (sst bit-for-bit incl. NaN pattern, source flags, coords).
- Size estimate before download: ~133 MB raw per box (92 MB x 48^2/40^2), ~270 MB total < 500 MB, so no stop-and-ask.
- NaN / land points in the halo are counted and reported. IF ANY halo point is NaN: STOP and ask before modelling (pairs would differ from the existing validation).
- Anomalies of halo points: harmonics3 climatology fitted on 1982-2015 only (existing code, per point). ONE standardising std per region = the std of
  the TRAINING anomalies of the inner 40x40, unchanged (asserted equal to the existing LSTM std). Halo anomalies are divided by that same number.

## 2. Models (fixed now)
Common: targets and evaluation on the inner 40x40 only; pairs = exactly the existing LSTM validation samples (issue date in 2016-2019, all
three targets t+1, t+7, t+30 in 2016-2019, 30-day history complete; history may reach into 2015). Forecast SST = clim(t+h) + predicted anomaly. Fit data: targets <= 2015-12-31.
- Patch: 5x5 centred on the target point (centre = target point). Edge points use the halo (patch of an edge point reaches 2 halo points out; the other 2 halo rings are unused by the 5x5 patch).
- Ridge+doy: existing Ridge features (anomaly lags 0..6 + mean t-29..t, own point; 8) + sin/cos of the day of year of the ISSUE day t (10 features).
  One model per point and lead; same training-pair rule as the existing Ridge; standardised with training stats.
- Ridge-patch: for each of the 25 patch points lags 0..6 + 30-day mean (8) = 200 features + sin/cos doy of day t = 202. One model per point and lead,
  standardised with training mean/std. Same penalty convention as ridge.py (MSE + alpha*|w|^2 on standardised features).
- Alpha (both new ridges): same 13-value grid 1e-3..1e3; one alpha per region and lead, chosen on VALIDATION RMSE (cos-lat weighted, SST, inner 40x40) over the evaluation pairs above; ties impossible in practice. No refit on validation. Grid-edge choices are flagged.
  (The existing Ridge keeps its alphas, chosen on its own frozen pair set; it is re-scored here on the LSTM pairs as in lstm_eval.py.)
- LSTM-patch: same as the existing LSTM, but each day's input = 25 patch anomalies + sin/cos doy of that day = (30 x 27). One model per region, H=64 (chosen earlier; NO new search),
  Adam lr 1e-3, batch 1024, 200k random (point, issue-date) pairs per epoch, max 30 epochs, patience 5 on validation MSE (standardised, mean over 3 leads), best-val weights kept, seeds 0,1,2. 6 runs.
  Per-run seeds/generators/deterministic algorithms exactly as lstm_train.py. The training loop is a copy (lstm_train.py is frozen and not edited); a test checks that with a 1x1 patch and no halo it reproduces lstm_train.train_run (same loss history).
- Existing LSTM (point) results (results/lstm_val.json, H=64, seeds 0,1,2) are reused as they are; not retrained.
- Seed handling: report mean +- sample std (ddof=1) over seeds of the PER-SEED scores. No seed picking, no ensembling.

## 3. Rules
No sudo; work only in this project; never delete log/result lines (append dated corrections); new results in new files; frozen results and tags untouched.
Manifest for every run (commit, settings, sha256 of source/config/data files, weights_sha256). Code committed locally before training; no push.
4 parallel jobs OK, timing is not a result. Keep the laptop awake (systemd-inhibit).
Tests: no future inputs; no training target after 2015-12-31; patch indexing (centre = target, right neighbours, edge uses halo); inner 40x40 unchanged; same seed -> same loss.

## 4. Validation report (then STOP)
Per region and lead: RMSE and ACC (temporal) for anomaly persistence, Ridge, Ridge+doy, Ridge-patch, LSTM (point), LSTM-patch (mean +- std over seeds), plus RMSE skill vs Ridge.
Three questions (answered in 3-4 lines): (a) does doy alone help Ridge (Ridge+doy vs Ridge)? (b) does the patch help the linear model (Ridge-patch vs Ridge+doy; Ridge-patch vs Ridge)?
(c) does LSTM-patch beat Ridge-patch (nonlinear gain with inputs held equal)? Plus season split (2 lines), learning curves (best epochs, overfitting), alphas chosen.
Only numbers produced by code that was run.
