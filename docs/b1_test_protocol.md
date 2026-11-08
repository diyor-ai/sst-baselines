# B1 (spatial context) test protocol (pre-registered)

Written and committed **before** the tag `pre-b1-test` and before any test-period (2020–2025) data was loaded by any B1 model.
Companion: [`docs/b1_spatial_protocol.md`](b1_spatial_protocol.md) (models, halo data, validation; approved by Teacher Lu).
Question: does adding the neighbouring grid points (5×5 patch) help a linear model, and does an LSTM beat a linear model when both
get the same inputs? A learning experiment, not an ocean forecasting model.

## Protocol
1. **Models on test** (per region; all fitted on 1982–2015 only, nothing is refitted):
   - **Ridge+doy** and **Ridge-patch**, with the alphas chosen on validation (one per region and lead; stored in
     `results/b1_ridge_val.json`, copied into `results/b1_manifest.json`; the runner checks they are equal). Features, standardisation
     with training mean/std and the training rows are exactly those of the validation run.
   - **LSTM-patch**, H = 64, seeds 0, 1, 2: the six existing checkpoints (`models/b1_<region>_patch_H64_s<seed>.pt`). The
     `weights_sha256` of each is verified against `results/b1_manifest.json` before predicting; the sha256 of the source/config/data
     files recorded in the manifest are checked too (the runner aborts on any difference).
   - References re-used from the frozen work: anomaly persistence and the point **Ridge** (frozen alphas of `config.yaml`) are
     recomputed on the test pairs and must equal `results/test_all.json` to 1e-9 (the run aborts otherwise). **LSTM (point)** is *not*
     re-run; its test numbers stay those of `results/lstm_test.json` (same pairs).
2. **Test pairs.** Exactly the pairs of the frozen baselines: for each lead h, every target day of 2020-01-01 … 2025-12-31, issue date =
   target − h; the 30-day input history may reach back into 2019 (past data, allowed). Targets and evaluation on the inner 40×40 only;
   halo points are inputs only. Climatology (harmonics3) and the standardising std are the training-years (1982–2015) ones, unchanged.
   *Note:* validation was scored on the LSTM's own pairs (1,431 issue dates inside 2016–2019); the test uses the baselines' pairs
   (a few more issue dates for h = 1 and 7). This convention difference is stated, not corrected.
3. **Reported** per region and lead: RMSE, MAE, ACC (temporal; primary), ACC (spatial; secondary), overall and by season; skill
   (1 − RMSE/RMSE_ref) vs anomaly persistence, vs Ridge and vs Ridge-patch (`results/b1_test.json`). LSTM-patch as mean ± std
   (ddof = 1) over seeds 0, 1, 2 of the **per-seed** scores.
4. **Not allowed:** retraining or refitting on train + validation, any change of alpha, H, epochs/early stopping, patch size,
   standardisation or climatology, seed picking, ensembles or averaged predictions, any second test run.
5. **One run**, from the tag `pre-b1-test`: `python -m sst.b1_test`. It refuses to run unless the tag exists, HEAD is the tagged commit,
   the working tree is clean and `results/b1_test.json` does not exist. A dry run on validation data (`--dry-run`, test split pointed at
   the validation years, data cut at 2019-12-31, output outside the repository) is allowed before the tag. Results are reported as they
   come, including if a patch model loses to Ridge or persistence.
6. After the run the README (via `make_docs`, numbers asserted against the JSON files) and the report are updated once; the caveats
   below are reported alongside.

## Caveats stated now (known before the test)
- **Alphas at the lower edge of the grid.** The point Ridge and Ridge+doy alphas are 1e-3 (the grid minimum) at 7 and 30 days in both regions.
  The validation curve is flat there: RMSE varies by at most 0.003 °C (7 d) and 0.0006 °C (30 d) over the four lowest alphas
  (`results/b1_ridge_val.json`, `results/ridge_alphas.json`). The grid is **not** extended and the alphas are **not** changed.
  The Ridge-patch alphas are interior at every lead.
- **Validation numbers are slightly optimistic.** The new ridge alphas and the LSTM early stopping were both selected on the validation
  years (2016–2019) that the validation tables are scored on (the same for every method). The 1-day differences on validation were
  at noise level.
- **Old point-LSTM gains shrank or vanished on test.** On the frozen LSTM test, skill vs Ridge was −0.061/−0.017/−0.001 (Kuroshio) and
  +0.029/+0.025/+0.022 (subtropical) at 1/7/30 days, against validation +0.006/+0.014 (Kuroshio) and +0.026/+0.045 (subtropical) at 7/30 days.
  The validation gains of the patch models may shrink in the same way; the test period may behave differently (Ridge was already worse than
  anomaly persistence at 1 and 7 days in the subtropical region there).
- **Doy.** On validation Ridge+doy was identical to Ridge to three decimals (skill ≈ 0), so the day of year alone was not the reason for the
  earlier point-LSTM gain; the other input difference (lags 7–29 beyond the 30-day mean) remains confounded with nonlinearity.
- **Selection of reported contrasts.** The questions asked of the test are fixed now: (a) does the linear patch gain
  (Ridge-patch vs Ridge+doy / Ridge) hold; (b) does LSTM-patch beat Ridge-patch (nonlinear gain, inputs held equal); (c) do the
  validation gains hold.

## Context known before the test (validation, `results/b1_val.json`; RMSE skill vs Ridge, 1 / 7 / 30 d)
Kuroshio: Ridge-patch +0.001/+0.020/+0.012; LSTM-patch −0.010/+0.022/+0.026. Subtropical: Ridge-patch +0.000/+0.031/+0.025;
LSTM-patch −0.018/+0.062/+0.082. LSTM-patch vs Ridge-patch: Kuroshio −0.011/+0.002/+0.014; subtropical −0.018/+0.033/+0.058.
