# LSTM test protocol (pre-registered)

Written and committed **before** the tag `pre-lstm-test` and before any test-period (2020–2025) data was loaded for the LSTM.
Research question: given the same information as Ridge (the last 30 days of anomaly at one grid point), does a small
nonlinear model (LSTM) forecast better than Ridge and anomaly persistence? A learning experiment, not an ocean forecasting model.

## Protocol
1. **Models.** The six already-trained checkpoints with hidden size H = 64 (seeds 0, 1, 2) per region. H = 64 was chosen on
   validation (rule in `config.yaml`, result in `results/lstm_val.json`). H = 32 models are not evaluated on test. Each checkpoint's
   `weights_sha256` is verified against `results/lstm_manifest.json` before predicting.
2. **Test pairs.** Exactly the pairs of the frozen baselines (`results/test_all.json`): for each lead h, every target day of
   2020-01-01 … 2025-12-31, issue date = target − h (the 30-day input history may reach back into 2019: past data, allowed).
   Anomaly persistence and Ridge (frozen, fitted on 1982–2015 with the validation-chosen alphas) are recomputed on these pairs and
   must equal the frozen `test_all.json` numbers (the run aborts otherwise). *Note:* validation was scored on the LSTM's own
   pairs (issue date in 2016–2019 and all three targets inside 2016–2019, 1,431 dates), the test uses the baselines' pairs
   (≈ 29 more issue dates for h = 1 and 7); this convention difference is stated, not corrected.
3. **Reported** per region and lead: RMSE, MAE, ACC (temporal), skill (1 − RMSE/RMSE_ref) vs anomaly persistence and vs Ridge,
   overall and by season; mean ± std (ddof = 1) over seeds 0, 1, 2 of the **per-seed** scores.
4. **Not allowed:** seed picking, ensembles or averaged predictions, retraining or refitting on train + validation, any change of H,
   early stopping, standardisation or climatology (these stay the training-years 1982–2015 ones), any second test run.
5. **One run**, from the tag `pre-lstm-test` (`python -m sst.lstm_test`; refuses to run if HEAD is not the tagged commit, the
   tree is dirty, or `results/lstm_test.json` exists). Results are reported as they come, including if the LSTM loses to
   Ridge or persistence.
6. The fairness caveat below is reported alongside the results.

## Fairness caveat
The inputs differ: the LSTM gets the full 30-day anomaly sequence plus sin/cos of the day of year of every input day; Ridge gets
anomaly lags 0..6 plus the mean anomaly of t−29..t, and **no** day of year. Part of any LSTM gain may therefore come from the extra
inputs (seasonal information, lags 7–29) and not from the nonlinear model. No experiment was run to separate the two.

## Context known before the test (validation, H = 64; `results/lstm_val.json`)
Skill vs Ridge was ≈ 0 at 1 day (−0.001 Kuroshio, −0.007 subtropical) and positive at 7 and 30 days (+0.006/+0.014 Kuroshio,
+0.026/+0.045 subtropical). On the test period Ridge was already worse than anomaly persistence at 1 and 7 days in the
subtropical region (see the Day 4 notes), so the test period may behave differently.
