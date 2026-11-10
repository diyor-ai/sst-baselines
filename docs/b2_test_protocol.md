# B2 (whole-region models) test protocol (pre-registered)

Written 2026-11-11 and committed **before** the residual runs are trained, before the tag `pre-b2-test`, and before any test-period (2020–2025) data was loaded by any B2 model.
Companions: [`docs/b2_protocol.md`](b2_protocol.md) (models, validation, residual variant), [`docs/b1_test_protocol.md`](b1_test_protocol.md) (the B1 test this one follows).
Question: does a model that sees the whole 48×48 box at once (U-Net, ConvLSTM encoder-forecaster) beat the patch models (Ridge-patch, LSTM-patch) on the untouched test years?
A learning experiment, not an ocean forecasting model.

## 1. Which models go to the test (decided by code, not by hand)
- Families: **U-Net** (C=32) and **ConvLSTM-ED** (C=16), per region (Kuroshio, subtropical). C is the one already chosen on validation (`results/b2_val_report.md`, correction 2026-11-10); it is not re-searched.
- Variant per region and family: **plain or residual**, whichever has the **lowest mean validation RMSE of SST over seeds 0, 1, 2 and leads 1, 7, 30** (cos-lat weighted, the same
  metric code; ties go to plain). The rule is applied by code to `results/b2_val_res.json` (key `variant_selection`) and written to `results/b2_test_choice.json`, which is committed before the tag.
  The test runner reads that file, re-derives the choice from `results/b2_val_res.json` and aborts if the two differ.
- So the test contains 2 regions × 2 families × 3 seeds = **12 existing checkpoints** (`models/b2_<model>_<region>_C<C>_s<seed>.pt`). The variant not chosen is **not** evaluated on test.

## 2. Test pairs and data
Exactly the pairs of the frozen baselines (as in the B1 test): for each lead h in {1, 7, 30}, every target day of 2020-01-01 … 2025-12-31, issue date = target − h; the 30-day input
history may reach back into 2019 (past data). Evaluation on the inner 40×40 only (the 48×48 output is cropped to [4:44, 4:44]). Climatology (harmonics3, 1982–2015), the single regional std and
the anomalies are those of training (unchanged). Forecast SST = clim(t+h) + predicted anomaly × std, scored with the existing metric code.
*Convention note (stated, not corrected):* validation was scored on the LSTM's own pairs (issue dates inside 2016–2019 with t+30 ≤ 2019-12-31); the test uses the baselines' pairs (a few more issue dates at h = 1 and 7).

## 3. What is reported
Per region and lead, overall and by season (DJF, MAM, JJA, SON): RMSE, MAE, ACC (temporal, primary), ACC (spatial, secondary) for anomaly persistence, Ridge, Ridge-patch, LSTM-patch,
U-Net and ConvLSTM-ED; skill (1 − RMSE/RMSE_ref) vs anomaly persistence, vs Ridge, vs Ridge-patch and vs LSTM-patch. Deep models (LSTM-patch, U-Net, ConvLSTM-ED): mean ± sample std (ddof = 1) over seeds 0, 1, 2 of the
**per-seed** scores (skill relative to the LSTM-patch uses the seed-mean LSTM-patch score block as reference). Output: `results/b2_test.json`; a markdown table report is generated from that JSON by code.
Results are reported as they come, including if a whole-region model loses to Ridge-patch, LSTM-patch or persistence.

## 4. Verification inside the runner (abort on any difference)
- `weights_sha256` of each of the 12 checkpoints equals `results/b2_manifest.json` (plain runs) / `results/b2_manifest_res.json` (residual runs). The sha256 of the data files recorded in the manifests must also match;
  differences in the sha256 of source/config files since training are **listed** in the output (the residual code legitimately changed `b2_models.py`, `config_b2.yaml`, tests) but are not by themselves a failure.
- The checkpoints, loaded in the runner, reproduce the validation RMSE of the chosen variants in `results/b2_val_res.json` (per seed, leads 1/7/30; tolerance 1e-4: different thread counts can change floating-point summation order).
- Anomaly persistence and the point Ridge recomputed on the test pairs equal `results/test_all.json` to 1e-9; Ridge-patch and the three LSTM-patch seeds (existing B1 checkpoints, hashes checked against `results/b1_manifest.json`)
  recomputed on the test pairs equal `results/b1_test.json` to 1e-9. The B1 numbers stay frozen; nothing of B1 is rewritten.

## 5. Not allowed
Retraining, refitting or fine-tuning (including on train + validation), any change of C, variant, epochs/early stopping, standardisation or climatology, choosing seeds, ensembles or averaged predictions,
test-time augmentation, evaluating the unchosen variant, any second test run, any change of the choice after seeing a test number.

## 6. One run
From the tag `pre-b2-test`: `python -m sst.b2_test`. It refuses to run unless the tag exists, HEAD is the tagged commit, the working tree is clean, `results/b2_test_choice.json` is the committed one and
`results/b2_test.json` does not exist (the test years are evaluated exactly once). A dry run (`--dry-run OUT.json`, test split := validation years, data cut at 2019-12-31, output outside the repository) is allowed before the tag.
After the run the README (via `make_docs`, numbers asserted against the JSON files) and `report.pdf` are updated once.

## 7. Contrasts asked of the test (fixed now)
(a) Do the whole-region models beat Ridge-patch? (b) Do they beat LSTM-patch? (c) U-Net vs ConvLSTM-ED. (d) Kuroshio vs subtropical. (e) Do the validation conclusions hold
(`results/b2_val_report.md`, `results/b2_val_res_report.md`)? Residual vs plain is answered on **validation only** (the unchosen variant is not run on test).

## 8. Caveats stated now
- Validation numbers are slightly optimistic: C, the variant and early stopping were all selected on the validation years (2016–2019) that the validation tables are scored on.
- ~12k training samples (one per issue date; neighbouring days are strongly correlated) for networks with 4×10⁴ (ConvLSTM-ED) to 2×10⁶ (U-Net) parameters; the validation curves show early overfitting (best epoch 2–8 in the plain runs).
- The std over three seeds is a rough measure of training noise only; it does not include the sampling uncertainty of 6 test years.
- Earlier tests showed validation gains shrinking on test (B1 protocol, caveats): the same may happen here.
