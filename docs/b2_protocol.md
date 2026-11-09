# B2 "whole-region models" (approved by Teacher Lu) - Part 1: models + VALIDATION ONLY
Written 2026-11-09 BEFORE any B2 code, trial or result. The test period (2020-2025) is not touched by any model, statistic or fit;
STOP after the validation report. (Copy of notes/2026-11-09.md; notes/ is gitignored, this file is the committed record.)

## Question
B1 showed neighbours help (Ridge-patch, LSTM-patch). B2 tests models that see the whole 48x48 box at once. One sample is one issue date
(one whole field), so the training set is only ~12k samples: models are kept small.

## 1. Data and samples
- Existing 48x48 halo files and anomalies (harmonics3 climatology fitted on 1982-2015; the same single regional std, taken from sst.b1_data.make_data_b1). No new downloads.
- Input for issue date t: anomaly fields t-29..t over 48x48, plus sin/cos of the day of year of each input day as constant channels (30 x 3).
- Target: inner 40x40 anomaly fields at t+1, t+7, t+30 (3 channels). Model output is 48x48; crop [4:44, 4:44] for the loss and the evaluation.
- Training samples: issue dates whose 30 input days AND all three targets lie inside 1982-01-01..2015-12-31.
- Validation / evaluation: for each lead exactly the pairs of the frozen baselines Ridge-patch and LSTM-patch (= the LSTM validation samples):
  issue date in 2016-2019 with t+30 <= 2019-12-31, history complete (may reach into 2015); target date = issue date + h.
  Early stopping uses the loss on these same validation issue dates.
- Forecast SST = clim(t+h) + predicted anomaly * std (existing climatology, existing metric code).

## 2. Models (fixed now)
- **U-Net**: input = 30 days x (anomaly, sin, cos) = 90 channels at 48x48. Three downsamplings 48 -> 24 -> 12 -> 6 (max-pool 2), channels C, 2C, 4C, 8C
  (6x6 = bottleneck). Block = conv3x3 -> **GroupNorm (8 groups)** -> ReLU, twice. Up-path: ConvTranspose2d (k=2, s=2), concat skip, block. Head: 1x1 conv -> 3 channels. Zero padding.
  GroupNorm (not BatchNorm): batch 32 and per-sample statistics, same behaviour in train and eval mode.
- **ConvLSTM**: 2 layers, hidden channels C, kernel 3x3, zero padding, zero initial states, forget-gate bias 1 (other biases 0). Per-step input = anomaly + sin/cos maps (3 channels);
  30 steps; a 1x1 conv on the last hidden state of layer 2 gives 3 channels. (Implementation detail with identical maths: the input-to-gate conv of each layer is applied to all 30 steps at once.)
- One model per region (kuroshio, subtropical). C in {16, 32}; **chosen per model and region by the lowest mean validation RMSE of SST over seeds 0,1,2 and leads 1,7,30** (the rule used for the LSTM).
- Training: MSE over the inner 40x40 x 3 leads (standardised units); Adam lr 1e-3; batch 32; the full training set every epoch, shuffled; max 60 epochs;
  early stopping on validation loss, patience 8, best-validation weights kept; seeds 0,1,2; torch CPU, torch.use_deterministic_algorithms(True), 4 threads per job.
- First a 1-epoch trial per model/C (kuroshio, seed 0, written to results/b2_trials/, not a result). Time per epoch and the estimated total are reported;
  if the total is over ~6 hours the user is told BEFORE launching the rest.

## 3. Rules (as before)
No sudo; work only in this project; never delete log/result lines (append dated corrections); new results in new files; frozen results and tags untouched.
Manifest for every run: commit, settings, sha256 of source/config/data, weights_sha256. Code committed locally before training; no push. Laptop kept awake (systemd-inhibit).
Tests: no future inputs; no training target after 2015-12-31; the crop maps exactly onto the inner 40x40 grid (lat/lon); evaluation pairs = baseline pairs;
same seed -> same loss; a tiny batch can be overfitted.

## 4. Validation report (2016-2019), then STOP
Per region and lead: RMSE and ACC for anomaly persistence, Ridge, Ridge-patch, LSTM-patch, U-Net, ConvLSTM (deep models: mean +- sample std over seeds); skill vs Ridge-patch.
Chosen C per model and region; learning curves (best epoch, overfitting); season split in 2 lines; spatial check = map of skill vs Ridge-patch over the 40x40 at 7 and 30 days (figB2),
edge vs centre; 3-4 lines: whole-region vs Ridge-patch? U-Net vs ConvLSTM? Kuroshio vs subtropical? Only numbers produced by code that was run.

## Addendum 2026-11-09 (after the trials, on the user's instruction): U-Net runs only
- Trial results: results/b2_trial_summary.md. ConvLSTM is NOT launched (cost far above 6 h; awaiting a decision). Only the 12 U-Net runs
  (2 regions x C{16,32} x seeds 0,1,2) are started now.
- Execution: 2 jobs in parallel (one per region), 3 torch threads each = 6 threads, pinned with taskset to one hardware thread per P-core
  (CPUs 0,2,4 and 6,8,10). `--threads 3` overrides train.num_threads (4 in config_b2.yaml); the value used is recorded in every run's settings.
  Everything else unchanged. Floating-point summation order can depend on the thread count, so identical seeds are only guaranteed to
  reproduce with the same thread count.
- **Timing is not a result.** Seconds per epoch / run in the trials, logs and manifests depend on the machine load and thread settings and are
  reported only for planning; no conclusion rests on them.

## Revised ConvLSTM spec (written 2026-11-09, BEFORE any ConvLSTM training and before any ConvLSTM validation result exists) - documented deviation
Reason (timing, see results/b2_trial_summary.md): the specified ConvLSTM (30 steps at 48x48) costs about 1.4 h/epoch (C=16) and 3.0 h/epoch (C=32), i.e. >500 h
for the 24-run plan. The deviation is a change of architecture and input sampling, decided only for compute reasons. The original `convlstm` code stays in the repository, unrun.
0. Profile first (<=15 min; src/sst/b2_profile.py): time per batch with deterministic algorithms on/off and 4 vs 6 threads, with data-gather / forward / backward / optimiser
   timed separately. If something other than the model is the bottleneck it is reported. Determinism is kept unless the slowdown is >2x AND the same seed still gives the same loss without it.
1. `convlstm_ed` (encoder-forecaster): strided 3x3 conv 48->24 (3->16 channels) + GroupNorm(8) + ReLU, applied to every step with shared weights;
   ConvLSTM, 2 layers, C=16, kernel 3x3, at 24x24 (zero init states, forget-gate bias 1); on the LAST hidden state: bilinear upsample x2 (24->48) + conv3x3 (16->16) + GroupNorm(8) + ReLU + 1x1 conv -> 3 channels at 48x48; crop [4:44,4:44] -> inner 40x40.
2. Input: every 3rd day of the same 30-day window: t-27, t-24, ..., t (10 steps); each step = anomaly + sin/cos doy (3 channels). No day later than t.
3. C fixed at 16 (no C search); seeds 0,1,2; max 30 epochs; patience 8. All other settings as in the spec (MSE inner 40x40 x 3 leads, Adam 1e-3, batch 32, full training set shuffled, best weights, same pairs).
4. 1-epoch trial, time and estimated total reported; if the total is over ~8 h with 2-3 parallel jobs the user is told before launching. Then the same validation report with U-Net and ConvLSTM side by side. STOP before the test period.
Timing is not a result. The `C_selection` rule does not apply to this model (one C); U-Net keeps C in {16,32}.
