# B2 trial summary (2026-11-09; kuroshio, seed 0, 4 torch threads, one job at a time, machine otherwise idle)

| model | C | params | s/epoch (incl. validation) | how measured |
|---|---|---|---|---|
| U-Net | 16 | 496,003 | 57 | full 1-epoch trial (12,359 train / 1,431 val dates) |
| U-Net | 32 | 1,953,539 | 166 | full 1-epoch trial |
| ConvLSTM | 16 | 29,555 | ~5,117 (1.42 h) | EXTRAPOLATED: 20 real training batches (13.0 s/batch, 386 batches/epoch) + validation 68 s |
| ConvLSTM | 32 | 114,403 | ~10,982 (3.05 h) | EXTRAPOLATED: 20 real batches (28.0 s/batch) + validation 163 s |

Deviation: a full 1-epoch ConvLSTM trial would itself take 1.4 h / 3.1 h, so it was replaced by the extrapolation above (results/b2_trials/*_timed.json).

## Total for 24 runs (2 regions x 3 seeds x C{16,32} per model), run one after another; epochs per run = 60 (max) or 20 (assumed, NOT measured)

| model | per epoch for all 12 runs | 20 epochs/run | 60 epochs/run |
|---|---|---|---|
| U-Net | 0.4 h | 7 h | 22 h |
| ConvLSTM | 26.8 h | 537 h | 1610 h |

Parallel jobs (4 x 4 threads on a 10-core / 16-thread CPU) can give at most about 2-3x, not measured.

## Addendum 2026-11-09: revised ConvLSTM (`convlstm_ed`, C=16, 10 steps at 24x24) - profile and 1-epoch trial
Measured on the 4 E-cores (CPUs 12-15, 4 threads) because the P-cores were busy with the U-Net runs; so these numbers are NOT the speed on P-cores and are not a result.
Profile (results/b2_profile/profile_E4.jsonl, batch 32, mean of 6 batches after 2 warm-up):

| deterministic | gather | forward | backward | optimiser | total / batch |
|---|---|---|---|---|---|
| on | 0.007 s | 0.331 s | 0.990 s | 0.003 s | 1.331 s |
| off | 0.008 s | 0.311 s | 0.900 s | 0.003 s | 1.222 s |

Deterministic mode costs 1.09x, the data gather and optimiser are negligible: the model itself is the bottleneck (the recurrence + backward). Determinism kept.
The 6-thread vs 4-thread comparison on P-cores was NOT done (P-cores in use by the U-Net runs); to be done when they are free.
1-epoch trial (kuroshio, seed 0, full epoch incl. validation, results/b2_trials/b2_convlstm_ed_kuroshio_C16_s0.json): **554 s/epoch**.
Total for 6 runs (2 regions x 3 seeds), = 3,324 s per epoch of all runs sequentially; ideal parallel scaling assumed, NOT measured:

| epochs per run (assumed; early stopping not yet observed) | sequential | 2 jobs | 3 jobs |
|---|---|---|---|
| 15 | 13.9 h | 6.9 h | 4.6 h |
| 20 | 18.5 h | 9.2 h | 6.2 h |
| 30 (max) | 27.7 h | 13.9 h | 9.2 h |
