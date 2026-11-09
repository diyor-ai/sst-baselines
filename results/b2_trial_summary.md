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
