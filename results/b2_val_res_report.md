# B2 plain vs residual whole-region models - validation 2016-2019 (inner 40x40, identical pairs for every method)

RMSE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation; skill = 1 - RMSE/RMSE(reference). Deep models: mean ± sample std (ddof=1) over seeds 0,1,2. Residual = last input anomaly (day t, same for all 3 leads) + network output; same architecture, C, seeds and training settings as the plain model. C fixed from the plain-model selection (U-Net 32, ConvLSTM-ED 16). Reference rows are the frozen B1 numbers (re-checked). Variant chosen per model and region by the lowest mean validation RMSE over seeds and leads (marked ◀). Timing is not a result.

## Variant selection (plain vs residual)

| region | model | mean RMSE plain | mean RMSE residual | chosen |
|---|---|---|---|---|
| kuroshio | U-Net | 0.8064 | 0.7947 | residual |
| kuroshio | ConvLSTM | 0.8141 | 0.8022 | residual |
| subtropical | U-Net | 0.3730 | 0.3621 | residual |
| subtropical | ConvLSTM | 0.3732 | 0.3617 | residual |

## kuroshio  (issue dates 2016-01-01..2019-12-01, n=1431)

### lead 1 d

| method | RMSE | ACC | skill vs Ridge-patch | skill vs anomaly persistence |
|---|---|---|---|---|
| anomaly persistence | 0.383 | 0.948 | -0.043 | +0.000 |
| Ridge | 0.368 | 0.951 | -0.001 | +0.040 |
| Ridge-patch | 0.368 | 0.951 | +0.000 | +0.041 |
| LSTM-patch | 0.372 ± 0.001 | 0.950 ± 0.000 | -0.011 ± 0.004 | +0.030 ± 0.004 |
| U-Net (C=32) | 0.403 ± 0.003 | 0.942 ± 0.002 | -0.096 ± 0.007 | -0.052 ± 0.007 |
| U-Net residual (C=32) ◀ | 0.376 ± 0.000 | 0.949 ± 0.000 | -0.024 ± 0.001 | +0.018 ± 0.001 |
| ConvLSTM (C=16) | 0.422 ± 0.009 | 0.938 ± 0.002 | -0.148 ± 0.025 | -0.101 ± 0.024 |
| ConvLSTM residual (C=16) ◀ | 0.378 ± 0.000 | 0.949 ± 0.000 | -0.027 ± 0.001 | +0.015 ± 0.001 |

### lead 7 d

| method | RMSE | ACC | skill vs Ridge-patch | skill vs anomaly persistence |
|---|---|---|---|---|
| anomaly persistence | 0.968 | 0.682 | -0.105 | +0.000 |
| Ridge | 0.893 | 0.682 | -0.020 | +0.077 |
| Ridge-patch | 0.876 | 0.696 | +0.000 | +0.095 |
| LSTM-patch | 0.874 ± 0.001 | 0.697 ± 0.002 | +0.002 ± 0.002 | +0.097 ± 0.001 |
| U-Net (C=32) | 0.868 ± 0.002 | 0.703 ± 0.003 | +0.009 ± 0.002 | +0.103 ± 0.002 |
| U-Net residual (C=32) ◀ | 0.863 ± 0.002 | 0.705 ± 0.002 | +0.014 ± 0.003 | +0.108 ± 0.003 |
| ConvLSTM (C=16) | 0.868 ± 0.002 | 0.700 ± 0.003 | +0.009 ± 0.002 | +0.103 ± 0.002 |
| ConvLSTM residual (C=16) ◀ | 0.868 ± 0.001 | 0.702 ± 0.001 | +0.008 ± 0.002 | +0.103 ± 0.002 |

### lead 30 d

| method | RMSE | ACC | skill vs Ridge-patch | skill vs anomaly persistence |
|---|---|---|---|---|
| anomaly persistence | 1.453 | 0.281 | -0.230 | +0.000 |
| Ridge | 1.196 | 0.287 | -0.013 | +0.177 |
| Ridge-patch | 1.181 | 0.313 | +0.000 | +0.187 |
| LSTM-patch | 1.165 ± 0.003 | 0.328 ± 0.008 | +0.014 ± 0.002 | +0.199 ± 0.002 |
| U-Net (C=32) | 1.148 ± 0.010 | 0.362 ± 0.020 | +0.028 ± 0.008 | +0.210 ± 0.007 |
| U-Net residual (C=32) ◀ | 1.145 ± 0.004 | 0.368 ± 0.002 | +0.031 ± 0.003 | +0.212 ± 0.003 |
| ConvLSTM (C=16) | 1.152 ± 0.003 | 0.348 ± 0.002 | +0.024 ± 0.002 | +0.207 ± 0.002 |
| ConvLSTM residual (C=16) ◀ | 1.161 ± 0.003 | 0.338 ± 0.002 | +0.017 ± 0.002 | +0.201 ± 0.002 |

## subtropical  (issue dates 2016-01-01..2019-12-01, n=1431)

### lead 1 d

| method | RMSE | ACC | skill vs Ridge-patch | skill vs anomaly persistence |
|---|---|---|---|---|
| anomaly persistence | 0.203 | 0.924 | -0.023 | +0.000 |
| Ridge | 0.199 | 0.926 | -0.000 | +0.022 |
| Ridge-patch | 0.199 | 0.925 | +0.000 | +0.022 |
| LSTM-patch | 0.202 ± 0.004 | 0.922 ± 0.002 | -0.018 ± 0.018 | +0.005 ± 0.018 |
| U-Net (C=32) | 0.237 ± 0.005 | 0.894 ± 0.006 | -0.194 ± 0.026 | -0.167 ± 0.026 |
| U-Net residual (C=32) ◀ | 0.196 ± 0.000 | 0.927 ± 0.000 | +0.014 ± 0.002 | +0.036 ± 0.002 |
| ConvLSTM (C=16) | 0.234 ± 0.006 | 0.899 ± 0.007 | -0.176 ± 0.033 | -0.150 ± 0.032 |
| ConvLSTM residual (C=16) ◀ | 0.198 ± 0.001 | 0.926 ± 0.000 | +0.005 ± 0.006 | +0.028 ± 0.006 |

### lead 7 d

| method | RMSE | ACC | skill vs Ridge-patch | skill vs anomaly persistence |
|---|---|---|---|---|
| anomaly persistence | 0.437 | 0.659 | -0.095 | +0.000 |
| Ridge | 0.412 | 0.667 | -0.032 | +0.058 |
| Ridge-patch | 0.400 | 0.684 | +0.000 | +0.087 |
| LSTM-patch | 0.387 ± 0.002 | 0.692 ± 0.002 | +0.033 ± 0.004 | +0.116 ± 0.004 |
| U-Net (C=32) | 0.390 ± 0.005 | 0.682 ± 0.004 | +0.024 ± 0.012 | +0.109 ± 0.011 |
| U-Net residual (C=32) ◀ | 0.386 ± 0.001 | 0.692 ± 0.007 | +0.035 ± 0.002 | +0.118 ± 0.002 |
| ConvLSTM (C=16) | 0.387 ± 0.002 | 0.689 ± 0.001 | +0.031 ± 0.004 | +0.115 ± 0.004 |
| ConvLSTM residual (C=16) ◀ | 0.386 ± 0.002 | 0.693 ± 0.002 | +0.035 ± 0.006 | +0.118 ± 0.005 |

### lead 30 d

| method | RMSE | ACC | skill vs Ridge-patch | skill vs anomaly persistence |
|---|---|---|---|---|
| anomaly persistence | 0.619 | 0.325 | -0.130 | +0.000 |
| Ridge | 0.562 | 0.329 | -0.025 | +0.092 |
| Ridge-patch | 0.548 | 0.353 | +0.000 | +0.115 |
| LSTM-patch | 0.516 ± 0.005 | 0.381 ± 0.013 | +0.058 ± 0.009 | +0.167 ± 0.008 |
| U-Net (C=32) | 0.492 ± 0.007 | 0.410 ± 0.008 | +0.102 ± 0.013 | +0.205 ± 0.011 |
| U-Net residual (C=32) ◀ | 0.504 ± 0.007 | 0.370 ± 0.014 | +0.079 ± 0.013 | +0.185 ± 0.011 |
| ConvLSTM (C=16) | 0.499 ± 0.005 | 0.415 ± 0.012 | +0.090 ± 0.009 | +0.194 ± 0.008 |
| ConvLSTM residual (C=16) ◀ | 0.502 ± 0.002 | 0.376 ± 0.005 | +0.084 ± 0.003 | +0.189 ± 0.003 |

## Residual vs plain (RMSE skill of residual relative to plain = 1 - RMSE_res/RMSE_plain; mean over seeds, ± std of the per-seed values)

| region | model | 1 d | 7 d | 30 d |
|---|---|---|---|---|
| kuroshio | U-Net | +0.066 ± 0.005 | +0.006 ± 0.002 | +0.003 ± 0.005 |
| kuroshio | ConvLSTM | +0.105 ± 0.020 | -0.000 ± 0.004 | -0.007 ± 0.003 |
| subtropical | U-Net | +0.174 ± 0.017 | +0.011 ± 0.012 | -0.026 ± 0.002 |
| subtropical | ConvLSTM | +0.154 ± 0.024 | +0.004 ± 0.007 | -0.006 ± 0.007 |

## Chosen C (lowest mean validation RMSE over seeds and leads)
(C selection of the plain models: see the table in results/b2_val_report.md, Correction 2026-11-10. Used here: U-Net 32, ConvLSTM-ED 16.)


## Learning curves (all C; best epoch / epochs run; last val - best val; train loss at best epoch; threads)

| region | model | C | seed | best epoch / run | best val | last val - best val | train at best | threads |
|---|---|---|---|---|---|---|---|---|
| kuroshio | U-Net | 32 | 0 | 4/12 | 0.5596 | +0.1574 | 0.4490 | 3 |
| kuroshio | U-Net | 32 | 1 | 3/11 | 0.5480 | +0.1204 | 0.4642 | 3 |
| kuroshio | U-Net | 32 | 2 | 4/12 | 0.5574 | +0.1185 | 0.4543 | 3 |
| kuroshio | U-Net residual | 32 | 0 | 4/12 | 0.5478 | +0.1107 | 0.4522 | 3 |
| kuroshio | U-Net residual | 32 | 1 | 4/12 | 0.5436 | +0.1277 | 0.4499 | 3 |
| kuroshio | U-Net residual | 32 | 2 | 3/11 | 0.5464 | +0.1837 | 0.4570 | 3 |
| kuroshio | ConvLSTM | 16 | 0 | 8/16 | 0.5615 | +0.0410 | 0.4357 | 4 |
| kuroshio | ConvLSTM | 16 | 1 | 5/13 | 0.5613 | +0.0275 | 0.4610 | 4 |
| kuroshio | ConvLSTM | 16 | 2 | 6/14 | 0.5608 | +0.0294 | 0.4528 | 4 |
| kuroshio | ConvLSTM residual | 16 | 0 | 3/11 | 0.5574 | +0.0300 | 0.4625 | 4 |
| kuroshio | ConvLSTM residual | 16 | 1 | 5/13 | 0.5563 | +0.0573 | 0.4535 | 4 |
| kuroshio | ConvLSTM residual | 16 | 2 | 4/12 | 0.5586 | +0.0333 | 0.4567 | 4 |
| subtropical | U-Net | 32 | 0 | 3/11 | 0.3123 | +0.1309 | 0.4629 | 3 |
| subtropical | U-Net | 32 | 1 | 2/10 | 0.3066 | +0.1434 | 0.4757 | 3 |
| subtropical | U-Net | 32 | 2 | 2/10 | 0.3094 | +0.0567 | 0.4786 | 3 |
| subtropical | U-Net residual | 32 | 0 | 3/11 | 0.3092 | +0.1213 | 0.4622 | 3 |
| subtropical | U-Net residual | 32 | 1 | 2/10 | 0.2995 | +0.0672 | 0.4691 | 3 |
| subtropical | U-Net residual | 32 | 2 | 4/12 | 0.3018 | +0.0999 | 0.4539 | 3 |
| subtropical | ConvLSTM | 16 | 0 | 2/10 | 0.3139 | +0.0201 | 0.4802 | 4 |
| subtropical | ConvLSTM | 16 | 1 | 2/10 | 0.3106 | +0.0146 | 0.4884 | 4 |
| subtropical | ConvLSTM | 16 | 2 | 5/13 | 0.3101 | +0.0523 | 0.4562 | 4 |
| subtropical | ConvLSTM residual | 16 | 0 | 2/10 | 0.3011 | +0.0256 | 0.4757 | 4 |
| subtropical | ConvLSTM residual | 16 | 1 | 2/10 | 0.3017 | +0.0353 | 0.4785 | 4 |
| subtropical | ConvLSTM residual | 16 | 2 | 4/12 | 0.3032 | +0.0235 | 0.4615 | 4 |

## Skill vs Ridge-patch by season (mean over seeds, chosen C)

| region | lead | model | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|---|
| kuroshio | 1 | U-Net | -0.104 | -0.071 | -0.098 | -0.128 |
| kuroshio | 1 | U-Net residual | -0.013 | -0.017 | -0.029 | -0.043 |
| kuroshio | 1 | ConvLSTM | -0.129 | -0.131 | -0.148 | -0.204 |
| kuroshio | 1 | ConvLSTM residual | -0.014 | -0.020 | -0.032 | -0.048 |
| kuroshio | 7 | U-Net | -0.001 | +0.001 | +0.014 | +0.022 |
| kuroshio | 7 | U-Net residual | +0.008 | +0.008 | +0.019 | +0.022 |
| kuroshio | 7 | ConvLSTM | +0.013 | +0.009 | +0.009 | +0.005 |
| kuroshio | 7 | ConvLSTM residual | +0.010 | +0.000 | +0.011 | +0.014 |
| kuroshio | 30 | U-Net | +0.013 | +0.019 | +0.051 | +0.022 |
| kuroshio | 30 | U-Net residual | +0.014 | +0.028 | +0.044 | +0.033 |
| kuroshio | 30 | ConvLSTM | +0.023 | +0.026 | +0.037 | +0.007 |
| kuroshio | 30 | ConvLSTM residual | +0.010 | +0.021 | +0.025 | +0.010 |
| subtropical | 1 | U-Net | -0.129 | -0.232 | -0.170 | -0.239 |
| subtropical | 1 | U-Net residual | +0.037 | +0.002 | +0.010 | +0.013 |
| subtropical | 1 | ConvLSTM | -0.139 | -0.246 | -0.148 | -0.145 |
| subtropical | 1 | ConvLSTM residual | +0.027 | -0.009 | +0.003 | +0.004 |
| subtropical | 7 | U-Net | +0.046 | +0.003 | +0.028 | +0.034 |
| subtropical | 7 | U-Net residual | +0.052 | +0.018 | +0.023 | +0.071 |
| subtropical | 7 | ConvLSTM | +0.051 | -0.005 | +0.040 | +0.063 |
| subtropical | 7 | ConvLSTM residual | +0.039 | +0.029 | +0.024 | +0.063 |
| subtropical | 30 | U-Net | +0.129 | +0.049 | +0.103 | +0.174 |
| subtropical | 30 | U-Net residual | +0.132 | +0.026 | +0.049 | +0.176 |
| subtropical | 30 | ConvLSTM | +0.119 | +0.057 | +0.071 | +0.150 |
| subtropical | 30 | ConvLSTM residual | +0.125 | +0.054 | +0.041 | +0.171 |

## Spatial check: skill vs Ridge-patch, edge ring (outer 4) vs centre (inner 32x32), mean ± std over seeds (maps: figB2_maps.png)

| region | lead | model | edge | centre |
|---|---|---|---|---|
| kuroshio | 1 | U-Net | -0.103 ± 0.006 | -0.092 ± 0.008 |
| kuroshio | 1 | U-Net residual | -0.030 ± 0.001 | -0.020 ± 0.001 |
| kuroshio | 1 | ConvLSTM | -0.168 ± 0.029 | -0.134 ± 0.023 |
| kuroshio | 1 | ConvLSTM residual | -0.032 ± 0.001 | -0.024 ± 0.001 |
| kuroshio | 7 | U-Net | +0.003 ± 0.001 | +0.013 ± 0.004 |
| kuroshio | 7 | U-Net residual | +0.008 ± 0.003 | +0.019 ± 0.003 |
| kuroshio | 7 | ConvLSTM | +0.006 ± 0.000 | +0.011 ± 0.004 |
| kuroshio | 7 | ConvLSTM residual | +0.004 ± 0.002 | +0.011 ± 0.003 |
| kuroshio | 30 | U-Net | +0.029 ± 0.006 | +0.027 ± 0.011 |
| kuroshio | 30 | U-Net residual | +0.034 ± 0.004 | +0.028 ± 0.006 |
| kuroshio | 30 | ConvLSTM | +0.029 ± 0.005 | +0.021 ± 0.001 |
| kuroshio | 30 | ConvLSTM residual | +0.020 ± 0.004 | +0.015 ± 0.002 |
| subtropical | 1 | U-Net | -0.206 ± 0.025 | -0.187 ± 0.028 |
| subtropical | 1 | U-Net residual | +0.005 ± 0.003 | +0.019 ± 0.002 |
| subtropical | 1 | ConvLSTM | -0.191 ± 0.033 | -0.166 ± 0.033 |
| subtropical | 1 | ConvLSTM residual | -0.004 ± 0.007 | +0.011 ± 0.007 |
| subtropical | 7 | U-Net | +0.015 ± 0.009 | +0.030 ± 0.014 |
| subtropical | 7 | U-Net residual | +0.024 ± 0.003 | +0.041 ± 0.001 |
| subtropical | 7 | ConvLSTM | +0.020 ± 0.002 | +0.038 ± 0.005 |
| subtropical | 7 | ConvLSTM residual | +0.023 ± 0.005 | +0.042 ± 0.006 |
| subtropical | 30 | U-Net | +0.096 ± 0.017 | +0.106 ± 0.010 |
| subtropical | 30 | U-Net residual | +0.073 ± 0.017 | +0.083 ± 0.011 |
| subtropical | 30 | ConvLSTM | +0.078 ± 0.008 | +0.097 ± 0.009 |
| subtropical | 30 | ConvLSTM residual | +0.076 ± 0.006 | +0.089 ± 0.001 |

## Answers (generated from the numbers above by code; whole-region models = the CHOSEN variant per model and region; read with the std)

- **Chosen variants:** kuroshio U-Net: residual; kuroshio ConvLSTM: residual; subtropical U-Net: residual; subtropical ConvLSTM: residual
- **U-Net vs Ridge-patch (RMSE skill, mean over seeds):** positive in 5/6 region-lead cells (positive by more than one seed-std in 5); range -0.024 to +0.079. kuroshio 1 d -0.024; kuroshio 7 d +0.014; kuroshio 30 d +0.031; subtropical 1 d +0.014; subtropical 7 d +0.035; subtropical 30 d +0.079
- **ConvLSTM vs Ridge-patch (RMSE skill, mean over seeds):** positive in 5/6 region-lead cells (positive by more than one seed-std in 4); range -0.027 to +0.084. kuroshio 1 d -0.027; kuroshio 7 d +0.008; kuroshio 30 d +0.017; subtropical 1 d +0.005; subtropical 7 d +0.035; subtropical 30 d +0.084
- **U-Net vs ConvLSTM:** U-Net has the lower mean RMSE in 4/6 cells (RMSE U-Net / ConvLSTM: kuroshio 1 d 0.376/0.378; kuroshio 7 d 0.863/0.868; kuroshio 30 d 1.145/1.161; subtropical 1 d 0.196/0.198; subtropical 7 d 0.386/0.386; subtropical 30 d 0.504/0.502).
- **U-Net, kuroshio vs subtropical (skill vs Ridge-patch):** 1 d: kuroshio -0.024 vs subtropical +0.014 (opposite sign); 7 d: kuroshio +0.014 vs subtropical +0.035; 30 d: kuroshio +0.031 vs subtropical +0.079
- **ConvLSTM, kuroshio vs subtropical (skill vs Ridge-patch):** 1 d: kuroshio -0.027 vs subtropical +0.005 (opposite sign); 7 d: kuroshio +0.008 vs subtropical +0.035; 30 d: kuroshio +0.017 vs subtropical +0.084
- **U-Net edge vs centre (skill vs Ridge-patch, outer 4 rings vs inner 32x32):** kuroshio 7 d edge +0.008 / centre +0.019; kuroshio 30 d edge +0.034 / centre +0.028; subtropical 7 d edge +0.024 / centre +0.041; subtropical 30 d edge +0.073 / centre +0.083
- **ConvLSTM edge vs centre (skill vs Ridge-patch, outer 4 rings vs inner 32x32):** kuroshio 7 d edge +0.004 / centre +0.011; kuroshio 30 d edge +0.020 / centre +0.015; subtropical 7 d edge +0.023 / centre +0.042; subtropical 30 d edge +0.076 / centre +0.089