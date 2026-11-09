# B2 whole-region models - validation 2016-2019 (inner 40x40, identical pairs for every method)

RMSE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation; skill = 1 - RMSE/RMSE(Ridge-patch). Deep models: mean ± sample std (ddof=1) over seeds 0,1,2 at the chosen C. Reference rows are the frozen B1 numbers (re-checked). Timing is not a result.

## kuroshio  (issue dates 2016-01-01..2019-12-01, n=1431)

### lead 1 d

| method | RMSE | ACC | skill vs Ridge-patch |
|---|---|---|---|
| anomaly persistence | 0.383 | 0.948 | -0.043 |
| Ridge | 0.368 | 0.951 | -0.001 |
| Ridge-patch | 0.368 | 0.951 | +0.000 |
| LSTM-patch | 0.372 ± 0.001 | 0.950 ± 0.000 | -0.011 ± 0.004 |
| U-Net (C=32) | 0.403 ± 0.003 | 0.942 ± 0.002 | -0.096 ± 0.007 |
| ConvLSTM (C=16) | 0.422 ± 0.009 | 0.938 ± 0.002 | -0.148 ± 0.025 |

### lead 7 d

| method | RMSE | ACC | skill vs Ridge-patch |
|---|---|---|---|
| anomaly persistence | 0.968 | 0.682 | -0.105 |
| Ridge | 0.893 | 0.682 | -0.020 |
| Ridge-patch | 0.876 | 0.696 | +0.000 |
| LSTM-patch | 0.874 ± 0.001 | 0.697 ± 0.002 | +0.002 ± 0.002 |
| U-Net (C=32) | 0.868 ± 0.002 | 0.703 ± 0.003 | +0.009 ± 0.002 |
| ConvLSTM (C=16) | 0.868 ± 0.002 | 0.700 ± 0.003 | +0.009 ± 0.002 |

### lead 30 d

| method | RMSE | ACC | skill vs Ridge-patch |
|---|---|---|---|
| anomaly persistence | 1.453 | 0.281 | -0.230 |
| Ridge | 1.196 | 0.287 | -0.013 |
| Ridge-patch | 1.181 | 0.313 | +0.000 |
| LSTM-patch | 1.165 ± 0.003 | 0.328 ± 0.008 | +0.014 ± 0.002 |
| U-Net (C=32) | 1.148 ± 0.010 | 0.362 ± 0.020 | +0.028 ± 0.008 |
| ConvLSTM (C=16) | 1.152 ± 0.003 | 0.348 ± 0.002 | +0.024 ± 0.002 |

## subtropical  (issue dates 2016-01-01..2019-12-01, n=1431)

### lead 1 d

| method | RMSE | ACC | skill vs Ridge-patch |
|---|---|---|---|
| anomaly persistence | 0.203 | 0.924 | -0.023 |
| Ridge | 0.199 | 0.926 | -0.000 |
| Ridge-patch | 0.199 | 0.925 | +0.000 |
| LSTM-patch | 0.202 ± 0.004 | 0.922 ± 0.002 | -0.018 ± 0.018 |
| U-Net (C=32) | 0.237 ± 0.005 | 0.894 ± 0.006 | -0.194 ± 0.026 |
| ConvLSTM (C=16) | 0.234 ± 0.006 | 0.899 ± 0.007 | -0.176 ± 0.033 |

### lead 7 d

| method | RMSE | ACC | skill vs Ridge-patch |
|---|---|---|---|
| anomaly persistence | 0.437 | 0.659 | -0.095 |
| Ridge | 0.412 | 0.667 | -0.032 |
| Ridge-patch | 0.400 | 0.684 | +0.000 |
| LSTM-patch | 0.387 ± 0.002 | 0.692 ± 0.002 | +0.033 ± 0.004 |
| U-Net (C=32) | 0.390 ± 0.005 | 0.682 ± 0.004 | +0.024 ± 0.012 |
| ConvLSTM (C=16) | 0.387 ± 0.002 | 0.689 ± 0.001 | +0.031 ± 0.004 |

### lead 30 d

| method | RMSE | ACC | skill vs Ridge-patch |
|---|---|---|---|
| anomaly persistence | 0.619 | 0.325 | -0.130 |
| Ridge | 0.562 | 0.329 | -0.025 |
| Ridge-patch | 0.548 | 0.353 | +0.000 |
| LSTM-patch | 0.516 ± 0.005 | 0.381 ± 0.013 | +0.058 ± 0.009 |
| U-Net (C=32) | 0.492 ± 0.007 | 0.410 ± 0.008 | +0.102 ± 0.013 |
| ConvLSTM (C=16) | 0.499 ± 0.005 | 0.415 ± 0.012 | +0.090 ± 0.009 |

## Chosen C (lowest mean validation RMSE over seeds and leads)

| region | model | mean RMSE C=16 | mean RMSE C=32 | chosen |
|---|---|---|---|---|
| kuroshio | U-Net | - | - | 32 |
| kuroshio | ConvLSTM | - | - | 16 |
| subtropical | U-Net | - | - | 32 |
| subtropical | ConvLSTM | - | - | 16 |

## Learning curves (all C; best epoch / epochs run; last val - best val; train loss at best epoch; threads)

| region | model | C | seed | best epoch / run | best val | last val - best val | train at best | threads |
|---|---|---|---|---|---|---|---|---|
| kuroshio | U-Net | 16 | 0 | 2/10 | 0.5761 | +0.1283 | 0.4783 | 3 |
| kuroshio | U-Net | 16 | 1 | 3/11 | 0.5659 | +0.1521 | 0.4668 | 3 |
| kuroshio | U-Net | 16 | 2 | 2/10 | 0.5698 | +0.1216 | 0.4813 | 3 |
| kuroshio | U-Net | 32 | 0 | 4/12 | 0.5596 | +0.1574 | 0.4490 | 3 |
| kuroshio | U-Net | 32 | 1 | 3/11 | 0.5480 | +0.1204 | 0.4642 | 3 |
| kuroshio | U-Net | 32 | 2 | 4/12 | 0.5574 | +0.1185 | 0.4543 | 3 |
| kuroshio | ConvLSTM | 16 | 0 | 8/16 | 0.5615 | +0.0410 | 0.4357 | 4 |
| kuroshio | ConvLSTM | 16 | 1 | 5/13 | 0.5613 | +0.0275 | 0.4610 | 4 |
| kuroshio | ConvLSTM | 16 | 2 | 6/14 | 0.5608 | +0.0294 | 0.4528 | 4 |
| subtropical | U-Net | 16 | 0 | 3/11 | 0.3144 | +0.1922 | 0.4680 | 3 |
| subtropical | U-Net | 16 | 1 | 2/10 | 0.3101 | +0.0362 | 0.4886 | 3 |
| subtropical | U-Net | 16 | 2 | 5/13 | 0.3076 | +0.1125 | 0.4446 | 3 |
| subtropical | U-Net | 32 | 0 | 3/11 | 0.3123 | +0.1309 | 0.4629 | 3 |
| subtropical | U-Net | 32 | 1 | 2/10 | 0.3066 | +0.1434 | 0.4757 | 3 |
| subtropical | U-Net | 32 | 2 | 2/10 | 0.3094 | +0.0567 | 0.4786 | 3 |
| subtropical | ConvLSTM | 16 | 0 | 2/10 | 0.3139 | +0.0201 | 0.4802 | 4 |
| subtropical | ConvLSTM | 16 | 1 | 2/10 | 0.3106 | +0.0146 | 0.4884 | 4 |
| subtropical | ConvLSTM | 16 | 2 | 5/13 | 0.3101 | +0.0523 | 0.4562 | 4 |

## Skill vs Ridge-patch by season (mean over seeds, chosen C)

| region | lead | model | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|---|
| kuroshio | 1 | U-Net | -0.104 | -0.071 | -0.098 | -0.128 |
| kuroshio | 1 | ConvLSTM | -0.129 | -0.131 | -0.148 | -0.204 |
| kuroshio | 7 | U-Net | -0.001 | +0.001 | +0.014 | +0.022 |
| kuroshio | 7 | ConvLSTM | +0.013 | +0.009 | +0.009 | +0.005 |
| kuroshio | 30 | U-Net | +0.013 | +0.019 | +0.051 | +0.022 |
| kuroshio | 30 | ConvLSTM | +0.023 | +0.026 | +0.037 | +0.007 |
| subtropical | 1 | U-Net | -0.129 | -0.232 | -0.170 | -0.239 |
| subtropical | 1 | ConvLSTM | -0.139 | -0.246 | -0.148 | -0.145 |
| subtropical | 7 | U-Net | +0.046 | +0.003 | +0.028 | +0.034 |
| subtropical | 7 | ConvLSTM | +0.051 | -0.005 | +0.040 | +0.063 |
| subtropical | 30 | U-Net | +0.129 | +0.049 | +0.103 | +0.174 |
| subtropical | 30 | ConvLSTM | +0.119 | +0.057 | +0.071 | +0.150 |

## Spatial check: skill vs Ridge-patch, edge ring (outer 4) vs centre (inner 32x32), mean ± std over seeds (maps: figB2_maps.png)

| region | lead | model | edge | centre |
|---|---|---|---|---|
| kuroshio | 1 | U-Net | -0.103 ± 0.006 | -0.092 ± 0.008 |
| kuroshio | 1 | ConvLSTM | -0.168 ± 0.029 | -0.134 ± 0.023 |
| kuroshio | 7 | U-Net | +0.003 ± 0.001 | +0.013 ± 0.004 |
| kuroshio | 7 | ConvLSTM | +0.006 ± 0.000 | +0.011 ± 0.004 |
| kuroshio | 30 | U-Net | +0.029 ± 0.006 | +0.027 ± 0.011 |
| kuroshio | 30 | ConvLSTM | +0.029 ± 0.005 | +0.021 ± 0.001 |
| subtropical | 1 | U-Net | -0.206 ± 0.025 | -0.187 ± 0.028 |
| subtropical | 1 | ConvLSTM | -0.191 ± 0.033 | -0.166 ± 0.033 |
| subtropical | 7 | U-Net | +0.015 ± 0.009 | +0.030 ± 0.014 |
| subtropical | 7 | ConvLSTM | +0.020 ± 0.002 | +0.038 ± 0.005 |
| subtropical | 30 | U-Net | +0.096 ± 0.017 | +0.106 ± 0.010 |
| subtropical | 30 | ConvLSTM | +0.078 ± 0.008 | +0.097 ± 0.009 |

## Answers (generated from the numbers above by code; read with the std)

- **U-Net vs Ridge-patch (RMSE skill, mean over seeds):** positive in 4/6 region-lead cells (positive by more than one seed-std in 4); range -0.194 to +0.102. kuroshio 1 d -0.096; kuroshio 7 d +0.009; kuroshio 30 d +0.028; subtropical 1 d -0.194; subtropical 7 d +0.024; subtropical 30 d +0.102
- **ConvLSTM vs Ridge-patch (RMSE skill, mean over seeds):** positive in 4/6 region-lead cells (positive by more than one seed-std in 4); range -0.176 to +0.090. kuroshio 1 d -0.148; kuroshio 7 d +0.009; kuroshio 30 d +0.024; subtropical 1 d -0.176; subtropical 7 d +0.031; subtropical 30 d +0.090
- **U-Net vs ConvLSTM:** U-Net has the lower mean RMSE in 4/6 cells (RMSE U-Net / ConvLSTM: kuroshio 1 d 0.403/0.422; kuroshio 7 d 0.868/0.868; kuroshio 30 d 1.148/1.152; subtropical 1 d 0.237/0.234; subtropical 7 d 0.390/0.387; subtropical 30 d 0.492/0.499).
- **U-Net, kuroshio vs subtropical (skill vs Ridge-patch):** 1 d: kuroshio -0.096 vs subtropical -0.194; 7 d: kuroshio +0.009 vs subtropical +0.024; 30 d: kuroshio +0.028 vs subtropical +0.102
- **ConvLSTM, kuroshio vs subtropical (skill vs Ridge-patch):** 1 d: kuroshio -0.148 vs subtropical -0.176; 7 d: kuroshio +0.009 vs subtropical +0.031; 30 d: kuroshio +0.024 vs subtropical +0.090
- **U-Net edge vs centre (skill vs Ridge-patch, outer 4 rings vs inner 32x32):** kuroshio 7 d edge +0.003 / centre +0.013; kuroshio 30 d edge +0.029 / centre +0.027; subtropical 7 d edge +0.015 / centre +0.030; subtropical 30 d edge +0.096 / centre +0.106
- **ConvLSTM edge vs centre (skill vs Ridge-patch, outer 4 rings vs inner 32x32):** kuroshio 7 d edge +0.006 / centre +0.011; kuroshio 30 d edge +0.029 / centre +0.021; subtropical 7 d edge +0.020 / centre +0.038; subtropical 30 d edge +0.078 / centre +0.097