# B1 spatial context - validation 2016-2019 (inner 40x40, identical pairs for every method)

RMSE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation (primary ACC); skill = 1 - RMSE/RMSE(Ridge). LSTM entries: mean ± sample std (ddof=1) over seeds 0,1,2 of per-seed scores.

## kuroshio  (issue dates 2016-01-01..2019-12-01, n=1431)

### lead 1 d

| method | RMSE | ACC | skill vs Ridge |
|---|---|---|---|
| anomaly persistence | 0.383 | 0.948 | -0.042 |
| Ridge | 0.368 | 0.951 | +0.000 |
| Ridge+doy | 0.368 | 0.951 | -0.000 |
| Ridge-patch | 0.368 | 0.951 | +0.001 |
| LSTM (point) | 0.369 ± 0.000 | 0.951 ± 0.000 | -0.001 ± 0.001 |
| LSTM-patch | 0.372 ± 0.001 | 0.950 ± 0.000 | -0.010 ± 0.004 |

### lead 7 d

| method | RMSE | ACC | skill vs Ridge |
|---|---|---|---|
| anomaly persistence | 0.968 | 0.682 | -0.084 |
| Ridge | 0.893 | 0.682 | +0.000 |
| Ridge+doy | 0.893 | 0.682 | -0.000 |
| Ridge-patch | 0.876 | 0.696 | +0.020 |
| LSTM (point) | 0.888 ± 0.002 | 0.685 ± 0.001 | +0.006 ± 0.002 |
| LSTM-patch | 0.874 ± 0.001 | 0.697 ± 0.002 | +0.022 ± 0.001 |

### lead 30 d

| method | RMSE | ACC | skill vs Ridge |
|---|---|---|---|
| anomaly persistence | 1.453 | 0.281 | -0.215 |
| Ridge | 1.196 | 0.287 | +0.000 |
| Ridge+doy | 1.196 | 0.287 | -0.000 |
| Ridge-patch | 1.181 | 0.313 | +0.012 |
| LSTM (point) | 1.179 ± 0.003 | 0.306 ± 0.008 | +0.014 ± 0.003 |
| LSTM-patch | 1.165 ± 0.003 | 0.328 ± 0.008 | +0.026 ± 0.002 |

## subtropical  (issue dates 2016-01-01..2019-12-01, n=1431)

### lead 1 d

| method | RMSE | ACC | skill vs Ridge |
|---|---|---|---|
| anomaly persistence | 0.203 | 0.924 | -0.022 |
| Ridge | 0.199 | 0.926 | +0.000 |
| Ridge+doy | 0.199 | 0.926 | -0.000 |
| Ridge-patch | 0.199 | 0.925 | +0.000 |
| LSTM (point) | 0.200 ± 0.001 | 0.924 ± 0.000 | -0.006 ± 0.004 |
| LSTM-patch | 0.202 ± 0.004 | 0.922 ± 0.002 | -0.018 ± 0.018 |

### lead 7 d

| method | RMSE | ACC | skill vs Ridge |
|---|---|---|---|
| anomaly persistence | 0.437 | 0.659 | -0.061 |
| Ridge | 0.412 | 0.667 | +0.000 |
| Ridge+doy | 0.412 | 0.667 | -0.000 |
| Ridge-patch | 0.400 | 0.684 | +0.031 |
| LSTM (point) | 0.401 ± 0.002 | 0.672 ± 0.004 | +0.026 ± 0.005 |
| LSTM-patch | 0.387 ± 0.002 | 0.692 ± 0.002 | +0.062 ± 0.004 |

### lead 30 d

| method | RMSE | ACC | skill vs Ridge |
|---|---|---|---|
| anomaly persistence | 0.619 | 0.325 | -0.102 |
| Ridge | 0.562 | 0.329 | +0.000 |
| Ridge+doy | 0.562 | 0.328 | -0.000 |
| Ridge-patch | 0.548 | 0.353 | +0.025 |
| LSTM (point) | 0.536 ± 0.005 | 0.347 ± 0.005 | +0.045 ± 0.008 |
| LSTM-patch | 0.516 ± 0.005 | 0.381 ± 0.013 | +0.082 ± 0.009 |

## Skill vs Ridge by season (RMSE skill; LSTM = mean over seeds)

| region | lead | method | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|---|
| kuroshio | 1 | Ridge+doy | +0.000 | -0.000 | -0.000 | +0.000 |
| kuroshio | 1 | Ridge-patch | -0.004 | -0.001 | +0.004 | +0.005 |
| kuroshio | 1 | LSTM (point) | -0.005 | -0.005 | +0.003 | +0.002 |
| kuroshio | 1 | LSTM-patch | -0.014 | -0.017 | -0.001 | -0.011 |
| kuroshio | 7 | Ridge+doy | +0.000 | -0.000 | -0.000 | -0.000 |
| kuroshio | 7 | Ridge-patch | +0.019 | +0.027 | +0.015 | +0.018 |
| kuroshio | 7 | LSTM (point) | +0.002 | +0.002 | +0.010 | +0.008 |
| kuroshio | 7 | LSTM-patch | +0.027 | +0.026 | +0.018 | +0.017 |
| kuroshio | 30 | Ridge+doy | +0.001 | -0.000 | -0.001 | -0.000 |
| kuroshio | 30 | Ridge-patch | +0.015 | +0.017 | +0.010 | +0.007 |
| kuroshio | 30 | LSTM (point) | +0.002 | +0.011 | +0.026 | +0.013 |
| kuroshio | 30 | LSTM-patch | +0.023 | +0.033 | +0.029 | +0.016 |
| subtropical | 1 | Ridge+doy | -0.000 | +0.000 | +0.000 | -0.000 |
| subtropical | 1 | Ridge-patch | -0.012 | +0.005 | +0.001 | +0.005 |
| subtropical | 1 | LSTM (point) | -0.015 | -0.010 | -0.004 | +0.004 |
| subtropical | 1 | LSTM-patch | -0.035 | -0.028 | -0.009 | +0.002 |
| subtropical | 7 | Ridge+doy | +0.000 | +0.000 | +0.000 | -0.000 |
| subtropical | 7 | Ridge-patch | +0.042 | +0.031 | +0.020 | +0.037 |
| subtropical | 7 | LSTM (point) | +0.040 | +0.022 | +0.022 | +0.026 |
| subtropical | 7 | LSTM-patch | +0.092 | +0.060 | +0.044 | +0.068 |
| subtropical | 30 | Ridge+doy | -0.001 | -0.000 | +0.001 | -0.001 |
| subtropical | 30 | Ridge-patch | +0.030 | +0.025 | +0.016 | +0.032 |
| subtropical | 30 | LSTM (point) | +0.081 | +0.025 | +0.037 | +0.058 |
| subtropical | 30 | LSTM-patch | +0.123 | +0.055 | +0.071 | +0.104 |

## Ridge alphas chosen on validation (grid 1e-3..1e3; * = grid edge)

| region | model | lead 1 | lead 7 | lead 30 |
|---|---|---|---|---|
| kuroshio | ridge | 0.00316228 | 0.001 | 0.001 |
| kuroshio | ridge_doy | 0.00316228 | 0.001* | 0.001* |
| kuroshio | ridge_patch | 0.00316228 | 0.1 | 0.316228 |
| subtropical | ridge | 0.01 | 0.001 | 0.001 |
| subtropical | ridge_doy | 0.01 | 0.001* | 0.001* |
| subtropical | ridge_patch | 0.00316228 | 0.0316228 | 0.0316228 |

## Learning curves (best epoch / epochs run; last val - best val; train vs val at best epoch)

| region | model | seed | best epoch / run | best val | last val - best val | train loss at best epoch |
|---|---|---|---|---|---|---|
| kuroshio | lstm_point | 0 | 5/10 | 0.5764 | +0.0072 | 0.4770 |
| kuroshio | lstm_point | 1 | 3/8 | 0.5757 | +0.0054 | 0.4784 |
| kuroshio | lstm_point | 2 | 16/21 | 0.5727 | +0.0031 | 0.4720 |
| kuroshio | lstm_patch | 0 | 5/10 | 0.5614 | +0.0012 | 0.4637 |
| kuroshio | lstm_patch | 1 | 10/15 | 0.5582 | +0.0028 | 0.4618 |
| kuroshio | lstm_patch | 2 | 4/9 | 0.5629 | +0.0059 | 0.4677 |
| subtropical | lstm_point | 0 | 9/14 | 0.3320 | +0.0311 | 0.4830 |
| subtropical | lstm_point | 1 | 5/10 | 0.3400 | +0.0028 | 0.4854 |
| subtropical | lstm_point | 2 | 6/11 | 0.3357 | +0.0167 | 0.4898 |
| subtropical | lstm_patch | 0 | 3/8 | 0.3162 | +0.0214 | 0.4753 |
| subtropical | lstm_patch | 1 | 1/6 | 0.3142 | +0.0078 | 0.5263 |
| subtropical | lstm_patch | 2 | 7/12 | 0.3106 | +0.0074 | 0.4654 |

## Data checks
```
{
 "kuroshio": {
  "inner_sst_equal": true,
  "clim_inner_max_abs_diff": 0.0,
  "clim_inner_bitwise_equal": true,
  "std_halo_data": 1.167915570284816,
  "std_existing_lstm_data": 1.167915570284816,
  "std_bitwise_equal": true,
  "train_pairs_equal": true,
  "val_pairs_equal": true,
  "n_train_pairs": 19774400,
  "n_val_pairs": 2289600,
  "nan_in_halo_grid_up_to_val_end": 0,
  "centre_anomaly_equal_existing_max_abs_diff": 0.0
 },
 "subtropical": {
  "inner_sst_equal": true,
  "clim_inner_max_abs_diff": 0.0,
  "clim_inner_bitwise_equal": true,
  "std_halo_data": 0.6987572937605917,
  "std_existing_lstm_data": 0.6987572937605917,
  "std_bitwise_equal": true,
  "train_pairs_equal": true,
  "val_pairs_equal": true,
  "n_train_pairs": 19774400,
  "n_val_pairs": 2289600,
  "nan_in_halo_grid_up_to_val_end": 0,
  "centre_anomaly_equal_existing_max_abs_diff": 0.0
 }
}
```