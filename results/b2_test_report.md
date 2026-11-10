# B2 whole-region models - TEST 2020-2025 (inner 40x40, the frozen baselines' pairs)

Protocol: docs/b2_test_protocol.md. Commit 97e74f2, tag pre-b2-test. RMSE/MAE in deg C of forecast SST (cos-lat weighted); ACC = temporal anomaly correlation (spatial ACC in the JSON). Skill = 1 - RMSE/RMSE(reference). Deep models (LSTM-patch, U-Net, ConvLSTM-ED): mean ± sample std (ddof=1) over seeds 0,1,2 of per-seed scores. Variants chosen on validation: kuroshio U-Net: resunet; kuroshio ConvLSTM-ED: resconvlstm_ed; subtropical U-Net: resunet; subtropical ConvLSTM-ED: resconvlstm_ed.

## kuroshio

### lead 1 d  (n pairs 2192)

| method | RMSE | MAE | ACC | skill vs persistence | skill vs Ridge | skill vs Ridge-patch | skill vs LSTM-patch |
|---|---|---|---|---|---|---|---|
| anomaly persistence | 0.258 | 0.162 | 0.977 | +0.000 | -0.026 | -0.032 | +0.133 |
| Ridge | 0.251 | 0.163 | 0.978 | +0.025 | +0.000 | -0.006 | +0.155 |
| Ridge-patch | 0.250 | 0.162 | 0.978 | +0.031 | +0.006 | +0.000 | +0.160 |
| LSTM-patch | 0.297 ± 0.010 | 0.171 ± 0.004 | 0.977 ± 0.000 | -0.153 ± +0.039 | -0.183 ± +0.040 | -0.190 ± +0.041 | - |
| U-Net (resunet, C=32) | 0.261 ± 0.000 | 0.169 ± 0.001 | 0.977 ± 0.000 | -0.011 ± +0.002 | -0.037 ± +0.002 | -0.043 ± +0.002 | +0.124 ± +0.001 |
| ConvLSTM-ED (resconvlstm_ed, C=16) | 0.262 ± 0.002 | 0.171 ± 0.003 | 0.976 ± 0.000 | -0.017 ± +0.009 | -0.043 ± +0.009 | -0.050 ± +0.009 | +0.118 ± +0.007 |

### lead 7 d  (n pairs 2192)

| method | RMSE | MAE | ACC | skill vs persistence | skill vs Ridge | skill vs Ridge-patch | skill vs LSTM-patch |
|---|---|---|---|---|---|---|---|
| anomaly persistence | 0.843 | 0.593 | 0.761 | +0.000 | -0.007 | -0.022 | -0.000 |
| Ridge | 0.838 | 0.600 | 0.752 | +0.007 | +0.000 | -0.015 | +0.007 |
| Ridge-patch | 0.825 | 0.591 | 0.753 | +0.021 | +0.015 | +0.000 | +0.021 |
| LSTM-patch | 0.843 ± 0.006 | 0.595 ± 0.003 | 0.756 ± 0.003 | +0.000 ± +0.007 | -0.007 ± +0.007 | -0.022 ± +0.007 | - |
| U-Net (resunet, C=32) | 0.795 ± 0.005 | 0.569 ± 0.003 | 0.763 ± 0.005 | +0.057 ± +0.006 | +0.051 ± +0.006 | +0.037 ± +0.006 | +0.057 ± +0.006 |
| ConvLSTM-ED (resconvlstm_ed, C=16) | 0.802 ± 0.006 | 0.579 ± 0.003 | 0.759 ± 0.002 | +0.049 ± +0.007 | +0.042 ± +0.007 | +0.028 ± +0.007 | +0.049 ± +0.007 |

### lead 30 d  (n pairs 2192)

| method | RMSE | MAE | ACC | skill vs persistence | skill vs Ridge | skill vs Ridge-patch | skill vs LSTM-patch |
|---|---|---|---|---|---|---|---|
| anomaly persistence | 1.405 | 1.023 | 0.335 | +0.000 | -0.097 | -0.119 | -0.111 |
| Ridge | 1.281 | 0.939 | 0.330 | +0.088 | +0.000 | -0.021 | -0.013 |
| Ridge-patch | 1.256 | 0.920 | 0.342 | +0.107 | +0.020 | +0.000 | +0.007 |
| LSTM-patch | 1.265 ± 0.013 | 0.921 ± 0.009 | 0.324 ± 0.013 | +0.100 ± +0.009 | +0.013 ± +0.010 | -0.007 ± +0.010 | - |
| U-Net (resunet, C=32) | 1.194 ± 0.011 | 0.874 ± 0.011 | 0.372 ± 0.018 | +0.151 ± +0.008 | +0.068 ± +0.009 | +0.049 ± +0.009 | +0.056 ± +0.009 |
| ConvLSTM-ED (resconvlstm_ed, C=16) | 1.217 ± 0.006 | 0.907 ± 0.006 | 0.316 ± 0.017 | +0.134 ± +0.004 | +0.051 ± +0.004 | +0.031 ± +0.004 | +0.038 ± +0.004 |

## subtropical

### lead 1 d  (n pairs 2192)

| method | RMSE | MAE | ACC | skill vs persistence | skill vs Ridge | skill vs Ridge-patch | skill vs LSTM-patch |
|---|---|---|---|---|---|---|---|
| anomaly persistence | 0.122 | 0.083 | 0.972 | +0.000 | +0.047 | +0.009 | +0.049 |
| Ridge | 0.128 | 0.092 | 0.972 | -0.049 | +0.000 | -0.040 | +0.002 |
| Ridge-patch | 0.123 | 0.087 | 0.973 | -0.009 | +0.038 | +0.000 | +0.040 |
| LSTM-patch | 0.128 ± 0.007 | 0.090 ± 0.005 | 0.972 ± 0.002 | -0.051 ± +0.061 | -0.002 ± +0.059 | -0.042 ± +0.061 | - |
| U-Net (resunet, C=32) | 0.125 ± 0.002 | 0.088 ± 0.003 | 0.972 ± 0.001 | -0.026 ± +0.019 | +0.022 ± +0.018 | -0.017 ± +0.019 | +0.025 ± +0.018 |
| ConvLSTM-ED (resconvlstm_ed, C=16) | 0.127 ± 0.003 | 0.090 ± 0.003 | 0.971 ± 0.000 | -0.041 ± +0.023 | +0.007 ± +0.022 | -0.032 ± +0.022 | +0.010 ± +0.022 |

### lead 7 d  (n pairs 2192)

| method | RMSE | MAE | ACC | skill vs persistence | skill vs Ridge | skill vs Ridge-patch | skill vs LSTM-patch |
|---|---|---|---|---|---|---|---|
| anomaly persistence | 0.353 | 0.266 | 0.783 | +0.000 | +0.047 | +0.014 | -0.024 |
| Ridge | 0.370 | 0.285 | 0.769 | -0.049 | +0.000 | -0.034 | -0.075 |
| Ridge-patch | 0.358 | 0.275 | 0.777 | -0.014 | +0.033 | +0.000 | -0.039 |
| LSTM-patch | 0.344 ± 0.005 | 0.263 ± 0.005 | 0.780 ± 0.003 | +0.024 ± +0.015 | +0.069 ± +0.014 | +0.038 ± +0.015 | - |
| U-Net (resunet, C=32) | 0.330 ± 0.001 | 0.251 ± 0.002 | 0.795 ± 0.003 | +0.064 ± +0.003 | +0.108 ± +0.003 | +0.077 ± +0.003 | +0.041 ± +0.004 |
| ConvLSTM-ED (resconvlstm_ed, C=16) | 0.333 ± 0.005 | 0.253 ± 0.004 | 0.790 ± 0.002 | +0.054 ± +0.014 | +0.098 ± +0.014 | +0.067 ± +0.014 | +0.031 ± +0.015 |

### lead 30 d  (n pairs 2192)

| method | RMSE | MAE | ACC | skill vs persistence | skill vs Ridge | skill vs Ridge-patch | skill vs LSTM-patch |
|---|---|---|---|---|---|---|---|
| anomaly persistence | 0.571 | 0.440 | 0.442 | +0.000 | -0.006 | -0.039 | -0.096 |
| Ridge | 0.568 | 0.444 | 0.436 | +0.006 | +0.000 | -0.033 | -0.090 |
| Ridge-patch | 0.550 | 0.429 | 0.453 | +0.037 | +0.032 | +0.000 | -0.055 |
| LSTM-patch | 0.521 ± 0.002 | 0.409 ± 0.002 | 0.445 ± 0.009 | +0.088 ± +0.004 | +0.083 ± +0.004 | +0.053 ± +0.004 | - |
| U-Net (resunet, C=32) | 0.470 ± 0.011 | 0.364 ± 0.009 | 0.533 ± 0.009 | +0.178 ± +0.020 | +0.173 ± +0.020 | +0.146 ± +0.020 | +0.099 ± +0.022 |
| ConvLSTM-ED (resconvlstm_ed, C=16) | 0.480 ± 0.004 | 0.372 ± 0.003 | 0.499 ± 0.009 | +0.161 ± +0.007 | +0.156 ± +0.007 | +0.128 ± +0.008 | +0.080 ± +0.008 |

## RMSE by season (deg C; deep models mean ± std over seeds)

| region | lead | method | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|---|
| kuroshio | 1 | anomaly persistence | 0.223 | 0.298 | 0.282 | 0.216 |
| kuroshio | 1 | Ridge-patch | 0.219 | 0.287 | 0.270 | 0.214 |
| kuroshio | 1 | LSTM-patch | 0.305 ± 0.016 | 0.345 ± 0.010 | 0.280 ± 0.004 | 0.251 ± 0.011 |
| kuroshio | 1 | U-Net | 0.227 ± 0.001 | 0.298 ± 0.000 | 0.285 ± 0.001 | 0.221 ± 0.001 |
| kuroshio | 1 | ConvLSTM-ED | 0.228 ± 0.001 | 0.300 ± 0.001 | 0.287 ± 0.004 | 0.224 ± 0.003 |
| kuroshio | 7 | anomaly persistence | 0.764 | 0.895 | 0.947 | 0.748 |
| kuroshio | 7 | Ridge-patch | 0.775 | 0.847 | 0.907 | 0.762 |
| kuroshio | 7 | LSTM-patch | 0.795 ± 0.009 | 0.872 ± 0.004 | 0.929 ± 0.007 | 0.766 ± 0.009 |
| kuroshio | 7 | U-Net | 0.729 ± 0.004 | 0.831 ± 0.004 | 0.883 ± 0.012 | 0.723 ± 0.009 |
| kuroshio | 7 | ConvLSTM-ED | 0.731 ± 0.001 | 0.836 ± 0.004 | 0.900 ± 0.019 | 0.725 ± 0.008 |
| kuroshio | 30 | anomaly persistence | 1.384 | 1.401 | 1.551 | 1.269 |
| kuroshio | 30 | Ridge-patch | 1.232 | 1.201 | 1.355 | 1.228 |
| kuroshio | 30 | LSTM-patch | 1.229 ± 0.005 | 1.204 ± 0.009 | 1.380 ± 0.037 | 1.238 ± 0.017 |
| kuroshio | 30 | U-Net | 1.172 ± 0.006 | 1.161 ± 0.005 | 1.287 ± 0.010 | 1.147 ± 0.039 |
| kuroshio | 30 | ConvLSTM-ED | 1.182 ± 0.009 | 1.162 ± 0.013 | 1.322 ± 0.048 | 1.192 ± 0.025 |
| subtropical | 1 | anomaly persistence | 0.104 | 0.143 | 0.130 | 0.104 |
| subtropical | 1 | Ridge-patch | 0.108 | 0.145 | 0.127 | 0.107 |
| subtropical | 1 | LSTM-patch | 0.118 ± 0.011 | 0.150 ± 0.011 | 0.127 ± 0.003 | 0.111 ± 0.005 |
| subtropical | 1 | U-Net | 0.108 ± 0.003 | 0.147 ± 0.002 | 0.131 ± 0.002 | 0.108 ± 0.003 |
| subtropical | 1 | ConvLSTM-ED | 0.110 ± 0.003 | 0.150 ± 0.003 | 0.132 ± 0.002 | 0.109 ± 0.003 |
| subtropical | 7 | anomaly persistence | 0.292 | 0.434 | 0.377 | 0.282 |
| subtropical | 7 | Ridge-patch | 0.317 | 0.421 | 0.359 | 0.322 |
| subtropical | 7 | LSTM-patch | 0.302 ± 0.009 | 0.406 ± 0.002 | 0.349 ± 0.006 | 0.308 ± 0.012 |
| subtropical | 7 | U-Net | 0.285 ± 0.004 | 0.398 ± 0.002 | 0.345 ± 0.004 | 0.276 ± 0.002 |
| subtropical | 7 | ConvLSTM-ED | 0.286 ± 0.002 | 0.403 ± 0.005 | 0.347 ± 0.006 | 0.281 ± 0.011 |
| subtropical | 30 | anomaly persistence | 0.521 | 0.653 | 0.601 | 0.496 |
| subtropical | 30 | Ridge-patch | 0.551 | 0.575 | 0.513 | 0.560 |
| subtropical | 30 | LSTM-patch | 0.517 ± 0.011 | 0.545 ± 0.009 | 0.483 ± 0.006 | 0.537 ± 0.006 |
| subtropical | 30 | U-Net | 0.456 ± 0.011 | 0.522 ± 0.003 | 0.466 ± 0.015 | 0.428 ± 0.032 |
| subtropical | 30 | ConvLSTM-ED | 0.463 ± 0.007 | 0.522 ± 0.004 | 0.481 ± 0.010 | 0.449 ± 0.008 |

## Skill vs Ridge-patch by season (mean over seeds)

| region | lead | model | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|---|
| kuroshio | 1 | U-Net | -0.040 | -0.038 | -0.057 | -0.035 |
| kuroshio | 1 | ConvLSTM-ED | -0.045 | -0.043 | -0.063 | -0.045 |
| kuroshio | 7 | U-Net | +0.059 | +0.019 | +0.026 | +0.052 |
| kuroshio | 7 | ConvLSTM-ED | +0.056 | +0.013 | +0.008 | +0.049 |
| kuroshio | 30 | U-Net | +0.048 | +0.033 | +0.050 | +0.066 |
| kuroshio | 30 | ConvLSTM-ED | +0.041 | +0.032 | +0.024 | +0.030 |
| subtropical | 1 | U-Net | -0.002 | -0.015 | -0.035 | -0.007 |
| subtropical | 1 | ConvLSTM-ED | -0.014 | -0.040 | -0.045 | -0.016 |
| subtropical | 7 | U-Net | +0.101 | +0.056 | +0.039 | +0.142 |
| subtropical | 7 | ConvLSTM-ED | +0.098 | +0.043 | +0.033 | +0.127 |
| subtropical | 30 | U-Net | +0.173 | +0.092 | +0.090 | +0.234 |
| subtropical | 30 | ConvLSTM-ED | +0.160 | +0.092 | +0.063 | +0.198 |

## Generated statements (numbers from the table above)

- **U-Net vs Ridge-patch:** positive in 4/6 region-lead cells (by more than one seed-std in 4): kuroshio 1 d -0.043 ± 0.002; kuroshio 7 d +0.037 ± 0.006; kuroshio 30 d +0.049 ± 0.009; subtropical 1 d -0.017 ± 0.019; subtropical 7 d +0.077 ± 0.003; subtropical 30 d +0.146 ± 0.020
- **U-Net vs Lstm-patch:** positive in 6/6 region-lead cells (by more than one seed-std in 6): kuroshio 1 d +0.124 ± 0.001; kuroshio 7 d +0.057 ± 0.006; kuroshio 30 d +0.056 ± 0.009; subtropical 1 d +0.025 ± 0.018; subtropical 7 d +0.041 ± 0.004; subtropical 30 d +0.099 ± 0.022
- **ConvLSTM-ED vs Ridge-patch:** positive in 4/6 region-lead cells (by more than one seed-std in 4): kuroshio 1 d -0.050 ± 0.009; kuroshio 7 d +0.028 ± 0.007; kuroshio 30 d +0.031 ± 0.004; subtropical 1 d -0.032 ± 0.022; subtropical 7 d +0.067 ± 0.014; subtropical 30 d +0.128 ± 0.008
- **ConvLSTM-ED vs Lstm-patch:** positive in 6/6 region-lead cells (by more than one seed-std in 5): kuroshio 1 d +0.118 ± 0.007; kuroshio 7 d +0.049 ± 0.007; kuroshio 30 d +0.038 ± 0.004; subtropical 1 d +0.010 ± 0.022; subtropical 7 d +0.031 ± 0.015; subtropical 30 d +0.080 ± 0.008
- **U-Net vs ConvLSTM-ED, kuroshio (mean RMSE):** 1 d 0.261 / 0.262; 7 d 0.795 / 0.802; 30 d 1.194 / 1.217
- **U-Net vs ConvLSTM-ED, subtropical (mean RMSE):** 1 d 0.125 / 0.127; 7 d 0.330 / 0.333; 30 d 0.470 / 0.480
