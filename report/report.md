# Simple SST forecast baselines in two North Pacific boxes: a test-period evaluation

*Mukhammaddiyor Abdulazimov · Harbin Engineering University · November 2026*

**Question.** How hard are simple baselines to beat for 1-, 7- and 30-day SST forecasts, and does a per-grid-point Ridge regression help? This is a baselines study, not an ocean forecasting model.

**Data.** NOAA OISST v2.1 daily 0.25° SST, two 40x40-point boxes (Kuroshio extension, per-point anomaly std 1.13 °C; subtropical, 0.69 °C), 1982-01-01–2025-12-31 (16,071 days). ERDDAP is the primary source; 1,194 days it lacks were filled from NOAA PSL after checks (double fetch, range, step, exact match on overlaps: max |difference| 3.8e-06 °C).

**Protocol.** Training 1982–2015 (climatology: 3 harmonics; Ridge fit), validation 2016–2019 (Ridge alpha), test 2020–2025 (2192 days per lead) evaluated once after a frozen, tagged configuration. Score: cos(lat)-weighted RMSE; skill = 1 − RMSE/RMSE of anomaly persistence.

**Baselines.** persistence; climatology; anomaly persistence; damped persistence (training-year lag correlation); OCN (mean of the same day over the previous 10 years); Ridge (anomaly lags 0–6 and 30-day mean anomaly, one model per grid point and lead).

*Test results: RMSE in °C (skill vs anomaly persistence).*

| Method | Kuroshio 1 d | Kuroshio 7 d | Kuroshio 30 d | Subtropical 1 d | Subtropical 7 d | Subtropical 30 d |
|---|---|---|---|---|---|---|
| Persistence | 0.268 (-0.039) | 0.983 (-0.166) | 2.554 (-0.817) | 0.124 (-0.019) | 0.391 (-0.110) | 0.913 (-0.597) |
| Climatology | 1.591 (-5.174) | 1.591 (-0.886) | 1.591 (-0.132) | 0.846 (-5.955) | 0.846 (-1.399) | 0.846 (-0.480) |
| Anomaly persistence | 0.258 (+0.000) | 0.843 (+0.000) | 1.405 (+0.000) | 0.122 (+0.000) | 0.353 (+0.000) | 0.571 (+0.000) |
| Damped persistence | 0.262 (-0.017) | 0.854 (-0.012) | 1.324 (+0.058) | 0.135 (-0.109) | 0.415 (-0.178) | 0.623 (-0.091) |
| OCN (10 yr) | 1.390 (-4.394) | 1.390 (-0.648) | 1.390 (+0.011) | 0.595 (-3.890) | 0.595 (-0.687) | 0.595 (-0.041) |
| Ridge | 0.251 (+0.025) | 0.838 (+0.007) | 1.281 (+0.088) | 0.128 (-0.049) | 0.370 (-0.049) | 0.568 (+0.006) |

![Figure: test RMSE and skill by lead.](../results/figures/fig2_test_leads.png)

**Key findings.**

1. **Anomaly persistence is the bar to beat, especially at short range.** Its 1-day RMSE is 0.122 °C (subtropical) and 0.258 °C (Kuroshio), while climatology gives 0.846 and 1.591 °C and OCN 0.595 and 1.390 °C. Ridge's margin over anomaly persistence at 1 day is small (skill +0.025 Kuroshio, -0.049 subtropical).
2. **Ridge helps in the high-variability Kuroshio box, not reliably in the subtropical box.** Kuroshio: Ridge beats anomaly persistence at every lead (skill +0.025, +0.007, +0.088 at 1, 7, 30 d). Subtropical: Ridge is worse at 1 and 7 days (-0.049, -0.049) and about equal at 30 d (+0.006).
3. **Plain persistence fails at 30 days, and the warm shift hurts climatology-based forecasts.** Persistence at 30 d has skill -0.817 (Kuroshio) and -0.597 (subtropical) relative to anomaly persistence. Against the 1982–2015 climatology the test years are warm: mean bias of the climatology forecast is -0.827 °C (Kuroshio) and -0.656 °C (subtropical); Ridge at 30 d is also biased low (-0.442, -0.287 °C).
4. **The ranking changed between validation and test.** Methods that relax towards the 1982–2015 climatology (Ridge, damped persistence) beat anomaly persistence on validation but lost much or all of that advantage on test. Skill vs anomaly persistence at 1, 7, 30 d, validation → test: Ridge Kuroshio +0.041, +0.077, +0.176 → +0.025, +0.007, +0.088; Ridge subtropical +0.022, +0.056, +0.090 → -0.049, -0.049, +0.006; damped persistence Kuroshio +0.010, +0.067, +0.164 → -0.017, -0.012, +0.058; damped persistence subtropical -0.004, -0.017, +0.028 → -0.109, -0.178, -0.091 (negative at every test lead). This is consistent with a larger warm shift in the test years: the climatology forecast's bias is -0.500 / -0.539 °C on validation but -0.827 / -0.656 °C on test (Kuroshio / subtropical), and the 30-day bias of damped persistence goes from -0.324 / -0.325 to -0.527 / -0.392 °C. It is not shown here that the warm shift is the cause.

**LSTM (added after the baselines).** Same inputs as Ridge (30 days of anomaly at one grid point); one LSTM per region, 3 seeds, protocol tagged before the test and run once. RMSE in °C (skill vs Ridge, mean ± std over seeds; validation in brackets).

| LSTM | Kuroshio 1 d | Kuroshio 7 d | Kuroshio 30 d | Subtropical 1 d | Subtropical 7 d | Subtropical 30 d |
|---|---|---|---|---|---|---|
| RMSE (skill vs Ridge) | 0.266 (-0.061) [-0.001] | 0.852 (-0.017) [+0.006] | 1.283 (-0.001) [+0.014] | 0.124 (+0.029) [-0.006] | 0.361 (+0.025) [+0.026] | 0.556 (+0.022) [+0.045] |

**Result (test, H = 64, mean ± std over seeds 0, 1, 2).** The LSTM does **not** beat Ridge in the Kuroshio box (skill vs Ridge -0.061, -0.017, -0.001 at 1, 7, 30 d; on validation it had been -0.001, +0.006, +0.014, so the 7- and 30-day gains did not hold). In the subtropical box it beats Ridge at every lead (skill +0.029, +0.025, +0.022; validation -0.006, +0.026, +0.045), but Ridge itself is worse than anomaly persistence at 1 and 7 days there, and the LSTM does not beat anomaly persistence at those leads either (skill vs anomaly persistence -0.018, -0.023, +0.027; Kuroshio -0.034, -0.010, +0.087). At 30 days the LSTM is better than anomaly persistence in both boxes. ACC at 30 d is slightly lower than that of anomaly persistence in both boxes.

**Fairness caveat.** The LSTM gets the full 30-day sequence plus sin/cos of the day of year; Ridge gets lags 0–6 and the 30-day mean and no day of year, so part of any LSTM gain may come from the extra inputs, not from the nonlinear model (not tested). Validation and test used slightly different pair conventions (see the protocol). This is a learning experiment, not an ocean forecasting model, and the Kuroshio and subtropical results disagree, so no general claim is made.

**B1: spatial context (added after the LSTM; own protocol, tag `pre-b1-test`, run once).** Same pairs, inner 40×40 scored; inputs from a 5×5 patch of neighbouring points (1° halo data). RMSE skill vs Ridge, test [validation]; LSTM-patch mean ± std over 3 seeds.

| B1 | Kuroshio 1 d | Kuroshio 7 d | Kuroshio 30 d | Subtropical 1 d | Subtropical 7 d | Subtropical 30 d |
|---|---|---|---|---|---|---|
| Ridge-patch | +0.006 [+0.001] | +0.015 [+0.020] | +0.020 [+0.012] | +0.038 [+0.000] | +0.033 [+0.031] | +0.032 [+0.025] |
| LSTM-patch | -0.183 [-0.010] | -0.007 [+0.022] | +0.013 [+0.026] | -0.002 [-0.018] | +0.069 [+0.062] | +0.083 [+0.082] |

The day of year alone does not help Ridge; the patch does (Ridge-patch beats Ridge in all six cells). LSTM-patch beats Ridge-patch in the subtropical box at 7 and 30 d (+0.038, +0.053 skill vs Ridge-patch; its validation gain vs Ridge held) but not in the Kuroshio box (-0.190, -0.022, -0.007; validation gains did not hold). Two boxes that disagree, three seeds, one test run: no general claim.

**B2: whole-region models (U-Net, ConvLSTM-ED; own protocol, tag `pre-b2-test`, run once).** Test RMSE skill vs Ridge-patch, mean ± std over 3 seeds [validation]; variant (plain/residual) chosen on validation by code.

| B2 | Kuroshio 1 d | Kuroshio 7 d | Kuroshio 30 d | Subtropical 1 d | Subtropical 7 d | Subtropical 30 d |
|---|---|---|---|---|---|---|
| U-Net | -0.043 ± 0.002 [-0.024] | 0.037 ± 0.006 [+0.014] | 0.049 ± 0.009 [+0.031] | -0.017 ± 0.019 [+0.014] | 0.077 ± 0.003 [+0.035] | 0.146 ± 0.020 [+0.079] |
| ConvLSTM-ED | -0.050 ± 0.009 [-0.027] | 0.028 ± 0.007 [+0.008] | 0.031 ± 0.004 [+0.017] | -0.032 ± 0.022 [+0.005] | 0.067 ± 0.014 [+0.035] | 0.128 ± 0.008 [+0.084] |

Whole-region models (U-Net, ConvLSTM-ED; residual variants chosen on validation) beat Ridge-patch at 7 and 30 d in both boxes (U-Net skill vs Ridge-patch Kuroshio +0.037, +0.049; subtropical +0.077, +0.146) and LSTM-patch in all cells, but not at 1 d (U-Net -0.043, -0.017; also below anomaly persistence). Three seeds, one test run.

**Limitations.** Two small regions; simple statistical baselines; OISST is a smooth interpolated analysis; hybrid ERDDAP + PSL data (verified on overlaps); v2/v2.1 label change at 2016; warm shift vs the 1982–2015 climatology, not detrended; Ridge ignores spatial structure; its alpha sits at the grid's lower edge in several cases; one fixed split.

Code, protocols (`docs/lstm_test_protocol.md`, `docs/b1_spatial_protocol.md`, `docs/b1_test_protocol.md`, `docs/b2_protocol.md`, `docs/b2_test_protocol.md`) and data checks: `README.md`, `results/*.json`. Reference for deep learning in ocean forecasting: Hao et al. 2025 (review).

**References**

1. Huang, B., et al. (2021). Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1. Journal of Climate, 34(8), 2923–2939. doi:10.1175/JCLI-D-20-0166.1
2. Murphy, A. H. (1992). Climatology, persistence, and their linear combination as standards of reference in skill scores. Weather and Forecasting, 7(4), 692–698. doi:10.1175/1520-0434(1992)007<0692:CPATLC>2.0.CO;2
3. Huang, J., van den Dool, H. M., and Barnston, A. G. (1996). Long-lead seasonal temperature prediction using optimal climate normals. Journal of Climate, 9(4), 809–817. doi:10.1175/1520-0442(1996)009<0809:LLSTPU>2.0.CO;2
4. Hao, R., Zhao, Y., Zhang, S., and Deng, X. (2025). Deep learning for ocean forecasting: a comprehensive review of methods, applications, and datasets. IEEE Transactions on Cybernetics, 55(6), 2879–2898. doi:10.1109/TCYB.2025.3539990

*Code written with Claude Code; methodology, decisions and verification by the author.*
