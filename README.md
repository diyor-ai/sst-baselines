# sst-baselines

SST forecast baselines and a reproducible evaluation for two small North Pacific regions
(OISST v2.1, variable `sst` only). This is a baselines study, not an ocean forecasting model.

Placeholder README — status: day 1 (data, QC, figure 1).

## Data notes
- **1996 cold event (kept unchanged):** the record minimum of the Kuroshio box, -1.14 degC, is on 1996-03-30 at
  39.875N, 146.375E (source: NOAA PSL gap fill; the neighbouring days 03-29 and 03-31 come from ERDDAP and agree).
  It is not a download or processing artefact: two independent sources agree, it is smooth in time (~10 days) and
  spatially coherent. It is, however, extreme for the Oyashio region in March (about 8 degC below the local mean,
  z = -3.7 vs other years) and may reflect the AVHRR-only OISST analysis of that era; it lies in the training period.
