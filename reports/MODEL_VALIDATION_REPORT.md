# MODEL_VALIDATION_REPORT.md

**ALL numbers in this report were computed against the labelled DEMO/PROXY production target (`pv_production_mw_proxy`, `pv_production_source=DEMO_PROXY_PHYSICS_SIMULATION`). They demonstrate the pipeline is complete and correct; they are NOT a measurement of real-world forecasting skill, because no real STEG production data exists to validate against (see docs/REAL_PRODUCTION_DATA_SOURCES.md).**

- Dataset: `data/processed/training_dataset_15min.csv`, 438,050 rows, 50 districts, 2020-01-01 00:00:00 -> 2020-12-31 00:00:00
- Chronological split: train <= 2020-08-07 00:00:00, val <= 2020-10-19 00:00:00, test <= 2020-12-31 00:00:00 (fractions 60/20/20 of the date range, no random shuffling)

## Model comparison (test set, national-level metrics)

|                      |     n |   MAE_MW |   RMSE_MW |   nMAE_pct_of_capacity |   nRMSE_pct_of_capacity |   bias_MW |   sMAPE_pct |
|:---------------------|------:|---------:|----------:|-----------------------:|------------------------:|----------:|------------:|
| persistence          | 87600 |    0.33  |     0.765 |                  3.919 |                   7.044 |     0     |      35.675 |
| seasonal_persistence | 87600 |    0.168 |     0.637 |                  2.02  |                   5.913 |     0.003 |      10.97  |
| physics_baseline     | 87600 |    0.498 |     1.086 |                  5.899 |                   9.948 |    -0.034 |      53.665 |
| lightgbm_mw          | 87600 |    0.154 |     0.452 |                  2.166 |                   5.19  |     0.077 |      52.364 |
| xgboost_mw           | 87600 |    0.151 |     0.465 |                  2.111 |                   5.211 |     0.087 |      56.229 |
| lightgbm_cf          | 87600 |    0.145 |     0.455 |                  1.722 |                   4.243 |     0.074 |      65.223 |
| xgboost_cf           | 87600 |    0.135 |     0.447 |                  1.609 |                   4.202 |     0.077 |      89.391 |

## Uncertainty (quantile LightGBM + split-conformal)

- Quantile ordering (P10<=P25<=P50<=P75<=P90) respected on every test row: **True**
- 80% interval (P10-P90) empirical coverage: **88.3%** (nominal 80%)
- Mean interval width: **0.59 MW**
- Pinball losses: {'pinball_0.1': 0.047121630264874134, 'pinball_0.25': 0.0809617789304352, 'pinball_0.5': 0.08859198578654366, 'pinball_0.75': 0.0628789645311885, 'pinball_0.9': 0.03639582976135741}

## Hierarchical reconciliation

- Bottom-up reconciliation, sum(district)==governorate and sum(governorate)==national verified exactly: {'governorate_consistent': True, 'national_consistent': True, 'max_governorate_diff': 0.0, 'max_national_diff': 2.842170943040401e-14}

## Residual correction (LightGBM raw forecast, rolling bias corrector)

```json
{
  "raw": {
    "n": 43800,
    "MAE_MW": 0.12091274066418092,
    "RMSE_MW": 0.37848133675562207,
    "nMAE_pct_of_capacity": 1.4664068912238255,
    "nRMSE_pct_of_capacity": 3.8102172842021895,
    "bias_MW": 0.06529135760310933,
    "sMAPE_pct": 87.15495062813774
  },
  "corrected": {
    "n": 43800,
    "MAE_MW": 0.1085676379939021,
    "RMSE_MW": 0.36141810136191754,
    "nMAE_pct_of_capacity": 1.322980324723284,
    "nRMSE_pct_of_capacity": 3.6408409512206976,
    "bias_MW": 0.026499349146140046,
    "sMAPE_pct": 9.689564989536324
  }
}
```

## Physical validation of the target series

{'check_bounds_ok': 1.0, 'check_night_zero_ok': 0.9610592398128067, 'check_no_jump_ok': 0.999849332268006, 'check_daylight_consistency_ok': 0.9610592398128067, 'pct_flagged_anomalous': 3.9091427919187307}

## Physical post-processing (LightGBM, raw vs constrained)

Raw forecast anomaly rate: 38.17% (mostly non-zero-at-night). After clipping to `[0, capacity]` and forcing zero when `daylight_flag==0`: 0.00%. sMAPE improves from 89.4% to 26.7% (MAE/RMSE change marginally, since night errors are small in absolute MW).

## Limitations

- Results are against a physics-derived proxy target - a model trained on it partly re-learns the physics equation used to generate it, so absolute error magnitudes are not informative about real-world skill; only structural correctness (ordering, coverage, reconciliation, correction direction) should be read from this report.
- Only the intra-day (15-min) horizon was backtested in this run; J+1/J+2/J+3 horizons are supported by `src/forecasting/backtesting.py::shift_to_horizon` but were not separately retrained/backtested here (next iteration).
- ML baseline excludes production lag features to stay honest about what would be available operationally at longer horizons; intra-day performance would improve with lag features once real production data justifies using them.
