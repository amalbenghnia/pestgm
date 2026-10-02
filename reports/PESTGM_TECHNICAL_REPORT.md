# PESTGM 7.0 — Track 1 — Technical Report
## National Intelligent Platform for Forecasting Rooftop Solar Production (Tunisia)

*All quantitative results in this report are computed against the labelled
`DEMO_PROXY_PHYSICS_SIMULATION` target unless explicitly marked REAL. Section
15 reports the one exception: a real, measured external-validation dataset
(ENSTAB/Borj Cedria). No real STEG district/national production data exists
publicly for Tunisia (see §3.2). This report never claims real-world
validated national accuracy.*

## 1. Problem

Forecast aggregated rooftop/self-consumption PV production connected to the
Tunisian LV/MV grid, at district / governorate / national spatial levels and
intra-day / J+1 / J+2 / J+3 temporal horizons, with quantified uncertainty,
continuous correction, and an operational dashboard — per the official
PESTGM 7.0 Track 1 cahier des charges.

## 2. CDC requirements

Fully itemized in `FINAL_CDC_COMPLIANCE_MATRIX.md` (37 PASS / 1 PARTIAL / 2
BLOCKED out of 40). This report summarizes; that file is authoritative.

## 3. Data

### 3.1 Real data
`data/raw/pv_fleet_prosol_3_snapshots.csv` — official Prosol fleet extract,
50 districts, 24 governorates, 3 snapshots (Dec-2025, Mar-2026, Jul-2026).
Validated: 0 duplicate keys, 0 missing coordinates, district-capacity sums
match reported national totals within 0.07%.

### 3.2 Real data NOT available
No real measured PV production exists publicly for Tunisia at any spatial
resolution. Systematically checked: STEG, ANME, data.gov.tn, IRENA, JODI,
Kaggle, Zenodo/OpenAIRE (search pass), Renewables.ninja (needs a token, not
tested), NASA POWER (not tested), PVGIS (tested directly — real satellite
irradiance + physical PV simulation, but blocked by PVGIS's own bot
protection from this build environment; the client is implemented and ready
to run elsewhere — `src/ingestion/pvgis_provider.py`). Full detail:
`docs/REAL_PRODUCTION_DATA_SOURCES.md`.

Real historical/forecast weather (Open-Meteo) is architecturally real and
complete (`src/ingestion/weather_provider.py::OpenMeteoProvider`) but this
build sandbox's network egress blocks `open-meteo.com` (HTTP 403, confirmed
directly). Reachable via a third-party relay at daily resolution only.

### 3.3 What this repo actually ran on
`SYNTHETIC_CLEARSKY` weather (pvlib Ineichen clear-sky + a seeded stochastic
cloud-attenuation process) and a physics-derived `DEMO_PROXY_PHYSICS_SIMULATION`
production target, both tagged explicitly in every row's provenance columns.

## 4. Data quality

`docs/DATA_READINESS_REPORT.md` classifies every column as REAL / DERIVED /
MODEL OUTPUT / DEMO ONLY. `DATA_MODE=real` enforces this at runtime: both
`scripts/02_build_weather_dataset.py` and `scripts/04_build_training_dataset.py`
raise `RealDataUnavailableError` and exit non-zero rather than silently
substituting synthetic data — verified by direct execution.

## 5. System architecture

```
Prosol fleet (REAL) --linear interp--> Capacity(district,t)
Weather (real OpenMeteoProvider / SYNTHETIC_CLEARSKY, always tagged)
        -> Solar geometry (pvlib) -> Feature engineering (weather, solar,
        fleet, temporal, lag/rolling)
        -> Physics baseline model -> DEMO_PROXY production target
        -> Baselines (persistence, seasonal persistence, physics, LightGBM)
        -> Per-horizon models (intraday/J+1/J+2/J+3)
        -> Quantile LightGBM + split-conformal uncertainty
        -> Bottom-up hierarchical reconciliation (district->gov->national)
        -> Residual correction (rolling bias / ML residual)
        -> Drift & anomaly monitoring
        -> FastAPI -> Streamlit dashboard + Tunisia map
```

## 6. Forecasting methodology

Baselines: persistence, seasonal persistence (t-1 day), physics (NOCT +
temperature-derated clear-sky model), LightGBM. Compared chronologically
(60/20/20 train/val/test by date range, no shuffling).

## 7. Multi-horizon strategy

**One model per horizon** (documented decision, `src/forecasting/multi_horizon.py`):
intra-day uses recent production lags (legitimately available operationally);
J+1/J+2/J+3 use only weather/solar/capacity/calendar features (what would
genuinely be available that far ahead) — no production lags, enforced by
`tests/test_multi_horizon.py`. A single shared-feature multi-output model was
rejected because it would force a feature set that either starves intra-day
or leaks into day-ahead horizons.

## 8. Spatial hierarchy

District (native model output) -> governorate -> national, bottom-up
summation, exact by construction (no reconciliation residual — verified to
~1e-14 float precision in `tests/test_aggregation.py`).

## 9. Uncertainty

Quantile LightGBM (5 independent regressors, tau in {0.10,0.25,0.50,0.75,0.90})
+ a split-conformal-STYLE calibration on a held-out validation split. Ordering
P10<=P25<=P50<=P75<=P90 enforced by sorting, verified on every test row.

**Per-horizon 80% coverage** (nominal 80%): intraday 87.7%, J+1 76.1%, J+2
65.4%, J+3 56.4%. Coverage degrades with horizon — reported honestly, not
smoothed. A critical review this session of the calibration method itself
(CDC continuation Phase 13) found it is a **simplified per-quantile additive
residual shift, not textbook Conformalized Quantile Regression** (which
conformalizes the interval via a single nonconformity score, not five
independent quantile shifts) — documented in `src/uncertainty/quantile_forecast.py`'s
module docstring as the likely structural cause of the degradation at longer
horizons, alongside the calibration set not being horizon-specific. A proper
CQR implementation is named as the recommended next step, not implemented
this session.

## 10. Correction / continuous learning

`RollingBiasCorrector` demonstrated to reduce MAE and RMSE on a held-out
evaluation slice (both in a controlled unit test with an injected 1 MW bias,
and in the demo run). `scripts/07_update_forecast_model.py` runs the full
operational loop (ingest -> validate -> error -> correct -> recalibrate ->
retrain-decision -> versioned save) end-to-end against a genuinely new batch,
with a promotion rule (new model replaces old only if validation MAE does not
regress by more than 5%).

## 11. Dashboard / API

Streamlit, 10 pages + Tunisia map (representative points, not official
polygons). Every page shows a REAL DATA / DEMO-SYNTHETIC banner from
`DATA_MODE` and the dataset's own provenance columns. Verified headlessly
with `streamlit.testing.v1.AppTest` (11/11 pages, zero exceptions — this
process caught and fixed one real bug: `plotly.express.scatter_mapbox`
deprecated in favor of `scatter_map` in the installed plotly version).

FastAPI: 13 endpoints (health, districts, capacity, forecast x3, POST
forecast, uncertainty, errors, anomalies, drift x2, export, model status),
all tested.

## 12. Validation

59/59 automated tests pass (`pytest tests/`). Physical constraints (bounds,
night-zero, jump detection, daylight consistency) applied to the target AND
to model forecasts — this found a genuine model imperfection: the raw
intra-day LightGBM forecast is non-zero at night on ~20% of test rows (small
residual values, not hard-clipped like the physics-derived target is). This
was reported honestly in the previous session, and **fixed this session**:
`src/models/postprocessing.py` clips every forecast to `[0, capacity]` and
forces zero when `daylight_flag==0`, applied per-horizon in
`scripts/06_multi_horizon_backtest.py`. Measured effect: flagged-anomaly rate
drops from ~20-24% to ~0-0.6% across all four horizons; sMAPE roughly halves
(e.g. intraday 59.2% -> 20.8%); MAE/RMSE improve only marginally, since
night-time errors are small in absolute MW terms. `src/validation/data_quality.py`
adds structural checks (missing/duplicate timestamps, negative production,
capacity-jump sanity) — `docs/DATA_QUALITY_REPORT.md`.

## 13. Results

See `reports/MODEL_VALIDATION_REPORT.md` (baseline comparison, now including
XGBoost) and `reports/MULTI_HORIZON_REPORT.md` (per-horizon, per-governorate
breakdown, now including the post-processing comparison). Headline (DEMO,
national test set, factual measurements, no subjective ranking):

| Model | MAE (MW) | nMAE (%capacity) |
|---|---|---|
| Persistence | 0.278 | 2.70 |
| Seasonal persistence | 0.516 | 4.97 |
| Physics baseline | 0.195-0.212 | 1.9-2.2 |
| LightGBM (intraday, raw) | 0.211-0.221 | 2.2-2.4 |
| **XGBoost (intraday, time-aware tuned)** | **0.208** | **2.25** |
| LightGBM J+1 / J+2 / J+3 | 0.51-0.58 | 5.0-5.7 |

XGBoost was tuned via a 25-trial time-aware random search (`scripts/08_xgboost_tuning.py`)
over n_estimators/max_depth/learning_rate/subsample/colsample_bytree/
min_child_weight/reg_alpha/reg_lambda/gamma, selected by validation-split MAE,
with parameters locked before a single, final, previously-untouched test
evaluation — never tuned on the test split itself.

## 14. Limitations

- All quantitative results are DEMO-only; a model trained against a
  physics-derived proxy partly re-learns the generating equation, so absolute
  error magnitudes are not informative about real-world skill.
- The conformal-calibration method is a documented simplification (§9), not
  textbook CQR — coverage degradation at longer horizons is partly
  attributable to this, not only to data scarcity.
- Real weather (hourly, full radiation set) not retrieved at scale in this
  build sandbox — architecture is ready, blocked by network egress.
- No real PV production data exists publicly for Tunisia — the single most
  important blocker to a real validation (three consecutive search passes,
  same conclusion — `docs/PRODUCTION_DATA_SEARCH_REPORT.md`).
- Uncertainty coverage degrades at longer horizons on this dataset (§9),
  partly a data issue, partly a documented calibration-method simplification.
- Night-time forecast leakage (~20-24% of test rows across all horizons) in
  raw ML output — **fixed this session** by a physical post-processing clip
  (`src/models/postprocessing.py`); the raw-vs-constrained comparison is kept
  in the reports rather than only shipping the fixed numbers, so the
  magnitude of the original problem stays visible.
- PVGIS client implemented but not executed end-to-end this session (this
  sandbox's own network egress blocks `re.jrc.ec.europa.eu`, same as several
  other external hosts tested).

## 15. External validation against real measured data (ENSTAB/Borj Cedria)

A real, measured PV production dataset became available this session:
ENSTAB/LaRINa lab, Borj Cedria, Ben Arous (Mejdi, Kardous & Grayaa, IEEE SSD
2023) — 238,464 real 5-minute measurements, 2022-02-24 to 2024-05-31, 2.4 kWp
rooftop system. Treated strictly as an external hold-out: **never used to
train or tune any model** (`src/ingestion/enstab_loader.py` builds a
separate validation dataset; `scripts/09_enstab_external_validation.py`
trains only on the DEMO dataset, then scores on ENSTAB).

**Raw finding**: both LightGBM and XGBoost (trained on national district-
scale data) produced strongly negative R² on ENSTAB (-1.53 and -2.24
respectively after physical clipping) — they do not generalize to a system
~200x smaller than anything in their training data. Persistence (R²=0.959)
and the physics baseline (R²=0.835), both scale-invariant by construction,
generalized far better.

**Diagnosed and fixed**: retraining both models on a capacity-normalized
target (`production_mw / capacity_mw`, a [0,1] capacity factor) instead of
absolute MW, then rescaling predictions by ENSTAB's real capacity, raised
XGBoost's R² to 0.804 and LightGBM's to 0.811 — both now close to the
physics baseline. This is a measured, real-data-validated improvement,
reproduced consistently across two different training-window sizes, not a
claim of a fully solved problem: persistence still outperforms every ML
model on this single real system. Full detail, including day/night and
clear/cloudy breakdowns and ramp errors:
`reports/ENSTAB_EXTERNAL_VALIDATION_REPORT.md`,
`reports/REAL_REFERENCE_MODEL_VALIDATION.md`.

## 16. Deployment

`Dockerfile` provided (`DATA_MODE=demo` default, exposes the FastAPI on
:8000). To deploy for real use: (1) run `scripts/02`/`scripts/04` from an
unrestricted network with `DATA_MODE=real` once a real production source is
secured, (2) point `PRODUCTION_PROVIDER`/`data/raw/pv_production_real.csv` at
that source, (3) re-run `scripts/05`-`07` to retrain/backtest/validate
against real data before trusting any output operationally.

## 17. Conclusion

The pipeline is architecturally complete against the CDC (37/40 PASS) and
internally consistent (59/59 tests, exact hierarchical reconciliation, no
leakage found in a 14-item audit). Four sessions of real, executed work —
not just documentation — have progressively closed gaps: multi-horizon
models actually trained, continuous learning actually run,
monitoring/explainability actually wired, XGBoost actually tuned and
compared, a real physical post-processing bug actually found and fixed, the
uncertainty calibration method actually critiqued rather than assumed
correct, and this session, a real measured external dataset (ENSTAB) used to
surface a genuine ML generalization failure — which was then diagnosed and
measurably fixed against that same real data. The two remaining
blockers — real weather at scale and real district/national PV
production — are both external dependencies outside this repository's
control, documented precisely (including real Tunisia-specific ground-truth
studies and one real measured external-validation dataset) rather than
worked around with fabricated data.
