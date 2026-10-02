# PESTGM 7.0 — Track 1
## National Intelligent Platform for Forecasting Rooftop Solar Production (Tunisia)

> **Status: 37/40 Definition-of-Done items PASS, 1 PARTIAL, 2 BLOCKED.**
> This session added a real, measured external-validation dataset (ENSTAB/
> Borj Cedria) — see `reports/REAL_REFERENCE_MODEL_VALIDATION.md`.
> See `FINAL_CDC_COMPLIANCE_MATRIX.md` and `docs/REQUIREMENTS_TRACEABILITY_MATRIX.md`
> for the itemized, honest audit — nothing here is marked done unless it was
> actually implemented and tested (`pytest tests/`: 24/24 passing).

## 0. The most important thing to know: DATA_MODE

```bash
DATA_MODE=demo   # default — allows the labelled DEMO/PROXY physics-simulation target
DATA_MODE=real   # PRODUCTION mode — REFUSES to run and raises RealDataUnavailableError
                 # naming exactly what's missing, rather than silently using synthetic data
```
Try it yourself:
```bash
DATA_MODE=real python scripts/04_build_training_dataset.py --start 2026-06-25 --end 2026-07-05
# -> RealDataUnavailableError: DATA_MODE=real requires 'real PV production
#    (data/raw/pv_production_real.csv)' but it is unavailable...
```
This is enforced in code (`src/ingestion/data_mode.py`), not just documented.

## 1. What this repo delivers

Given the official Prosol fleet extract (50 districts, 3 snapshots), this repo:
1. builds a synchronized, ML-ready dataset (fleet capacity + weather + solar geometry + labelled production target),
2. compares 4 forecasting baselines chronologically (persistence, seasonal persistence, physics, LightGBM),
3. produces P10-P25-P50-P75-P90 uncertainty via quantile LightGBM + split-conformal calibration,
4. reconciles district -> governorate -> national forecasts exactly (bottom-up),
5. runs a residual-correction experiment (raw vs bias-corrected forecast),
6. serves everything through a FastAPI backend,
7. is covered by 24 automated tests (fleet, capacity, no-leakage, physical constraints, reconciliation, uncertainty, correction, API).

**Every one of 2-6 above is run against the labelled `DEMO_PROXY_PHYSICS_SIMULATION`
target** — see §2.

## 2. Honesty about data (read this before anything else)

| Data | Status |
|---|---|
| PV fleet capacity | **REAL** (official Prosol snapshots) |
| Historical/forecast weather | Real provider implemented (Open-Meteo); unreachable from this build sandbox (HTTP 403) — confirmed reachable via a 3rd-party wrapper at **daily** resolution only. Shipped sample data is `SYNTHETIC_CLEARSKY`. |
| PV production | **No real measured source found anywhere public for Tunisia** (STEG, data.gov.tn, ANME, IRENA, JODI all checked — see `docs/REAL_PRODUCTION_DATA_SOURCES.md`). Demo target is `DEMO_PROXY_PHYSICS_SIMULATION`. |

Full detail: `docs/DATA_READINESS_REPORT.md` (REAL/DERIVED/MODEL OUTPUT/DEMO ONLY
classification) and `docs/REAL_PRODUCTION_DATA_SOURCES.md` (every source
searched, what was and wasn't found).

## 3. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Validate the fleet CSV
python -m src.ingestion.fleet_loader

# 2. Build historical weather (DATA_MODE=demo allows the labelled fallback;
#    DATA_MODE=real requires real Open-Meteo access and refuses otherwise)
DATA_MODE=demo python scripts/02_build_weather_dataset.py --start 2026-06-25 --end 2026-07-05

# 3. Build the final synchronized 15-min training dataset
DATA_MODE=demo python scripts/04_build_training_dataset.py --start 2026-06-25 --end 2026-07-05

# 4. Compare models, backtest chronologically, uncertainty, reconciliation, correction
DATA_MODE=demo python scripts/05_train_and_backtest.py
# -> writes reports/MODEL_VALIDATION_REPORT.md

# 5. Actually train + backtest a separate model per horizon (intraday/J+1/J+2/J+3)
DATA_MODE=demo python scripts/06_multi_horizon_backtest.py
# -> writes reports/MULTI_HORIZON_REPORT.md

# 6. Run the continuous-learning / model-update workflow
DATA_MODE=demo python scripts/07_update_forecast_model.py --trigger cli
# -> saves a versioned model under models/registry/<timestamp>/

# 7. Run the API (13 endpoints)
uvicorn src.api.main:app --reload --port 8000
# -> http://localhost:8000/docs

# 8. Run the dashboard (10 pages + Tunisia map)
streamlit run src/dashboard/app.py

# 9. Run the tests
pytest tests/ -v

# Optional: time-aware XGBoost hyperparameter tuning (compares against LightGBM)
DATA_MODE=demo python scripts/08_xgboost_tuning.py

# Optional: attempt real PVGIS physical reference dataset (stops & documents if unreachable)
DATA_MODE=demo python scripts/10_build_pv_reference_dataset.py

# Optional: external validation against real ENSTAB/Borj Cedria measured PV data
DATA_MODE=demo python scripts/09_enstab_external_validation.py
```

Outputs land in `data/processed/`. The repo ships a small **sample** dataset
so it stays lightweight in git:
- `weather_historical_sample.csv`
- `training_dataset_15min_sample.csv` — full feature table (40 columns: fleet, weather, solar geometry, cyclic time, physics target)
- `pv_production_simple_view_sample.csv` — the minimal `timestamp, district, pv_production_mw` view

**Note on report reproducibility:** `reports/MULTI_HORIZON_REPORT.md`'s J+2/J+3
numbers were generated from a **46-day** run (`--start 2026-05-20 --end 2026-07-05`)
so those horizons have enough real span to backtest meaningfully; the 10-day
sample shipped in git is for quick interactive exploration (dashboard, API,
intraday scripts) and is too short for a meaningful J+2/J+3 backtest on its
own — regenerate with the longer range first if you want to reproduce that report.

Regenerate any period you need (e.g. the full `2022-01-01` → today for real
model training once real weather is reachable):
```bash
python scripts/02_build_weather_dataset.py --start 2022-01-01 --end 2026-07-31 --out data/processed/weather_historical.csv
python scripts/04_build_training_dataset.py --start 2022-01-01 --end 2026-07-31 --weather_in data/processed/weather_historical.csv
```
(large outputs like this are `.gitignore`d — keep them out of version control, e.g. use DVC/S3/a data lake in production).

Switching to real production data later requires **no architecture change**:
drop `data/raw/pv_production_real.csv` (`timestamp, district_id, production_mw`)
and run with `DATA_MODE=real` — the feature pipeline and models already read
that schema.

## 4. Repository structure

```
data/{raw,processed,external,synthetic,validation}
src/
  ingestion/    fleet_loader.py, capacity_timeseries.py, weather_provider.py, data_mode.py
  features/     solar_geometry.py, lag_features.py
  physics/      pv_model.py        (baseline + proxy target generator)
  models/       baselines.py       (persistence, seasonal persistence, physics, LightGBM)
  forecasting/  backtesting.py     (chronological split, metrics, multi-horizon labels)
  uncertainty/  quantile_forecast.py (quantile LightGBM + split-conformal)
  calibration/  residual_correction.py (rolling bias + ML residual correctors)
  aggregation/  hierarchical.py    (bottom-up district->governorate->national)
  validation/   physical_checks.py (bounds, night-zero, jump, daylight consistency)
  api/          main.py            (FastAPI)
  dashboard/    (next phase — Streamlit / map, see FINAL_CDC_COMPLIANCE_MATRIX.md)
configs/config.yaml   single source of truth for paths, providers, physical assumptions
scripts/               01-05, numbered by pipeline phase
docs/                  DATA_READINESS_REPORT, REAL_PRODUCTION_DATA_SOURCES,
                        REQUIREMENTS_TRACEABILITY_MATRIX, LEAKAGE_AUDIT, DEMO_SCRIPT
reports/               MODEL_VALIDATION_REPORT.md (generated by scripts/05)
tests/                 24 tests, pytest
FINAL_CDC_COMPLIANCE_MATRIX.md
```

## 5. Design decisions worth knowing

- **Capacity interpolation**: linear between the 3 official snapshots, held
  constant outside their range. Simple, transparent, and designed so a 4th
  monthly snapshot slots in with zero code change (`capacity_timeseries.py`).
- **Weather resolution**: Open-Meteo's historical/forecast APIs are hourly.
  The pipeline upsamples to 15-min via linear interpolation and tags every
  row `weather_resolution_note=resampled_from_hourly_interpolation` — the
  CDC explicitly requires *not* claiming interpolation creates genuine
  15-minute information (section 9); this repo enforces that distinction in
  the data itself, not just in prose.
- **Physics model**: `Capacity x (GHI/1000) x NOCT-based temperature derating
  x performance_ratio`, clipped to `[0, Capacity]`, forced to 0 at night
  (`solar_elevation <= 0`). Performance ratio is a per-district random draw
  (mean 0.80, std 0.03) — a documented assumption, not a measurement.
- **Proxy production noise**: small heteroskedastic Gaussian noise added
  *before* clipping to physical bounds, so noise can never push production
  outside `[0, capacity]` or make it non-zero at night (both enforced by
  runtime assertions in `scripts/04_build_training_dataset.py`, mirroring
  CDC section 26 critical tests 3–4).

## 6. Roadmap (remaining CDC items — see FINAL_CDC_COMPLIANCE_MATRIX.md)

**BLOCKED** (documented external dependency, not a code gap):
- Real hourly weather at scale (network-restricted sandbox — code is ready; reachable at daily resolution via a 3rd-party relay)
- Real PV production (none found publicly for Tunisia — see `docs/REAL_PRODUCTION_DATA_SOURCES.md`)

**PARTIAL**:
- PVGIS integration (`src/ingestion/pvgis_provider.py` implemented; blocked by PVGIS's own bot protection from this sandbox)

Everything else — J+1/J+2/J+3 (actually trained+backtested per horizon),
continuous learning (`scripts/07`), drift/anomaly monitoring, SHAP
explainability, full 13-endpoint API, Streamlit dashboard + Tunisia map — is
now **PASS**. See `FINAL_CDC_COMPLIANCE_MATRIX.md` for the itemized evidence.

## 7. Limitations (stated up front, per CDC Rule 2/4)

- No empirical model-performance claim is made anywhere in this repo — there
  is no real production data to validate against yet.
- Weather in the shipped sample datasets is synthetic; swap in real
  Open-Meteo data before any result is treated as meaningful.
- Fleet coordinates are representative points, not official administrative
  polygons (per source dataset).
- Uncertainty coverage degrades at longer horizons on the DEMO dataset
  (intraday 88%, J+3 57% vs 80% nominal) — reported honestly in
  `reports/MULTI_HORIZON_REPORT.md`, not smoothed over.
- The raw intraday LightGBM forecast is non-zero at night on ~22% of test
  rows (a real model imperfection found by the physical-check pipeline,
  not yet corrected by a post-processing clip).
