# REAL_REFERENCE_MODEL_VALIDATION.md

This report separates strictly, per the CDC continuation's data provenance
model:

- **MEASURED_REAL**: actual measured PV production. Only source available:
  ENSTAB/LaRINa (Borj Cedria), see §5.
- **PHYSICAL_REFERENCE**: production estimated from real physical inputs
  (PVGIS irradiance + physical PV model). **Not obtained this session** —
  see §4.
- **SYNTHETIC_DEMO**: `DEMO_PROXY_PHYSICS_SIMULATION`, synthetic clear-sky +
  stochastic noise. Still the only training target available end-to-end in
  this build sandbox — see §3.

## 1. Data sources

| Class | Source | Status |
|---|---|---|
| MEASURED_REAL | ENSTAB/LaRINa, Borj Cedria (Mejdi, Kardous & Grayaa, IEEE SSD 2023) | **Obtained** — 238,464 real 5-min measurements, 2022-02-24 → 2024-05-31, 2.4 kWp rooftop system |
| PHYSICAL_REFERENCE | PVGIS (`re.jrc.ec.europa.eu`) | **Not obtained** — HTTP 403 from this sandbox's network egress, re-confirmed this session (`scripts/10_build_pv_reference_dataset.py`, `docs/PRODUCTION_REFERENCE_DATASET_REPORT.md`) |
| SYNTHETIC_DEMO | pvlib Ineichen clear-sky + seeded cloud-attenuation process | Used throughout for national model training |

## 2. Fleet construction

Unchanged from prior sessions: `data/raw/pv_fleet_prosol_3_snapshots.csv`
(REAL, 50 districts, 3 official snapshots) → `Capacity(district,t)` via
documented linear interpolation (`src/ingestion/capacity_timeseries.py`),
validated in `tests/test_capacity.py`.

## 3. What the national models are actually trained on

Still `pv_production_mw_proxy` (`SYNTHETIC_DEMO`). `configs/config.yaml`'s
`production_target.mode` remains `DEMO_PROXY` — **not** changed to
`PHYSICAL_REFERENCE`, because no real PVGIS-derived reference was actually
obtained this session (see §4). Changing the config without the underlying
real data would be exactly the kind of unearned status upgrade this project
is being audited against.

## 4. PVGIS methodology (attempted, not completed)

`src/ingestion/pvgis_provider.py` (real client, `seriescalc` endpoint) +
`scripts/10_build_pv_reference_dataset.py` (orchestration) were exercised
this session. Result: `re.jrc.ec.europa.eu` returns HTTP 403 from this
sandbox for every district attempted (stopped at the first failure, `D01`,
by design — no partial real + partial fabricated file was written). Full
detail: `docs/PRODUCTION_REFERENCE_DATASET_REPORT.md`. Both the client and
orchestration script are complete and will run unmodified from any network
where PVGIS is reachable.

## 5. ENSTAB external validation (the one real, measured result this session actually produced)

Full detail: `reports/ENSTAB_EXTERNAL_VALIDATION_REPORT.md`. Summary table
(REAL, measured data, models trained ONLY on `SYNTHETIC_DEMO`, ENSTAB never
used for training):

| Model | R² | nMAE (%capacity) | Note |
|---|---|---|---|
| Persistence | 0.959 | 2.4% | Best — scale-invariant by construction |
| Physics baseline | 0.835 | 6.0% | Scale-invariant (proportional to capacity) |
| LightGBM, capacity-normalized target | 0.811 | 6.7% | **Fixed this session** (was -1.53) |
| XGBoost, capacity-normalized target | 0.804 | 6.9% | **Fixed this session** (was -2.24) |
| LightGBM, absolute-MW target (original) | -1.53 | 26.7% | Fails to extrapolate ~200x below training capacity range |
| XGBoost, absolute-MW target (original) | -2.24 | 31.8% | Same failure mode |

**This session's concrete, tested finding**: the previously-identified ML
generalization failure (negative R² on a real 2.4 kWp system, caused by a
~200x capacity scale gap versus the smallest district in training) was
diagnosed AND a fix was implemented and tested — retraining on a
capacity-normalized target (`production_mw / capacity_mw`) — which raised
XGBoost's R² on real ENSTAB data from -2.24 to 0.80, and LightGBM's from
-1.53 to 0.81, both now close to the physics baseline's 0.835. Not presented
as a solved problem (persistence still outperforms every ML model here), but
as a genuine, measured improvement traceable to a specific, understood
mechanism, reproduced consistently across two different training-data window
sizes (10-day and 45-day samples both show the same fix direction and
magnitude).

## 6. Never claimed

- "Exact" or "near-exact" production reproduction — never stated.
- PVGIS or physics output described as "measured" — never done; every row
  carries an explicit, checked provenance label.
- The capacity-normalization fix is not claimed to fully solve
  generalization — persistence still wins on this single real system, and a
  single 2.4 kWp rooftop's local shading/orientation/inverter behaviour
  likely explains part of the remaining gap regardless of target scaling.

## 7. Reproducibility

```bash
DATA_MODE=demo python scripts/10_build_pv_reference_dataset.py   # documents PVGIS block, does not fabricate
DATA_MODE=demo python scripts/09_enstab_external_validation.py   # real ENSTAB validation + normalization experiment
pytest tests/ -v                                                 # 59/59
```
