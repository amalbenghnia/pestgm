# DEMO_SCRIPT.md — 3-5 minute live demo

**Say up front:** "Everything here runs against a clearly-labelled DEMO/PROXY
dataset — no real STEG production data exists publicly (documented in
`docs/REAL_PRODUCTION_DATA_SOURCES.md`). This demonstrates a complete, tested
pipeline ready to be pointed at real data the moment it exists."

## 1. Data foundation (20s)
```bash
python -m src.ingestion.fleet_loader        # 50 districts, 0 errors
```

## 2. DATA_MODE=real refusal — the most important moment (30s)
```bash
DATA_MODE=real python scripts/04_build_training_dataset.py --start 2026-06-25 --end 2026-07-05
# -> RealDataUnavailableError: no real production source available. Refuses
#    to run rather than fabricate a result.
```

## 3. Build the dataset + baselines (30s)
```bash
DATA_MODE=demo python scripts/02_build_weather_dataset.py --start 2026-05-20 --end 2026-07-05
DATA_MODE=demo python scripts/04_build_training_dataset.py --start 2026-05-20 --end 2026-07-05
DATA_MODE=demo python scripts/05_train_and_backtest.py
```
Show `reports/MODEL_VALIDATION_REPORT.md`: 4-model comparison, P10-P90
coverage, exact hierarchical reconciliation, raw-vs-corrected forecast.

## 4. Multi-horizon (30s)
```bash
DATA_MODE=demo python scripts/06_multi_horizon_backtest.py
```
`reports/MULTI_HORIZON_REPORT.md`: intra-day, J+1, J+2, J+3 each actually
retrained and backtested separately — point out MAE roughly doubling from
intra-day to J+1, and coverage honestly degrading at longer horizons.

## 5. Continuous learning (20s)
```bash
DATA_MODE=demo python scripts/07_update_forecast_model.py --trigger cli
```
Watch the 7 steps execute live: ingest -> validate -> error -> correct ->
recalibrate -> retrain-decision -> versioned save under `models/registry/`.

## 6. Explainability + monitoring (20s)
Show `reports/explainability/EXPLAINABILITY_SUMMARY.md` (SHAP: radiation and
capacity dominate, as physically expected).
```python
from src.monitoring.drift import weather_drift_report
```
or hit `GET /drift/weather`, `/anomalies` on the API.

## 7. API (20s)
```bash
uvicorn src.api.main:app --port 8000
```
`/health`, `/forecast/national`, `/forecast/district/D01`, `/uncertainty`,
`/export?fmt=csv` — every response tagged `DEMO_PROXY_PHYSICS_SIMULATION`.

## 8. Dashboard + map (40s)
```bash
streamlit run src/dashboard/app.py
```
Walk through: National overview -> Governorate -> District -> Uncertainty
band -> Monitoring/anomalies -> Tunisia map (point out the "representative
points, not official polygons" label).

## 9. Tests (15s)
```bash
pytest tests/ -q
```
49/49 passing.

## 10. Close (15s)
`FINAL_CDC_COMPLIANCE_MATRIX.md`: 32 PASS / 1 PARTIAL / 2 BLOCKED — the 2
BLOCKED items are named external dependencies (real weather at scale, real
STEG production), not gaps in the code.
