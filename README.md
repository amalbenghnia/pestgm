<div align="center">
  <img src="dashboard/logo.jpg" alt="Shams'na Logo" width="180"/>

# 🇹🇳 Shams'na — National Intelligent Platform for Rooftop PV Forecasting

**AI-powered photovoltaic forecasting platform for Tunisia's national electricity grid**

> **PESTGM 7.0 — CDC Tech Challenge | STEG**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost--CF-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-24%2F24%20passing-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-1.75M%20rows%20%7C%202020--2023-blueviolet)
![ENSTAB](https://img.shields.io/badge/Validation-ENSTAB%20Real%20Sensor-success)
![Coverage](https://img.shields.io/badge/nMAE%20Intraday-3.39%25-green)
![J+3](https://img.shields.io/badge/nMAE%20J%2B3-4.07%25-yellowgreen)

</div>

---

## 📖 Overview

The rapid deployment of decentralized rooftop photovoltaic systems across Tunisia introduces a growing forecasting challenge for the national electricity grid operator (**STEG**).

**Shams'na** is a national-scale intelligent forecasting platform designed to estimate the production of Tunisia's **144,979 rooftop PV installations** (422 MW installed) across **50 districts** and **24 governorates**, from **intraday** to **J+3** horizons.

The system combines:
- 🛰️ PVGIS SARAH-3 physical reference data (2020–2023)
- 🌦️ ERA5 reanalysis + Open-Meteo archived weather forecasts
- 🤖 XGBoost Capacity Factor (CF) machine learning
- 📐 Conformal Prediction uncertainty intervals
- 🧩 Bottom-up hierarchical aggregation
- ✅ External zero-shot validation on the ENSTAB/Borj Cedria real sensor

---

## 🎯 Forecasting Objectives

| Spatial Level | Output |
|---|---|
| District (50) | PV production forecast (MW) |
| Governorate (24) | Aggregated PV production forecast (MW) |
| National | Total Tunisian rooftop PV production (MW) |

| Horizon | Use Case |
|---|---|
| Intraday | Real-time operational dispatching |
| J+1 | Day-ahead planning |
| J+2 | Short-term reserve management |
| J+3 | Extended operational scheduling |

**Resolution:** 15-minute timesteps.

---

## 🏗️ System Architecture

```
┌────────────────────────────────┐    ┌───────────────────────────────┐
│   PV Fleet — PROSOL / STEG     │    │   PVGIS SARAH-3 API (JRC)     │
│   144,979 installations        │    │   Physical PV reference        │
│   422 MW / 50 districts        │    │   2020 → 2023                 │
└──────────────┬─────────────────┘    └────────────────┬──────────────┘
               │                                       │
               ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA ENGINEERING LAYER                        │
│   Fleet capacity interpolation · Weather merge · Solar geometry      │
│   Lag features · Cyclic encoding · 15-min synchronization           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┴──────────────────────┐
          ▼                                           ▼
┌──────────────────────┐                  ┌───────────────────────────┐
│  Historical Weather   │                  │  Historical Forecast API  │
│  Open-Meteo / ERA5    │                  │  Open-Meteo archived      │
│  (Training: ERA5)     │                  │  (J+1/J+2/J+3 test set)  │
└──────────────────────┘                  └───────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FORECASTING ENGINE                            │
│                                                                     │
│   Persistence → Seasonal → Physics → LightGBM → XGBoost-CF ←best  │
│                                                                     │
│   Target: Capacity Factor = P(t) / C(t)                            │
│   Output: P̂(t) = CF̂(t) × C(t)                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
      Point Forecast P50              Uncertainty Intervals
                                   P10 │ P25 │ P50 │ P75 │ P90
                                       (Conformal Prediction)
               │                               │
               └───────────────┬───────────────┘
                               ▼
                    Physical Post-Processing
                    · Force P=0 at night
                    · Clip P ∈ [0, Capacity]
                    · Ramp / jump checks
                               │
                               ▼
                  Hierarchical Aggregation
               District → Governorate → National
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Shams'na Dashboard   │
                    │ + FastAPI Backend    │
                    └──────────────────────┘
```

---

## 📊 Datasets & Sources

### 1. 🇹🇳 Tunisian PV Fleet — PROSOL / STEG

The foundation of the platform is the official **PROSOL** fleet database, containing information on rooftop PV installations across Tunisia:

- Installed capacity per district (MW)
- Number of installations per district
- Geographic coordinates (latitude, longitude)
- Fleet evolution across time (3 official snapshots)

As of March 2026: **144,979 installations | 422 MW installed | 50 districts covered**.

> 📁 `data/raw/pv_fleet_prosol_3_snapshots.csv`
> 🔗 [STEG — PROSOL Programme](https://www.steg.com.tn/)

---

### 2. ☀️ PVGIS SARAH-3 — Physical PV Reference Dataset

[**PVGIS**](https://re.jrc.ec.europa.eu/pvgis.html) (Photovoltaic Geographical Information System) developed by the **European Commission Joint Research Centre (JRC)** is used to build a multi-year physical PV production reference dataset.

**PVGIS SARAH-3** provides satellite-derived solar irradiance and PV production estimates at hourly resolution across Europe, Africa, and the Middle East.

| Parameter | Value |
|---|---|
| Database | PVGIS SARAH-3 |
| Coverage | 2020 · 2021 · 2022 · 2023 |
| Spatial coverage | All 50 Tunisian districts |
| Output | Hourly PV power (W), irradiance, temperature |
| System | Fixed tilt, optimal angle, performance ratio |

This dataset serves as the **training target** (Capacity Factor) for the XGBoost-CF model.

> 📁 `data/external/pvgis_cache/D{01..50}_2020_2023.json`
> 🔗 [PVGIS API — European Commission JRC](https://re.jrc.ec.europa.eu/pvgis.html)
> 🔗 [PVGIS API Documentation](https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system/getting-started-pvgis/api-non-interactive-service_en)

---

### 3. 🌦️ Historical Weather — Open-Meteo / ERA5

Meteorological variables for model training are obtained from the [**Open-Meteo Historical Weather API**](https://open-meteo.com/en/docs/historical-weather-api), powered by **ERA5 reanalysis** from ECMWF.

| Variable | Source |
|---|---|
| Global Horizontal Irradiance | ERA5 |
| Temperature (2m) | ERA5 |
| Cloud cover | ERA5 |
| Relative humidity | ERA5 |
| Wind speed | ERA5 |
| Precipitation | ERA5 |

ERA5 provides hourly meteorological reanalysis data from **1940 to the present** at global coverage.

> 🔗 [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
> 🔗 [ECMWF ERA5 Dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)

---

### 4. 🔮 Archived Weather Forecasts — Open-Meteo Historical Forecast API

A critical requirement for operational forecasting evaluation is eliminating **look-ahead bias**: the model must only access information that would have been available at the forecasting time.

For J+1/J+2/J+3 evaluation, the project uses the [**Open-Meteo Historical Forecast API**](https://open-meteo.com/en/docs/historical-forecast-api), which archives weather forecasts issued at past dates — reproducing the exact information an operational system would have had.

| Feature | Detail |
|---|---|
| Coverage | ~2022 → present |
| Lead times | J+1, J+2, J+3, ..., J+7 |
| Purpose | No-look-ahead-bias test evaluation |
| Application | J+1/J+2/J+3 test set (2023) |

This ensures the test set evaluation is **genuinely operational** — not inflated by using perfect reanalysis weather for future horizons.

> 🔗 [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
> 🔗 [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)

---

### 5. 🧪 Real PV Production — ENSTAB / Borj Cedria (External Validation)

A physically measured PV production dataset from the **ENSTAB (École Nationale des Sciences et Technologies Avancées de Bordj Cédria)** solar installation is used as an **independent external validation dataset**.

| Parameter | Value |
|---|---|
| Location | Borj Cedria, Tunisia |
| Installed capacity | 2.4 kWp |
| Coverage | 2022 · 2023 · 2024 |
| Type | Real measured PV production (physical sensor) |
| Role | External zero-shot validation |

The model was **never trained on this dataset**. This validates the model's ability to generalize across installations of completely different sizes (2.4 kW vs 422 MW) — a **zero-shot generalization** test enabled by the Capacity Factor architecture.

> 📁 `data/validation/enstab_borj_cedria_real.csv`
> 📄 `reports/ENSTAB_EXTERNAL_VALIDATION_REPORT.md`

---

## 📐 Capacity Factor (CF) Architecture

The central modelling innovation is forecasting **Capacity Factor** rather than directly predicting MW:

$$CF_t = \frac{P_t}{C_t}$$

The final production forecast is then:

$$\hat{P}_t = \widehat{CF}_t \times C_t$$

**Why this matters:**
- The CF is bounded in [0, 1] regardless of fleet size
- The model learns a **normalized production profile** that generalizes across districts
- The same model can forecast a 2.4 kW installation (ENSTAB) and 422 MW (national fleet) without retraining
- Aggregation is exact: $CF_{district} = \frac{\sum_i P_i}{\sum_i C_i}$

---

## 🤖 Forecasting Models

| Model | Architecture | Target |
|---|---|---|
| Persistence | Previous timestep | MW |
| Seasonal Persistence | Same hour, previous day | MW |
| Physics Baseline | Clear-sky × PR × temp derating | MW |
| LightGBM-MW | Gradient boosting, direct MW | MW |
| LightGBM-CF | Gradient boosting + CF normalization | CF → MW |
| XGBoost-MW | XGBoost, direct MW | MW |
| **XGBoost-CF** ⭐ | **XGBoost + CF normalization** | **CF → MW** |

All ML models are evaluated using **chronological backtesting** (no random splits) to preserve the temporal forecasting structure.

---

## 📈 Results — Test Set 2023 (XGBoost-CF)

**Temporal split:**
- Train: 2020 + 2021 (877,200 rows)
- Validation: 2022 (438,000 rows)
- **Test: 2023** (436,827 rows — genuinely unseen)

**Weather for test J+1/J+2/J+3:** Real archived forecast weather (no look-ahead bias).

### Point-Forecast Metrics

| Horizon | n (samples) | MAE (MW) | RMSE (MW) | nMAE (% cap) | WAPE (%) | Bias (MW) |
|---|---|---|---|---|---|---|
| Intraday | 436,777 | 0.237 | 0.674 | **3.39%** | 16.7% | 0.011 |
| J+1 | 233,307 | 0.316 | 0.770 | **3.99%** | 22.0% | 0.026 |
| J+2 | 230,715 | 0.319 | 0.778 | **4.05%** | 22.1% | 0.030 |
| J+3 | 228,123 | 0.322 | 0.775 | **4.07%** | 22.2% | 0.031 |

> ✅ **Degradation J+1→J+3: only +0.68% nMAE** — the model relies on deterministic solar geometry and is highly stable across horizons.

### Conformal Prediction Coverage (target: 80%)

| Horizon | Coverage (overall) | Coverage (daytime) | Status |
|---|---|---|---|
| Intraday | 91.5% | 86.5% | ✅ |
| J+1 | 89.2% | 84.1% | ✅ |
| J+2 | 88.7% | 83.5% | ✅ |
| J+3 | 88.1% | 82.9% | ✅ |

### Physical Post-Processing Effect

| Horizon | Night anomalies (raw) | Night anomalies (after PP) |
|---|---|---|
| Intraday | 35.2% | **0.016%** |
| J+1 | 31.2% | **0%** |
| J+2 | 31.0% | **0%** |
| J+3 | 29.0% | **0%** |

---

## 🧪 External Validation — ENSTAB Borj Cedria

The XGBoost-CF model was applied to the ENSTAB installation without any retraining.

| Horizon | nMAE (% cap) | MAE (kW) | Bias | Result |
|---|---|---|---|---|
| Intraday | 11.7% | 0.28 kW | ~0 | ✅ OK |
| J+1 | 9.8% | 0.24 kW | ~0 | ✅ OK |
| J+2 | 9.9% | 0.24 kW | ~0 | ✅ OK |
| J+3 | 9.9% | 0.24 kW | ~0 | ✅ OK |

This validates the **zero-shot generalization** capability of the Capacity Factor architecture: the model trained on 422 MW of national fleet data transfers directly to a 2.4 kW individual installation.

---

## 🖥️ Shams'na Dashboard

The operational dashboard is built with **Streamlit** and provides 6 interactive pages:

| Page | Content |
|---|---|
| 🏠 Overview | 6 live KPIs · Forecast chart · Real-time alerts |
| 🗺️ Geographic View | Interactive Tunisia map · Color-coded district status |
| 📊 Forecast Explorer | Drill-down by district/governorate · J+1–J+3 charts · CSV export |
| ⚠️ Uncertainty & Alerts | Conformal coverage · P10/P50/P90 intervals · District alerts |
| 🔋 PV Fleet | PROSOL fleet snapshot · ENSTAB external validation |
| 📈 Model Performance | Full test 2023 metrics · Physical constraint results |

**Launch the dashboard:**
```bash
.\run_dashboard.bat
# Opens at http://localhost:8501
```

---

## ⚙️ Feature Engineering

| Category | Features |
|---|---|
| Fleet | Installed capacity, district, governorate |
| Weather | GHI, temperature, cloud cover, humidity, wind speed |
| Solar geometry | Solar elevation, azimuth, clear-sky index, day/night flag |
| Temporal | Hour, day of year, month, day of week (cyclic encoding) |
| Lag features | Previous production lags, rolling means (intraday only) |

---

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/amalbenghnia/pestgm.git
cd pestgm
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 2. Launch the dashboard directly
.\run_dashboard.bat
# Opens at http://localhost:8501

# 3. Rebuild the reference dataset (2020-2023, requires internet)
python scripts/10_build_pv_reference_dataset.py --start_year 2020 --end_year 2023

# 4. Run the multi-horizon backtest
python scripts/06_multi_horizon_backtest.py

# 5. Run the ENSTAB external validation
python scripts/09_enstab_external_validation.py

# 6. Run all tests (24/24 passing)
pytest tests/ -v
```

> **Note:** The full training dataset (1.75M rows) is excluded from git (`.gitignore`). The `data/processed/*_sample.csv` files are included for quick exploration. Regenerate the full dataset with the scripts above.

---

## 📁 Repository Structure

```
pestgm-pv-forecast/
├── dashboard/
│   └── app.py                    # Shams'na Streamlit dashboard (self-contained)
├── data/
│   ├── raw/                      # PROSOL fleet snapshots
│   ├── processed/                # *_sample.csv shipped; full CSV gitignored
│   ├── external/pvgis_cache/     # PVGIS SARAH-3 per-district JSON cache
│   └── validation/               # ENSTAB real measured data (2022-2024)
├── scripts/
│   ├── 02_build_weather_dataset.py
│   ├── 04_build_training_dataset.py
│   ├── 06_multi_horizon_backtest.py
│   ├── 08_xgboost_tuning.py
│   ├── 09_enstab_external_validation.py
│   ├── 10_build_pv_reference_dataset.py
│   └── 12_model_comparison.py
├── src/
│   ├── ingestion/                # Fleet, PVGIS, weather, ENSTAB loaders
│   ├── features/                 # Solar geometry, lag features
│   ├── models/                   # XGBoost-CF, LightGBM, baselines
│   ├── forecasting/              # Backtesting, multi-horizon
│   ├── uncertainty/              # Conformal prediction
│   ├── aggregation/              # Bottom-up hierarchical
│   ├── validation/               # Physical constraint checks
│   └── api/                      # FastAPI backend
├── reports/
│   ├── PESTGM_TECHNICAL_REPORT.md
│   ├── ENSTAB_EXTERNAL_VALIDATION_REPORT.md
│   └── MODEL_COMPARISON_REPORT.md
├── docs/
│   └── DATA_ARCHITECTURE.md
├── tests/                        # 24 pytest tests
└── run_dashboard.bat
```

---

## 🏆 Project Highlights

| Dimension | Achievement |
|---|---|
| 📏 Scale | 50 districts · 24 governorates · 422 MW · 1.75M training rows |
| 📅 Temporal depth | 4 years (2020–2023) · Multi-year temporal validation |
| 🔬 Scientific rigor | No look-ahead bias · Real archived forecast weather for J+1/J+2/J+3 |
| 🎯 Accuracy | 3.39% nMAE intraday · 4.07% at J+3 · +0.68% degradation |
| 📊 Uncertainty | Conformal coverage 82–91% (target 80%) — all horizons |
| 🧪 Validation | Zero-shot generalization on real ENSTAB sensor (2.4 kW → 422 MW) |
| 🚀 Deployment | Production-ready Streamlit dashboard with real-time alerts |

---

## 📚 References

- [PVGIS — European Commission JRC](https://re.jrc.ec.europa.eu/pvgis.html)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
- [STEG — PROSOL Programme](https://www.steg.com.tn/)
- [ENSTAB — École Nationale des Sciences et Technologies Avancées de Bordj Cédria](https://www.enstab.rnu.tn/)
- Angelopoulos et al. (2021) — *Conformal Risk Control*
- Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*

---

<div align="center">

**Built with ❤️ for Tunisia's clean energy future**

☀️ *Shams'na — Notre Soleil*

</div>
