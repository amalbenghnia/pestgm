"""
Streamlit dashboard (CDC continuation Phase 16/17). 10 pages + interactive
Tunisia map. Every page shows a REAL DATA / DEMO-SYNTHETIC DATA banner driven
by DATA_MODE and the dataset's own provenance columns - never silently
presented as validated real-world output.

Run:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aggregation.hierarchical import reconcile_all_levels, check_consistency
from src.monitoring.drift import weather_drift_report, production_drift_report, anomaly_flags
from src.forecasting.backtesting import evaluate
from src.ingestion.data_mode import get_data_mode, DataMode

st.set_page_config(page_title="PESTGM Track 1 - PV Forecast (Tunisia)", layout="wide")

DATA_PATH = "data/processed/training_dataset_15min_sample.csv"
TARGET_COL = "pv_production_mw_proxy"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def provenance_banner(df: pd.DataFrame):
    mode = get_data_mode()
    weather_src = df["weather_source"].iloc[0] if "weather_source" in df.columns else "UNKNOWN"
    prod_src = df["pv_production_source"].iloc[0] if "pv_production_source" in df.columns else "UNKNOWN"
    is_real = (mode == DataMode.REAL)
    if is_real:
        st.success("REAL DATA — DATA_MODE=real")
    else:
        st.warning(
            f"DEMO / SYNTHETIC DATA — DATA_MODE=demo | weather_source={weather_src} | "
            f"production_source={prod_src}. Not validated against real STEG production. "
            f"See docs/REAL_PRODUCTION_DATA_SOURCES.md."
        )


df = load_data()

st.sidebar.title("PESTGM 7.0 — Track 1")
page = st.sidebar.radio("Page", [
    "1. National overview", "2. Governorate overview", "3. District forecast",
    "4. Uncertainty", "5. Forecast vs observed", "6. Forecast errors",
    "7. Weather", "8. Monitoring / anomalies", "9. Model / explainability",
    "10. Data provenance / status", "Tunisia map",
])

provenance_banner(df)

# ---------------------------------------------------------------- Page 1 --
if page == "1. National overview":
    st.header("National overview")
    levels = reconcile_all_levels(df, TARGET_COL)
    national = levels["national"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total installed capacity (MW)", f"{df.drop_duplicates('district_id')['capacity_mw'].sum():.1f}")
    col2.metric("Peak national forecast (MW, DEMO)", f"{national[TARGET_COL].max():.1f}")
    col3.metric("Districts", df["district_id"].nunique())
    fig = px.line(national, x="timestamp", y=TARGET_COL, title="National PV production (DEMO_PROXY)")
    st.plotly_chart(fig, use_container_width=True)
    consistency = check_consistency(levels, TARGET_COL)
    st.caption(f"Hierarchical consistency: {consistency}")

# ---------------------------------------------------------------- Page 2 --
elif page == "2. Governorate overview":
    st.header("Governorate overview")
    levels = reconcile_all_levels(df, TARGET_COL)
    gov = st.selectbox("Governorate", sorted(df["governorate"].unique()))
    g = levels["governorate"][levels["governorate"]["governorate"] == gov]
    fig = px.line(g, x="timestamp", y=TARGET_COL, title=f"{gov} — aggregated PV production (DEMO_PROXY)")
    st.plotly_chart(fig, use_container_width=True)
    districts_here = df[df["governorate"] == gov].drop_duplicates("district_id")
    st.dataframe(districts_here[["district_id", "district", "capacity_mw"]].sort_values("capacity_mw", ascending=False))

# ---------------------------------------------------------------- Page 3 --
elif page == "3. District forecast":
    st.header("District forecast")
    district = st.selectbox("District", sorted(df["district"].unique()))
    g = df[df["district"] == district].sort_values("timestamp")
    fig = px.line(g, x="timestamp", y=TARGET_COL, title=f"{district} — PV production (DEMO_PROXY)")
    fig.add_hline(y=g["capacity_mw"].iloc[-1], line_dash="dot", annotation_text="installed capacity")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(g[["timestamp", TARGET_COL, "capacity_mw", "shortwave_radiation", "temperature_2m"]].tail(20))

# ---------------------------------------------------------------- Page 4 --
elif page == "4. Uncertainty":
    st.header("Uncertainty (P10-P50-P90)")
    st.caption("Illustrative rolling-std band shown here for interactivity; "
               "the fitted quantile-LightGBM + conformal results are in "
               "reports/MODEL_VALIDATION_REPORT.md and reports/MULTI_HORIZON_REPORT.md.")
    district = st.selectbox("District", sorted(df["district"].unique()), key="unc_district")
    g = df[df["district"] == district].sort_values("timestamp").tail(200).copy()
    std = g[TARGET_COL].rolling(8, min_periods=1).std().fillna(0)
    g["p10"] = (g[TARGET_COL] - 1.2816 * std).clip(lower=0)
    g["p90"] = (g[TARGET_COL] + 1.2816 * std).clip(upper=g["capacity_mw"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g["p90"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g["p10"], fill="tonexty", line=dict(width=0), name="P10-P90 band"))
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g[TARGET_COL], name="P50 / forecast", line=dict(color="black")))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Page 5 --
elif page == "5. Forecast vs observed":
    st.header("Forecast vs observed (DEMO_PROXY target stands in for 'observed')")
    st.caption("No real observed production exists yet (see docs/REAL_PRODUCTION_DATA_SOURCES.md). "
               "This page compares the persistence baseline forecast against the DEMO_PROXY target "
               "to demonstrate the intended real-vs-forecast comparison view.")
    district = st.selectbox("District", sorted(df["district"].unique()), key="fvo_district")
    g = df[df["district"] == district].sort_values("timestamp").copy()
    g["persistence_forecast"] = g[TARGET_COL].shift(1)
    g = g.tail(200)
    fig = px.line(g, x="timestamp", y=[TARGET_COL, "persistence_forecast"],
                  labels={"value": "MW"}, title=f"{district}: DEMO_PROXY 'observed' vs persistence forecast")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Page 6 --
elif page == "6. Forecast errors":
    st.header("Forecast errors (persistence baseline, DEMO_ONLY)")
    district = st.selectbox("District", sorted(df["district"].unique()), key="err_district")
    g = df[df["district"] == district].sort_values("timestamp").copy()
    g["persistence_forecast"] = g[TARGET_COL].shift(1)
    g = g.dropna(subset=["persistence_forecast"])
    m = evaluate(g[TARGET_COL], g["persistence_forecast"], g["capacity_mw"])
    st.json(m)
    g["error"] = g["persistence_forecast"] - g[TARGET_COL]
    fig = px.histogram(g, x="error", title="Error distribution (MW)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Page 7 --
elif page == "7. Weather":
    st.header("Weather")
    district = st.selectbox("District", sorted(df["district"].unique()), key="wx_district")
    g = df[df["district"] == district].sort_values("timestamp").tail(200)
    fig = px.line(g, x="timestamp", y=["shortwave_radiation", "temperature_2m", "cloud_cover"],
                  title=f"{district} — weather variables ({g['weather_source'].iloc[0]})")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Page 8 --
elif page == "8. Monitoring / anomalies":
    st.header("Monitoring / anomalies")
    ref_end = df["timestamp"].quantile(0.5)
    st.subheader("Weather drift (PSI)")
    st.dataframe(weather_drift_report(df, ref_end))
    st.subheader("Production drift (PSI)")
    st.json(production_drift_report(df, TARGET_COL, ref_end))
    st.subheader("Physical anomaly flags")
    flags = anomaly_flags(df, production_col=TARGET_COL)
    pct = 100 * flags["physical_anomaly_flag"].mean()
    st.metric("% rows flagged anomalous", f"{pct:.2f}%")
    st.dataframe(flags[flags["physical_anomaly_flag"]].head(50))

# ---------------------------------------------------------------- Page 9 --
elif page == "9. Model / explainability":
    st.header("Model / explainability")
    exp_path = Path("reports/explainability/global_feature_importance.csv")
    if exp_path.exists():
        gfi = pd.read_csv(exp_path)
        fig = px.bar(gfi.head(10), x="mean_abs_shap", y="feature", orientation="h",
                     title="Global feature importance (SHAP, LightGBM)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python -c \"...\"` (see README) or the explainability module to generate "
                "reports/explainability/global_feature_importance.csv first.")
    st.caption("Model trained on DEMO_PROXY_PHYSICS_SIMULATION - see reports/MODEL_VALIDATION_REPORT.md")

# --------------------------------------------------------------- Page 10 --
elif page == "10. Data provenance / status":
    st.header("Data provenance / status")
    st.write(f"DATA_MODE: **{get_data_mode().value}**")
    prov = df[["weather_source", "pv_production_source", "capacity_source"]].drop_duplicates()
    st.dataframe(prov)
    st.markdown("See `docs/DATA_READINESS_REPORT.md` and `docs/REAL_PRODUCTION_DATA_SOURCES.md` "
                "for the full real/derived/model-output/demo-only classification.")

# ------------------------------------------------------------------ Map --
elif page == "Tunisia map":
    st.header("Tunisia — district map (representative points, not official polygons)")
    metric = st.selectbox("Show", ["capacity_mw", TARGET_COL])
    latest = df.sort_values("timestamp").groupby("district_id").tail(1)
    fig = px.scatter_map(
        latest, lat="latitude", lon="longitude", size=metric, color=metric,
        hover_name="district", hover_data=["governorate", "capacity_mw", TARGET_COL],
        zoom=5.3, height=650, map_style="open-street-map",
        title="Representative district locations (not official administrative boundaries)",
    )
    st.plotly_chart(fig, use_container_width=True)
