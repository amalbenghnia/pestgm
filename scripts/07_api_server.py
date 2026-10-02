import os
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="☀ Shamsna PV Forecast API - STEG")

# Allow Streamlit to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. DATABASE LOADER (Real 2020+2026 Fleet Data)
# ==========================================
DATA_PATH = "data/processed/training_dataset_15min.csv"
print(f"Loading complete dataset from {DATA_PATH}...")
try:
    df_all = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    # We will use October as our "Live Demo" month
    mask = (df_all["timestamp"].dt.month == 10)
    df = df_all[mask].copy()
    
    # Pre-compute realistic forecasts for the demo
    np.random.seed(42)
    # P50 = Reference + error
    df["forecast_p50"] = df["pv_production_mw_reference"] * np.random.normal(1.0, 0.08, len(df))
    df["forecast_p50"] = df["forecast_p50"].clip(lower=0)
    df.loc[df["daylight_flag"] == 0, "forecast_p50"] = 0
    
    # P10 / P90
    margin = df["capacity_mw"] * 0.08
    df["forecast_p10"] = (df["forecast_p50"] - margin).clip(lower=0)
    df["forecast_p90"] = (df["forecast_p50"] + margin).clip(upper=df["capacity_mw"])
    df.loc[df["daylight_flag"] == 0, "forecast_p10"] = 0
    df.loc[df["daylight_flag"] == 0, "forecast_p90"] = 0
    
    # Actuals
    df["actual_mw"] = df["pv_production_mw_reference"]
    print("Database loaded successfully.")
except Exception as e:
    print(f"Failed to load data: {e}")
    df = pd.DataFrame()

# ==========================================
# 2. API ENDPOINTS
# ==========================================

@app.get("/health")
def health_check():
    return {"status": "ok", "rows_loaded": len(df)}

@app.get("/hierarchy")
def get_hierarchy():
    """Returns the list of Governorates and their Districts"""
    if df.empty: return {}
    hierarchy = {}
    for gov in sorted(df["governorate"].dropna().unique()):
        districts = sorted(df[df["governorate"] == gov]["district"].unique())
        hierarchy[gov] = districts
    return hierarchy

@app.get("/forecast")
def get_forecast(level: str, name: str, date_str: str, current_time: str):
    """Get forecast timeseries for a specific level and date."""
    if df.empty: return {"error": "No data"}
    
    target_date = pd.to_datetime(date_str).date()
    # Get the requested date + next 3 days (for J+1, J+2, J+3)
    end_date = target_date + pd.Timedelta(days=3)
    
    mask = (df["timestamp"].dt.date >= target_date) & (df["timestamp"].dt.date <= end_date)
    sub_df = df[mask].copy()
    
    if level == "Governorate" and name != "All":
        sub_df = sub_df[sub_df["governorate"] == name]
    elif level == "District" and name != "All":
        sub_df = sub_df[sub_df["district"] == name]
        
    # Aggregate
    agg_df = sub_df.groupby("timestamp").agg({
        "actual_mw": "sum",
        "forecast_p10": "sum",
        "forecast_p50": "sum",
        "forecast_p90": "sum",
        "capacity_mw": "sum"
    }).reset_index()
    
    # Hide actuals after "current_time"
    simulated_now = pd.to_datetime(f"{date_str} {current_time}")
    agg_df.loc[agg_df["timestamp"] > simulated_now, "actual_mw"] = np.nan
    
    return {
        "metadata": {"level": level, "name": name, "capacity_mw": float(agg_df["capacity_mw"].max() if not agg_df.empty else 0)},
        "timeseries": agg_df.to_dict(orient="records")
    }

@app.get("/map")
def get_map_data(date_str: str, time_str: str):
    """Returns district-level data for the Geographic View at a specific time."""
    if df.empty: return []
    
    target_dt = pd.to_datetime(f"{date_str} {time_str}")
    sub_df = df[df["timestamp"] == target_dt].copy()
    
    map_data = []
    for _, row in sub_df.iterrows():
        uncertainty = row["forecast_p90"] - row["forecast_p10"]
        capacity = row["capacity_mw"]
        relative_unc = (uncertainty / capacity) * 100 if capacity > 0 else 0
        
        # Determine alert color
        if relative_unc > 15:
            color, status = "#ff4b4b", "High Uncertainty" # Red
        elif relative_unc > 8:
            color, status = "#ffa500", "Attention" # Orange
        else:
            color, status = "#00d2ff", "Normal" # Blue
            
        map_data.append({
            "district": row["district"],
            "governorate": row["governorate"],
            "lat": row["latitude"],
            "lon": row["longitude"],
            "capacity_mw": capacity,
            "forecast_mw": row["forecast_p50"],
            "actual_mw": row["actual_mw"],
            "status": status,
            "color": color,
            "uncertainty_mw": uncertainty
        })
    return map_data

@app.get("/alerts")
def get_alerts(date_str: str, time_str: str):
    """Generate dynamic alerts based on the map data."""
    map_data = get_map_data(date_str, time_str)
    alerts = []
    for d in map_data:
        if d["status"] == "High Uncertainty":
            alerts.append({
                "level": "🔴 High", 
                "title": f"High uncertainty in {d['district']}", 
                "desc": f"Spread: ±{d['uncertainty_mw']/2:.1f} MW"
            })
    # Add a global alert if needed, or if empty add a normal status
    if not alerts:
        alerts.append({"level": "🟢 Normal", "title": "Grid Stable", "desc": "No significant deviations detected."})
    return alerts[:4] # Return top 4

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
