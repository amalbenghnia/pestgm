"""
Builds the final SYNCHRONIZED, ML-ready dataset:

    timestamp | district_id | district | governorate | capacity_mw |
    weather features | solar geometry features | pv_production_mw_proxy | ...

at the configured resolution (default 15 minutes), by:
  1. building the capacity time series from the 3 Prosol snapshots,
  2. loading the hourly weather dataset (built by 02_build_weather_dataset.py)
     and upsampling it to 15-min via linear interpolation (explicitly flagged:
     this is a RESOLUTION change, not new INFORMATION - CDC section 9),
  3. computing solar geometry at the target resolution,
  4. running the physics-based proxy production generator (CDC section 6/7/35).

Also writes a minimal "simple view" CSV matching the exact format requested
for downstream model training:  timestamp, district, pv_production_mw

Usage (run from repo root):
    python scripts/04_build_training_dataset.py                     # dynamic: last 45 days up to today
    python scripts/04_build_training_dataset.py --start 2026-06-01 --end 2026-08-01   # explicit override
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.fleet_loader import load_fleet_csv
from src.ingestion.capacity_timeseries import build_capacity_timeseries
from src.features.solar_geometry import add_solar_geometry
from src.physics.pv_model import generate_proxy_production
from src.ingestion.data_mode import get_data_mode, DataMode, RealDataUnavailableError


def upsample_weather_to_resolution(weather: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Linear interpolation of hourly weather to a finer resolution, per district.
    Flags the result as resampled (not genuine higher-frequency information).
    Also propagates archived forecast weather columns (previous_dayX) unchanged."""
    out = []
    interp_cols = [
        "temperature_2m", "relative_humidity_2m", "cloud_cover",
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "direct_normal_irradiance", "wind_speed_10m", "precipitation",
    ]
    # Detect and pass through archived forecast columns (no interpolation - hourly is fine)
    forecast_cols = [c for c in weather.columns if "_previous_day" in c]
    
    for did, g in weather.groupby("district_id"):
        g = g.set_index("timestamp").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        gi = g[interp_cols].reindex(g.index.union(idx)).interpolate("time").reindex(idx)
        meta_cols = ["district", "governorate", "requested_latitude", "requested_longitude",
                     "weather_grid_latitude", "weather_grid_longitude", "weather_source"]
        for c in meta_cols:
            gi[c] = g[c].iloc[0]
        gi["district_id"] = did
        gi["weather_resolution_note"] = "resampled_from_hourly_interpolation"
        # Forward-fill archived forecast columns (they are already hourly, no interpolation needed)
        for fc in forecast_cols:
            if fc in g.columns:
                gi[fc] = g[fc].reindex(g.index.union(idx)).ffill().reindex(idx)
        gi = gi.reset_index(names="timestamp")
        out.append(gi)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None,
                     help="Default: 45 days before --end (dynamic).")
    ap.add_argument("--end", default=None,
                     help="Default: today (dynamic, not a hardcoded literal date).")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--weather_in", default="data/processed/weather_historical.csv")
    ap.add_argument("--out_full", default="data/processed/training_dataset_15min.csv")
    ap.add_argument("--out_simple", default="data/processed/pv_production_simple_view.csv")
    args = ap.parse_args()

    user_gave_dates = args.start is not None or args.end is not None
    end_date = args.end or date.today().isoformat()
    start_date = args.start or (date.fromisoformat(end_date) - timedelta(days=45)).isoformat()
    args.start, args.end = start_date, end_date
    print(f"Date range: {args.start} -> {args.end} ({'explicit' if user_gave_dates else 'dynamic default: last 45 days up to today'})")

    cfg = yaml.safe_load(Path(args.config).read_text())
    res_min = cfg["production_target"]["resolution_minutes"]
    freq = f"{res_min}min"

    print("1/5 Loading fleet & building capacity time series...")
    fleet = load_fleet_csv(cfg["paths"]["raw_fleet_csv"])
    cap_ts = build_capacity_timeseries(fleet, freq=freq, start=args.start, end=args.end)

    print("2/5 Loading & upsampling weather...")
    weather = pd.read_csv(args.weather_in, parse_dates=["timestamp"])
    weather_hr = upsample_weather_to_resolution(weather, freq=freq)

    print("3/5 Merging capacity + weather...")
    merged = cap_ts.merge(
        weather_hr.drop(columns=["district", "governorate"]),
        on=["district_id", "timestamp"], how="inner",
    )

    print("4/5 Computing solar geometry...")
    merged = add_solar_geometry(merged, lat_col="latitude", lon_col="longitude", ts_col="timestamp")

    print("5/5 Generating production target...")
    mode = get_data_mode()
    target_mode = cfg["production_target"]["mode"]  # DEMO_PROXY | PVGIS_REFERENCE | REAL_CSV
    real_production_path = Path("data/raw/pv_production_real.csv")
    pvgis_reference_path = Path("data/processed/pv_production_reference_15min.csv")

    if target_mode == "PVGIS_REFERENCE":
        if not pvgis_reference_path.exists():
            if mode == DataMode.REAL:
                raise RealDataUnavailableError(
                    f"PVGIS physical reference ({pvgis_reference_path})",
                    detail="Run `python scripts/10_build_pv_reference_dataset.py` first.",
                )
            else:
                print("[WARN] PVGIS_REFERENCE requested but file not found; falling back to DEMO_PROXY.")
                target_mode = "DEMO_PROXY"
        else:
            ref = pd.read_csv(pvgis_reference_path, parse_dates=["timestamp"])
            keep_cols = ["district_id", "timestamp", "pv_production_mw_reference",
                         "ghi_wm2", "pvgis_source_label"]
            keep_cols = [c for c in keep_cols if c in ref.columns]
            merged = merged.merge(ref[keep_cols], on=["district_id", "timestamp"], how="left")
            merged["pv_production_source"] = "PVGIS_PHYSICAL_ESTIMATE_REAL_IRRADIANCE"
            n_missing = merged["pv_production_mw_reference"].isna().sum()
            if n_missing:
                print(f"[WARN] {n_missing:,} rows have no matching PVGIS timestamp "
                      f"(year mismatch between --start/--end and PVGIS year 2020). Left as NaN.")

    if target_mode == "REAL_CSV":
        if not real_production_path.exists():
            raise RealDataUnavailableError(
                "real PV production (data/raw/pv_production_real.csv)",
                detail="No real production file found. See docs/REAL_PRODUCTION_DATA_SOURCES.md.",
            )
        real_prod = pd.read_csv(real_production_path, parse_dates=["timestamp"])
        merged = merged.merge(real_prod, on=["district_id", "timestamp"], how="left")
        merged = merged.rename(columns={"production_mw": "pv_production_mw_real"})
        merged["pv_production_source"] = "REAL_MEASURED"

    if target_mode == "DEMO_PROXY":
        if mode == DataMode.REAL:
            raise ValueError(
                f"DATA_MODE=real requires production_target.mode to be PVGIS_REFERENCE or REAL_CSV "
                f"in configs/config.yaml, got {target_mode!r}."
            )
        pt_cfg = cfg["production_target"]
        merged = generate_proxy_production(
            merged,
            performance_ratio_mean=pt_cfg["performance_ratio_mean"],
            performance_ratio_std=pt_cfg["performance_ratio_std"],
            noise_std_fraction=pt_cfg["noise_std_fraction"],
            noct_c=pt_cfg["nominal_operating_cell_temp_c"],
            temp_coeff_pct_per_c=pt_cfg["temperature_coefficient_pct_per_c"],
            random_seed=cfg["random_seed"],
        )
        assert (merged["pv_production_mw_proxy"] <= merged["capacity_mw"] + 1e-6).all(), "Production exceeds capacity!"
        assert (merged["pv_production_mw_proxy"] >= -1e-9).all(), "Negative production!"
        night_mask = merged["daylight_flag"] == 0
        assert merged.loc[night_mask, "pv_production_mw_proxy"].max() < 1e-6, "Non-zero night production!"


    Path(args.out_full).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_full, index=False)
    print(f"Full synchronized dataset: {len(merged):,} rows, {merged.shape[1]} columns -> {args.out_full}")

    if target_mode == "PVGIS_REFERENCE" and "pv_production_mw_reference" in merged.columns:
        prod_col = "pv_production_mw_reference"
    elif target_mode == "REAL_CSV" and "pv_production_mw_real" in merged.columns:
        prod_col = "pv_production_mw_real"
    else:
        prod_col = "pv_production_mw_proxy"
    simple = merged[["timestamp", "district", prod_col]].dropna(subset=[prod_col]).rename(
        columns={prod_col: "pv_production_mw"}
    ).sort_values(["district", "timestamp"])
    simple.to_csv(args.out_simple, index=False)
    print(f"Simple view: {len(simple):,} rows -> {args.out_simple}")
    print("\nSample (SFAX NORD, around noon):")
    sample = simple[(simple["district"].str.contains("SFAX", case=False))]
    print(sample.head(8).to_string(index=False))



if __name__ == "__main__":
    main()
