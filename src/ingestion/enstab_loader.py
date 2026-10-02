"""
Loader for the REAL, measured PV production dataset from ENSTAB / LaRINa lab,
Borj Cedria, Ben Arous, Tunisia (Mejdi, Kardous & Grayaa, "Experimental
Validation of PV Power Prediction with ML Models for Improved Grid
Integration", IEEE SSD 2023 - PDF provided alongside the raw CSV).

THIS IS REAL, MEASURED DATA - production_source is always
"MEASURED_REAL_ENSTAB". It is treated as an EXTERNAL VALIDATION dataset only
(CDC continuation Phase 17-21): never merged into national training data,
never used to tune the national model's hyperparameters.

System facts (from the paper, not fabricated):
  - Location: Borj Cedria, Ben Arous (36.70739881217652, 10.42642804675277)
  - 6x Alphanis-72-400 poly-crystalline modules, wired in parallel
  - Nameplate peak power: 2.4 kWp = 0.0024 MW
  - Grid-connected single-phase inverter (Senergy SE 2KTL-S1/G2)
  - Measured natively at 5-minute resolution
  - Raw file covers 2022-02-24 to 2024-05-31 (238,464 rows in the copy provided
    here - substantially longer than the paper's own Feb-Sep 2022 analysis
    window, which used the first ~59,488 rows)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

ENSTAB_LATITUDE = 36.70739881217652
ENSTAB_LONGITUDE = 10.42642804675277
ENSTAB_CAPACITY_KWP = 2.4
ENSTAB_CAPACITY_MW = ENSTAB_CAPACITY_KWP / 1000.0


def load_raw(path: str | Path = "data/validation/enstab_borj_cedria_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def data_quality_summary(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "native_resolution_minutes": 5,
        "n_missing_any_col": int(df.isna().any(axis=1).sum()),
        "n_duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "n_negative_power": int((df["Power"] < 0).sum()),
        "n_power_above_nameplate_2400W": int((df["Power"] > 2400).sum()),
        "n_nonzero_power_at_zero_irradiation": int(((df["Irradiation"] == 0) & (df["Power"] > 5)).sum()),
        "power_units": "W (measured by Chauvin Arnoux C.A 8336 grid analyzer, per source paper)",
        "irradiation_units": "W/m^2 (global horizontal, measured by Perel WC224 weather station)",
    }


def build_validation_dataset(
    raw_path: str | Path = "data/validation/enstab_borj_cedria_raw.csv",
    resample_to: str = "15min",
) -> pd.DataFrame:
    """
    Builds the labelled external-validation dataset:
      - resamples the real 5-min measurements to 15-min by averaging (a real
        aggregation of real measurements, not fabricated data - documented,
        not silently presented as native 15-min sensor readings)
      - adds real solar geometry (pvlib) for the real Borj Cedria coordinates
      - splits measured global irradiation into DNI/DHI using the Erbs decomposition
        model (Erbs et al. 1982 - an established, real physical decomposition of
        the REAL measured GHI, not an invented value)
      - adds the feature columns our national model expects, so it can be
        scored on this data without retraining
    """
    df = load_raw(raw_path)
    df = df.set_index("timestamp").resample(resample_to).mean().reset_index()
    df["resolution_note"] = f"RESAMPLED_FROM_5MIN_REAL_MEASUREMENTS_TO_{resample_to.upper()}"

    times = pd.DatetimeIndex(df["timestamp"]).tz_localize("Africa/Tunis", ambiguous="NaT", nonexistent="NaT")
    valid = ~times.isna()
    df = df.loc[valid].reset_index(drop=True)
    times = times[valid]

    solpos = pvlib.solarposition.get_solarposition(times, ENSTAB_LATITUDE, ENSTAB_LONGITUDE, altitude=50)
    linke = pvlib.clearsky.lookup_linke_turbidity(times, ENSTAB_LATITUDE, ENSTAB_LONGITUDE)
    airmass_rel = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    pressure = pvlib.atmosphere.alt2pres(50)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pressure)
    cs = pvlib.clearsky.ineichen(solpos["apparent_zenith"], airmass_abs, linke, altitude=50)

    df["solar_zenith_deg"] = solpos["apparent_zenith"].to_numpy()
    df["solar_elevation_deg"] = solpos["apparent_elevation"].to_numpy()
    df["solar_azimuth_deg"] = solpos["azimuth"].to_numpy()
    df["daylight_flag"] = (df["solar_elevation_deg"] > 0).astype(int)
    df["clearsky_ghi_wm2"] = cs["ghi"].to_numpy()

    # Erbs decomposition: real, established model applied to REAL measured GHI
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    erbs = pvlib.irradiance.erbs(df["Irradiation"].to_numpy(), solpos["apparent_zenith"].to_numpy(),
                                  times.dayofyear.to_numpy())
    df["direct_normal_irradiance"] = np.clip(np.asarray(erbs["dni"]), 0, None)
    df["diffuse_radiation"] = np.clip(np.asarray(erbs["dhi"]), 0, None)
    df["direct_radiation"] = np.clip(
        df["Irradiation"].to_numpy() - df["diffuse_radiation"].to_numpy(), 0, None
    )

    hour_frac = times.hour + times.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    doy = times.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["day_of_week"] = times.dayofweek
    df["month"] = times.month

    df["capacity_mw"] = ENSTAB_CAPACITY_MW
    df["shortwave_radiation"] = df["Irradiation"]
    df["temperature_2m"] = df["Temperature"]
    df["cell_temperature_c"] = df["Cell_temperature"]
    df["relative_humidity_2m"] = df["Humidity"]
    df["wind_speed_10m"] = df["Wind_speed"]
    df["cloud_cover"] = np.clip(
        100 * (1 - df["Irradiation"] / df["clearsky_ghi_wm2"].replace(0, np.nan)), 0, 100
    ).fillna(0)
    df["pv_production_mw_measured"] = df["Power"] / 1_000_000.0  # W -> MW
    df["production_source"] = "MEASURED_REAL_ENSTAB"
    df["location"] = "Borj Cedria, Ben Arous, Tunisia"
    df["latitude"] = ENSTAB_LATITUDE
    df["longitude"] = ENSTAB_LONGITUDE

    return df


if __name__ == "__main__":
    raw = load_raw()
    print("Raw data quality summary:")
    import json
    print(json.dumps(data_quality_summary(raw), indent=2))

    val = build_validation_dataset()
    print(f"\nBuilt validation dataset: {len(val):,} rows, {val['timestamp'].min()} -> {val['timestamp'].max()}")
    print(val[["timestamp", "pv_production_mw_measured", "capacity_mw", "solar_elevation_deg",
               "shortwave_radiation", "cloud_cover"]].iloc[100:105])
