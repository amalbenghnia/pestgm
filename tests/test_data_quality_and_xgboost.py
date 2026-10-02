import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.data_quality import (
    check_missing_timestamps, check_duplicate_timestamps, check_negative_production,
    check_production_exceeds_capacity, check_impossible_night_production, full_data_quality_report,
)
from src.models.baselines import XGBoostBaseline
from src.models.postprocessing import apply_physical_constraints, compare_raw_vs_constrained


def test_check_missing_timestamps_detects_gap():
    ts = pd.date_range("2026-01-01", periods=10, freq="15min").delete(5)  # drop one timestamp
    df = pd.DataFrame({"district_id": ["D01"] * len(ts), "timestamp": ts})
    report = check_missing_timestamps(df, freq="15min")
    assert report.iloc[0]["n_missing_timestamps"] == 1


def test_check_duplicate_timestamps():
    ts = pd.date_range("2026-01-01", periods=5, freq="15min")
    df = pd.DataFrame({"district_id": ["D01"] * 5, "timestamp": list(ts[:-1]) + [ts[0]]})
    dup = check_duplicate_timestamps(df)
    assert len(dup) == 2  # the two rows sharing ts[0]


def test_check_negative_and_over_capacity():
    df = pd.DataFrame({"production": [-1, 5, 15], "capacity_mw": [10, 10, 10]})
    assert len(check_negative_production(df, "production")) == 1
    assert len(check_production_exceeds_capacity(df, "production")) == 1


def test_check_impossible_night_production():
    df = pd.DataFrame({"production": [0, 3, 0], "daylight_flag": [0, 1, 0]})
    assert len(check_impossible_night_production(df, "production")) == 0
    df2 = pd.DataFrame({"production": [2, 3, 0], "daylight_flag": [0, 1, 0]})
    assert len(check_impossible_night_production(df2, "production")) == 1


def test_full_data_quality_report_runs():
    n = 100
    df = pd.DataFrame({
        "district_id": ["D01"] * n,
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "prod": np.clip(np.random.default_rng(0).normal(2, 1, n), 0, 10),
        "capacity_mw": 10.0,
    })
    report = full_data_quality_report(df, production_col="prod")
    assert report["n_rows"] == n


def test_xgboost_baseline_fits_and_predicts():
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "capacity_mw": 10.0,
        "shortwave_radiation": rng.uniform(0, 900, n),
        "direct_radiation": rng.uniform(0, 700, n),
        "diffuse_radiation": rng.uniform(0, 200, n),
        "temperature_2m": rng.uniform(15, 35, n),
        "relative_humidity_2m": rng.uniform(20, 90, n),
        "cloud_cover": rng.uniform(0, 100, n),
        "wind_speed_10m": rng.uniform(0, 20, n),
        "solar_elevation_deg": rng.uniform(-10, 80, n),
        "solar_azimuth_deg": rng.uniform(0, 360, n),
        "clearsky_ghi_wm2": rng.uniform(0, 1000, n),
        "daylight_flag": rng.integers(0, 2, n),
        "hour_sin": rng.uniform(-1, 1, n), "hour_cos": rng.uniform(-1, 1, n),
        "doy_sin": rng.uniform(-1, 1, n), "doy_cos": rng.uniform(-1, 1, n),
        "day_of_week": rng.integers(0, 7, n), "month": rng.integers(1, 13, n),
    })
    df["target"] = np.clip(df["shortwave_radiation"] / 100, 0, df["capacity_mw"])
    m = XGBoostBaseline()
    m.fit(df, "target")
    pred = m.predict(df)
    assert (pred >= 0).all()
    assert len(pred) == n


def test_apply_physical_constraints_clips_and_zeroes_night():
    forecast = pd.Series([-1, 5, 15, 3])
    capacity = pd.Series([10, 10, 10, 10])
    daylight = pd.Series([1, 1, 1, 0])
    out = apply_physical_constraints(forecast, capacity, daylight_flag=daylight)
    assert list(out) == [0, 5, 10, 0]


def test_compare_raw_vs_constrained_reduces_violations():
    n = 50
    rng = np.random.default_rng(0)
    y_true = pd.Series(np.clip(rng.normal(3, 1, n), 0, 10))
    raw = y_true + rng.normal(0, 0.3, n)  # small noise, some negative/over-cap/night leakage
    capacity = pd.Series([10.0] * n)
    daylight = pd.Series([1] * (n - 10) + [0] * 10)  # last 10 rows are "night"
    report = compare_raw_vs_constrained(y_true, raw, capacity, daylight)
    assert report["constrained_physical_check_summary"]["pct_flagged_anomalous"] <= \
        report["raw_physical_check_summary"]["pct_flagged_anomalous"]
