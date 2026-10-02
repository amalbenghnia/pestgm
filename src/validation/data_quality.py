"""
Data quality checks (CDC continuation Phase 16). Complements
src/validation/physical_checks.py (which checks production values) with
checks on the dataset's structural integrity.
"""
from __future__ import annotations

import pandas as pd


def check_missing_timestamps(df: pd.DataFrame, freq: str, group_col: str = "district_id") -> pd.DataFrame:
    """Per-group: expected vs actual row count at the given frequency, and
    the list of missing timestamps (capped for readability)."""
    rows = []
    for gid, g in df.groupby(group_col):
        expected = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=freq)
        missing = expected.difference(pd.DatetimeIndex(g["timestamp"]))
        rows.append({
            group_col: gid, "expected_rows": len(expected), "actual_rows": len(g),
            "n_missing_timestamps": len(missing),
            "pct_missing": round(100 * len(missing) / max(len(expected), 1), 3),
            "sample_missing": list(missing[:5].astype(str)),
        })
    return pd.DataFrame(rows)


def check_duplicate_timestamps(df: pd.DataFrame, group_col: str = "district_id") -> pd.DataFrame:
    dup = df.duplicated(subset=[group_col, "timestamp"], keep=False)
    return df[dup].sort_values([group_col, "timestamp"])


def check_negative_production(df: pd.DataFrame, production_col: str) -> pd.DataFrame:
    return df[df[production_col] < 0]


def check_production_exceeds_capacity(df: pd.DataFrame, production_col: str,
                                       capacity_col: str = "capacity_mw", tol: float = 1e-6) -> pd.DataFrame:
    return df[df[production_col] > df[capacity_col] + tol]


def check_impossible_night_production(df: pd.DataFrame, production_col: str,
                                       daylight_col: str = "daylight_flag", tol: float = 1e-6) -> pd.DataFrame:
    return df[(df[daylight_col] == 0) & (df[production_col].abs() > tol)]


def check_weather_gaps(df: pd.DataFrame, weather_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in weather_cols:
        if c not in df.columns:
            continue
        n_missing = df[c].isna().sum()
        rows.append({"variable": c, "n_missing": int(n_missing), "pct_missing": round(100 * n_missing / len(df), 3)})
    return pd.DataFrame(rows)


def check_capacity_changes(cap_ts: pd.DataFrame, max_daily_change_pct: float = 20.0) -> pd.DataFrame:
    """Flags any district/day where capacity jumps more than max_daily_change_pct
    day-over-day (sanity check on the snapshot interpolation)."""
    rows = []
    for did, g in cap_ts.sort_values("timestamp").groupby("district_id"):
        g = g.set_index("timestamp")
        daily = g["capacity_mw"].resample("1D").last()
        pct_change = 100 * daily.pct_change().abs()
        flagged = pct_change[pct_change > max_daily_change_pct]
        for ts, val in flagged.items():
            rows.append({"district_id": did, "date": ts, "pct_change": round(float(val), 2)})
    return pd.DataFrame(rows)


def full_data_quality_report(df: pd.DataFrame, production_col: str, freq: str = "15min") -> dict:
    weather_cols = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation",
                     "direct_radiation", "diffuse_radiation", "wind_speed_10m", "precipitation"]
    missing_ts = check_missing_timestamps(df, freq=freq)
    dup_ts = check_duplicate_timestamps(df)
    neg_prod = check_negative_production(df, production_col)
    over_cap = check_production_exceeds_capacity(df, production_col) if "capacity_mw" in df.columns else pd.DataFrame()
    night_prod = check_impossible_night_production(df, production_col) if "daylight_flag" in df.columns else pd.DataFrame()
    weather_gaps = check_weather_gaps(df, weather_cols)

    return {
        "n_rows": len(df),
        "n_districts": df["district_id"].nunique() if "district_id" in df.columns else None,
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "missing_timestamps_worst_district_pct": float(missing_ts["pct_missing"].max()) if len(missing_ts) else 0.0,
        "n_duplicate_timestamp_rows": len(dup_ts),
        "n_negative_production_rows": len(neg_prod),
        "n_production_exceeds_capacity_rows": len(over_cap),
        "n_impossible_night_production_rows": len(night_prod),
        "pct_impossible_night_production": round(100 * len(night_prod) / len(df), 3) if len(df) else 0.0,
        "weather_gaps": weather_gaps.to_dict(orient="records"),
    }
