"""
Physical post-processing layer (CDC continuation Phase 12).

The repo's own physical-check pipeline found that raw ML forecasts violate
0<=P<=capacity and P=0-at-night on a non-trivial fraction of rows (previous
session: ~22% of intraday test rows non-zero at night). Rather than hiding
this, this module implements the documented fix and reports its effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_physical_constraints(forecast_mw: pd.Series, capacity_mw: pd.Series,
                                daylight_flag: pd.Series | None = None,
                                solar_elevation_deg: pd.Series | None = None) -> pd.Series:
    """Clips forecast to [0, capacity]; zeroes it out at night. Prefers
    daylight_flag when given, else solar_elevation_deg <= 0."""
    out = forecast_mw.clip(lower=0)
    out = np.minimum(out, capacity_mw)
    if daylight_flag is not None:
        out = out.where(daylight_flag != 0, 0.0)
    elif solar_elevation_deg is not None:
        out = out.where(solar_elevation_deg > 0, 0.0)
    return out


def compare_raw_vs_constrained(y_true: pd.Series, raw_forecast: pd.Series,
                                capacity_mw: pd.Series, daylight_flag: pd.Series) -> dict:
    from src.forecasting.backtesting import evaluate
    from src.validation.physical_checks import run_all_checks

    constrained = apply_physical_constraints(raw_forecast, capacity_mw, daylight_flag=daylight_flag)

    raw_df = pd.DataFrame({"capacity_mw": capacity_mw, "daylight_flag": daylight_flag, "f": raw_forecast})
    con_df = pd.DataFrame({"capacity_mw": capacity_mw, "daylight_flag": daylight_flag, "f": constrained})
    raw_checked = run_all_checks(raw_df, production_col="f")
    con_checked = run_all_checks(con_df, production_col="f")

    return {
        "raw_metrics": evaluate(y_true, raw_forecast, capacity_mw),
        "constrained_metrics": evaluate(y_true, constrained, capacity_mw),
        "raw_physical_check_summary": raw_checked.attrs["physical_check_summary"],
        "constrained_physical_check_summary": con_checked.attrs["physical_check_summary"],
    }
