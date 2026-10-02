"""
Multi-horizon backtest with 4 production-grade improvements:
  1. Real archived forecast weather (GFS/ECMWF via Open-Meteo previous_dayX)
     for J+1/J+2/J+3 - eliminates "Perfect Forecast" illusion.
  2. Rolling Conformal Prediction for P10/P90 - dynamically adapts interval
     width to seasonal variance drift.
  3. Rolling Bias Corrector per horizon - de-biases the growing positive
     offset at longer horizons.
  4. WAPE + MAE-diurne + Ramp MAE as primary evaluation protocol.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.data_mode import get_data_mode, DataMode
from src.features.lag_features import add_lag_features, add_rolling_features
from src.forecasting.backtesting import make_chrono_split, evaluate, evaluate_by_group
from src.forecasting.multi_horizon import HorizonModel, HORIZONS
from src.uncertainty.quantile_forecast import QuantileForecastModel, evaluate_coverage, pinball_loss
from src.validation.physical_checks import run_all_checks
from src.models.postprocessing import compare_raw_vs_constrained

TARGET_COL_DEMO = "pv_production_mw_proxy"
TARGET_COL_PVGIS = "pv_production_mw_reference"
DATA_PATH = "data/processed/training_dataset_15min.csv"

# Mapping from horizon to Open-Meteo previous_dayX columns (real archived forecasts)
FORECAST_WEATHER_MAP = {
    "J+1": 1,
    "J+2": 2,
    "J+3": 3,
}

# Real forecast weather variables available via Open-Meteo historical-forecast-api
FORECAST_WEATHER_BASE_VARS = [
    "temperature_2m", "shortwave_radiation", "cloud_cover",
]


# ===========================================================================
# Improved evaluation metrics (Step 4)
# ===========================================================================

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error - robust to near-zero values."""
    denom = np.nansum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.nansum(np.abs(y_true - y_pred)) / denom * 100)


def mae_diurne(y_true: np.ndarray, y_pred: np.ndarray, daylight: np.ndarray) -> float:
    """MAE calculated only on daylight hours (elevation > 0)."""
    mask = daylight.astype(bool)
    if mask.sum() == 0:
        return np.nan
    return float(np.nanmean(np.abs(y_true[mask] - y_pred[mask])))


def ramp_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE of ramp rates: measures ability to predict rapid production changes."""
    ramp_true = np.diff(y_true)
    ramp_pred = np.diff(y_pred)
    valid = ~(np.isnan(ramp_true) | np.isnan(ramp_pred))
    if valid.sum() == 0:
        return np.nan
    return float(np.nanmean(np.abs(ramp_true[valid] - ramp_pred[valid])))


def evaluate_full(y_true: pd.Series, y_pred: pd.Series, capacity: pd.Series,
                  daylight: pd.Series) -> dict:
    """Full evaluation including standard + improved metrics."""
    from src.forecasting.backtesting import evaluate
    base = evaluate(y_true, y_pred, capacity)
    yt = y_true.to_numpy()
    yp = y_pred.to_numpy()
    dl = daylight.to_numpy() if daylight is not None else np.ones(len(yt))
    base["WAPE_pct"] = wape(yt, yp)
    base["MAE_diurne_MW"] = mae_diurne(yt, yp, dl)
    base["ramp_MAE_MW"] = ramp_mae(yt, yp)
    return base


# ===========================================================================
# Rolling Conformal Predictor (Step 2)
# ===========================================================================

class RollingConformalPredictor:
    """
    Maintains a rolling window of conformal nonconformity scores.
    At each step, dynamically adjusts the P10/P90 expansion factor
    based on the last `window` calibration observations.
    This ensures the 80% coverage guarantee is maintained even as
    the error distribution shifts seasonally.
    """

    def __init__(self, window: int = 14 * 24, target_coverage: float = 0.80):
        self.window = window
        self.target_coverage = target_coverage
        self._scores: list[float] = []

    def update(self, y_true: float, q_low: float, q_high: float, epsilon: float = 1e-6):
        """Add a new nonconformity score after observing y_true."""
        width = max(q_high - q_low, epsilon)
        score = max(q_low - y_true, y_true - q_high) / width
        self._scores.append(score)
        if len(self._scores) > self.window:
            self._scores.pop(0)

    def get_expansion(self, epsilon: float = 1e-6) -> float:
        """Compute the current expansion factor from the rolling score buffer."""
        if len(self._scores) == 0:
            return 0.0
        n = len(self._scores)
        q_level = min(self.target_coverage * (1 + 1.0 / n), 1.0)
        return float(np.quantile(self._scores, q_level))

    def calibrate_static(self, scores: list[float]):
        """Warm-start from a static calibration set (val set)."""
        self._scores = list(scores[-self.window:])


# ===========================================================================
# Rolling Bias Corrector per horizon (Step 3)
# ===========================================================================

class HorizonBiasCorrector:
    """
    Maintains a rolling mean bias estimate per horizon.
    Applies an additive correction so the forecast is de-biased.
    """

    def __init__(self, window: int = 7 * 24):
        self.window = window
        self._residuals: list[float] = []

    def update(self, y_true: float, y_pred: float):
        self._residuals.append(y_true - y_pred)
        if len(self._residuals) > self.window:
            self._residuals.pop(0)

    def correction(self) -> float:
        if len(self._residuals) == 0:
            return 0.0
        return float(np.mean(self._residuals))

    def calibrate_static(self, residuals: list[float]):
        self._residuals = list(residuals[-self.window:])


# ===========================================================================
# CF model wrappers (from previous step)
# ===========================================================================

class CFHorizonModelWrapper:
    def __init__(self, base_hm):
        self.model = base_hm
        if "capacity_mw" in self.model.feature_cols:
            self.model.feature_cols.remove("capacity_mw")

    def fit(self, df_h):
        df_cf = df_h.copy()
        df_cf["target_at_horizon"] = (df_cf["target_at_horizon"] / df_cf["capacity_mw"]).clip(0, 1)
        self.model.fit(df_cf)

    def predict(self, df_h):
        cf_pred = self.model.predict(df_h)
        return cf_pred * df_h["capacity_mw"]


class CFQuantileWrapper:
    def __init__(self, feature_cols):
        self.feature_cols = [c for c in feature_cols if c != "capacity_mw"]
        self.model = QuantileForecastModel(feature_cols=self.feature_cols)

    def fit(self, df_h, target_col):
        df_cf = df_h.copy()
        df_cf[target_col] = (df_cf[target_col] / df_cf["capacity_mw"]).clip(0, 1)
        self.model.fit(df_cf, target_col)

    def calibrate(self, df_h, target_col):
        df_cf = df_h.copy()
        df_cf[target_col] = (df_cf[target_col] / df_cf["capacity_mw"]).clip(0, 1)
        self.model.calibrate(df_cf, target_col)

    def predict(self, df_h):
        q_cf = self.model.predict(df_h)
        for c in q_cf.columns:
            q_cf[c] = q_cf[c] * df_h["capacity_mw"].values
        return q_cf


# ===========================================================================
# Main
# ===========================================================================

def swap_to_forecast_weather(df: pd.DataFrame, day: int) -> pd.DataFrame:
    """
    Step 1: Replace reanalysis weather columns with real archived forecast
    (previous_dayX). Falls back gracefully if columns are missing.
    """
    df = df.copy()
    for v in FORECAST_WEATHER_BASE_VARS:
        fc_col = f"{v}_previous_day{day}"
        if fc_col in df.columns:
            non_null = df[fc_col].notna().mean()
            if non_null > 0.5:  # only swap if data coverage > 50%
                df[v] = df[fc_col]
    return df


def main():
    import yaml
    mode = get_data_mode()
    if mode == DataMode.REAL:
        print("DATA_MODE=real: refusing (no real production target available).")
        sys.exit(1)

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    production_mode = cfg.get("production_target", {}).get("mode", "DEMO_PROXY")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    if production_mode == "PVGIS_REFERENCE" and TARGET_COL_PVGIS in df.columns:
        TARGET_COL = TARGET_COL_PVGIS
        print(f"[INFO] Using PVGIS_REFERENCE target: {TARGET_COL}")
        df = df.dropna(subset=[TARGET_COL])
    else:
        TARGET_COL = TARGET_COL_DEMO

    # Check if archived forecast weather is available
    forecast_cols_available = [c for c in df.columns if "_previous_day" in c]
    has_forecast_weather = len(forecast_cols_available) > 0
    if has_forecast_weather:
        print(f"[INFO] Real archived forecast weather columns found: {len(forecast_cols_available)} columns")
        print(f"       J+1/J+2/J+3 backtests will use real forecast weather (no Perfect Forecast illusion).")
    else:
        print("[WARN] No archived forecast weather columns found.")
        print("       Re-run 'python scripts/02_build_weather_dataset.py --start 2020-01-01 --end 2020-12-31' to fetch them.")
        print("       Falling back to reanalysis weather for all horizons (overoptimistic J+1/J+3).")

    df = add_lag_features(df, TARGET_COL)
    df = add_rolling_features(df, TARGET_COL)

    print("\n--- Applying multi-year temporal split ---")
    train = df[df["timestamp"].dt.year.isin([2020, 2021])].copy()
    val = df[df["timestamp"].dt.year == 2022].copy()
    test = df[df["timestamp"].dt.year == 2023].copy()
    
    print(f"Train (2020-2021): {len(train):,} rows")
    print(f"Validation (2022): {len(val):,} rows")
    print(f"Test (2023):       {len(test):,} rows")

    if len(test) == 0:
        print("WARNING: Test set is empty (2023 not found). Falling back to chrono split.")
        split = make_chrono_split(df, train_frac=0.6, val_frac=0.2)
        train, val, test = split.split(df)

    all_point_results = {}
    all_uncertainty_results = {}
    all_group_results = {}
    all_postproc_results = {}

    for horizon_name in HORIZONS:
        print(f"\n=== Horizon: {horizon_name} ===")
        feats = HORIZONS[horizon_name]["features"]
        hm = HorizonModel(horizon_name=horizon_name, feature_cols=list(feats))
        hm = CFHorizonModelWrapper(hm)

        # Step 1: Swap weather to real archived forecasts for J+1/J+2/J+3
        forecast_day = FORECAST_WEATHER_MAP.get(horizon_name)
        if forecast_day and has_forecast_weather:
            train_input = swap_to_forecast_weather(train, forecast_day)
            val_input = swap_to_forecast_weather(val, forecast_day)
            test_input = swap_to_forecast_weather(test, forecast_day)
            print(f"  [Step 1] Using real J+{forecast_day} archived forecast weather for training/evaluation.")
        else:
            train_input = train
            val_input = val
            test_input = test

        train_h = hm.model.build_supervised_frame(train_input, TARGET_COL)
        val_h = hm.model.build_supervised_frame(val_input, TARGET_COL)
        test_h = hm.model.build_supervised_frame(test_input, TARGET_COL)

        hm.fit(train_h)
        pred_h = hm.predict(test_h)

        # Step 3: TRUE Online Rolling Bias Corrector (sorted by ISSUE time, not target time)
        # Using forecast_issue_time avoids val/test target-timestamp overlap for J+1/J+2/J+3.
        val_pred = hm.predict(val_h)
        val_test_y = pd.concat([val_h["target_at_horizon"], test_h["target_at_horizon"]])
        val_test_pred = pd.concat([val_pred, pred_h])
        # Use ISSUE timestamps: val issues always precede test issues (no overlap)
        val_test_issue_ts = pd.concat([val_h["forecast_issue_time"], test_h["forecast_issue_time"]])

        residuals = val_test_y - val_test_pred
        res_df = pd.DataFrame({"issue_ts": val_test_issue_ts, "residual": residuals})

        # Average residual per issue-timestamp across all districts
        daily_res = res_df.groupby("issue_ts")["residual"].mean().sort_index()

        # Rolling mean over the last 7 days (7 * 96 steps = 672 steps at 15-min resolution)
        # Shift by 1: at issue time T, we use residuals from T-1 and earlier (causal)
        rolling_bias = daily_res.rolling(window=672, min_periods=1).mean().shift(1).fillna(0.0)

        # Map back to test_h by ISSUE timestamp
        test_h_bias = test_h["forecast_issue_time"].map(rolling_bias).fillna(0.0)
        pred_h_corrected = (pred_h + test_h_bias).clip(lower=0)

        avg_bias_applied = test_h_bias.mean()
        print(f"  [Step 3] Online Rolling Bias applied (average): {avg_bias_applied:+.4f} MW")

        valid_rows = test_h["target_at_horizon"].notna() & pred_h_corrected.notna()

        # Step 4: Full metrics with WAPE + MAE-diurne + Ramp MAE
        daylight_col = test_h["daylight_flag"] if "daylight_flag" in test_h.columns else pd.Series(
            np.ones(len(test_h)), index=test_h.index)
        m = evaluate_full(
            test_h.loc[valid_rows, "target_at_horizon"],
            pred_h_corrected[valid_rows],
            test_h.loc[valid_rows, "capacity_mw"],
            daylight_col[valid_rows],
        )
        all_point_results[horizon_name] = m
        print("Point forecast metrics:", {k: round(v, 4) for k, v in m.items()})

        # Physical post-processing
        fc_df = test_h.loc[valid_rows, ["capacity_mw", "daylight_flag"]].copy()
        fc_df["forecast_mw"] = pred_h_corrected[valid_rows].to_numpy()
        checked = run_all_checks(fc_df, production_col="forecast_mw")
        print("Forecast physical-check summary (raw):", checked.attrs["physical_check_summary"])

        postproc = compare_raw_vs_constrained(
            test_h.loc[valid_rows, "target_at_horizon"],
            pred_h_corrected[valid_rows],
            test_h.loc[valid_rows, "capacity_mw"],
            test_h.loc[valid_rows, "daylight_flag"],
        )
        all_postproc_results[horizon_name] = postproc
        print("Physical post-processing effect:", {
            "raw_pct_flagged": postproc["raw_physical_check_summary"]["pct_flagged_anomalous"],
            "constrained_pct_flagged": postproc["constrained_physical_check_summary"]["pct_flagged_anomalous"],
            "raw_sMAPE": postproc["raw_metrics"]["sMAPE_pct"],
            "constrained_sMAPE": postproc["constrained_metrics"]["sMAPE_pct"],
        })

        # Step 2: TRUE Online Rolling Conformal Prediction (causal, sorted by ISSUE time)
        # Sorting by forecast_issue_time guarantees val issues strictly precede test issues.
        # For J+3 target_timestamp, val and test OVERLAP (val target Oct = test issue Oct+3),
        # which contaminates the window. Issue time has no such overlap.
        qfeat = [c for c in feats if c in train_h.columns]
        qm = CFQuantileWrapper(feature_cols=qfeat)
        qm.fit(train_h, "target_at_horizon")
        qm.calibrate(val_h, "target_at_horizon")

        q_val = qm.predict(val_h)
        q_test = qm.predict(test_h)

        # Compute nonconformity scores for val + test (combined timeline by ISSUE time)
        val_test_ql = pd.concat([q_val["p10_mw"], q_test["p10_mw"]]).values
        val_test_qh = pd.concat([q_val["p90_mw"], q_test["p90_mw"]]).values
        val_test_y_arr = val_test_y.values
        # Daylight flag for both val and test (needed to filter night rows)
        val_test_dl = pd.concat([
            val_h["daylight_flag"] if "daylight_flag" in val_h.columns else pd.Series(np.ones(len(val_h)), index=val_h.index),
            test_h["daylight_flag"] if "daylight_flag" in test_h.columns else pd.Series(np.ones(len(test_h)), index=test_h.index),
        ]).values.astype(bool)

        score_width = np.maximum(val_test_qh - val_test_ql, 1e-6)
        scores_arr = np.maximum(val_test_ql - val_test_y_arr, val_test_y_arr - val_test_qh) / score_width

        # Build timeline sorted by ISSUE time — DAYLIGHT ONLY
        # CRITICAL: night scores (y=0, q10=q90=0) are trivially score=0 and dilute the rolling Q80
        # to ≈0, preventing the conformal predictor from detecting real daytime under-coverage.
        # Calibrating only on daytime hours gives a Q80 that truly reflects production uncertainty.
        n_districts = test_h["district_id"].nunique()
        score_time_df = pd.DataFrame({
            "issue_ts": val_test_issue_ts.values,
            "score": scores_arr,
            "is_day": val_test_dl,
        })
        # Filter to daylight rows only for the rolling calibration
        score_time_day = (
            score_time_df[score_time_df["is_day"]]
            .sort_values("issue_ts")
            .reset_index(drop=True)
        )

        # Detect actual data resolution (hourly vs 15-min)
        if len(score_time_day) > 1:
            median_gap_min = score_time_day["issue_ts"].diff().dt.total_seconds().median() / 60
            steps_per_hour = max(1, round(60 / median_gap_min)) if median_gap_min > 0 else 1
        else:
            steps_per_hour = 1

        # Rolling 80th percentile over a 5-day daylight-only window
        daylight_hrs_per_day = 11  # conservative for winter Tunisia
        window_rows = max(50, 5 * daylight_hrs_per_day * steps_per_hour * n_districts)
        min_rows = max(10, 1 * daylight_hrs_per_day * steps_per_hour * n_districts)
        rolling_q80 = (
            score_time_day["score"]
            .rolling(window=window_rows, min_periods=min_rows)
            .quantile(0.80)
        )
        # Shift by 1: causal — at issue time T, use only scores from T-1 and earlier
        score_time_day["rolling_expansion"] = rolling_q80.shift(1).fillna(0.0).clip(lower=0.0)

        # Aggregate by issue_ts (mean across districts)
        ts_expansion_map = score_time_day.groupby("issue_ts")["rolling_expansion"].mean()

        # Map to test_h by ISSUE timestamp. Night rows get 0 expansion (correct: q10=q90=0 anyway)
        expansion_series = test_h["forecast_issue_time"].map(ts_expansion_map).fillna(0.0)
        avg_expansion = expansion_series.mean()
        pct_nonzero = (expansion_series > 0).mean() * 100
        print(f"  [Step 2] Online Rolling Conformal (daylight-calibrated): avg expansion={avg_expansion:+.4f}, "
              f"active on {pct_nonzero:.1f}% of test rows")

        # Apply per-row rolling conformal expansion to test quantiles
        q_test_adj = q_test.copy()
        exp_vals = expansion_series.values
        width_test = (q_test["p90_mw"] - q_test["p10_mw"]).clip(lower=1e-6).values
        q_test_adj["p10_mw"] = np.maximum(q_test["p10_mw"].values - exp_vals * width_test, 0)
        q_test_adj["p90_mw"] = np.maximum(q_test["p90_mw"].values + exp_vals * width_test, 0)

        width_50 = (q_test["p75_mw"] - q_test["p25_mw"]).clip(lower=1e-6).values
        q_test_adj["p25_mw"] = np.maximum(q_test["p25_mw"].values - exp_vals * width_50, 0)
        q_test_adj["p75_mw"] = np.maximum(q_test["p75_mw"].values + exp_vals * width_50, 0)

        cov = evaluate_coverage(
            test_h["target_at_horizon"].reset_index(drop=True),
            q_test_adj["p10_mw"].reset_index(drop=True),
            q_test_adj["p90_mw"].reset_index(drop=True),
            daylight=test_h["daylight_flag"].reset_index(drop=True) if "daylight_flag" in test_h.columns else None
        )
        pb = {f"pinball_{t}": pinball_loss(test_h["target_at_horizon"].to_numpy(), q_test_adj[c].to_numpy(), t)
              for t, c in zip([0.10, 0.25, 0.50, 0.75, 0.90],
                               ["p10_mw", "p25_mw", "p50_mw", "p75_mw", "p90_mw"])}
        all_uncertainty_results[horizon_name] = {**cov, **pb}
        print("Uncertainty:", all_uncertainty_results[horizon_name])

        # Per-governorate breakdown
        gdf = test_h.loc[valid_rows, ["governorate", "capacity_mw"]].copy()
        gdf["y_true"] = test_h.loc[valid_rows, "target_at_horizon"].to_numpy()
        gdf["y_pred"] = pred_h_corrected[valid_rows].to_numpy()
        gres = evaluate_by_group(gdf, "y_true", "y_pred", "capacity_mw", ["governorate"])
        all_group_results[horizon_name] = gres

    # Write report
    Path("reports").mkdir(exist_ok=True)
    with open("reports/MULTI_HORIZON_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# MULTI_HORIZON_REPORT.md\n\n")
        f.write("## Methodological improvements applied\n\n")
        f.write("| Step | Improvement | Status |\n|---|---|---|\n")
        f.write(f"| 1 | Real archived forecast weather (J+1/J+2/J+3) | {'✅ Active' if has_forecast_weather else '⚠️ Fallback (reanalysis)'} |\n")
        f.write("| 2 | Rolling Conformal Prediction (P10/P90) | ✅ Active |\n")
        f.write("| 3 | Rolling Bias Corrector per horizon | ✅ Active |\n")
        f.write("| 4 | WAPE + MAE-diurne + Ramp MAE metrics | ✅ Active |\n\n")
        f.write("## Point-forecast metrics by horizon (national-level test set)\n\n")
        f.write(pd.DataFrame(all_point_results).T.round(4).to_markdown() + "\n\n")
        f.write("## Uncertainty by horizon (80% interval coverage, pinball losses)\n\n")
        f.write(pd.DataFrame(all_uncertainty_results).T.round(4).to_markdown() + "\n\n")
        f.write("## Per-governorate metrics, by horizon\n\n")
        for h, gres in all_group_results.items():
            f.write(f"### {h}\n\n")
            f.write(gres.round(3).to_markdown(index=False) + "\n\n")
        f.write("## Physical post-processing effect, by horizon\n\n")
        pp_rows = []
        for h, r in all_postproc_results.items():
            pp_rows.append({
                "horizon": h,
                "raw_pct_flagged_anomalous": r["raw_physical_check_summary"]["pct_flagged_anomalous"],
                "constrained_pct_flagged_anomalous": r["constrained_physical_check_summary"]["pct_flagged_anomalous"],
                "raw_sMAPE_pct": r["raw_metrics"]["sMAPE_pct"],
                "constrained_sMAPE_pct": r["constrained_metrics"]["sMAPE_pct"],
                "raw_MAE_MW": r["raw_metrics"]["MAE_MW"],
                "constrained_MAE_MW": r["constrained_metrics"]["MAE_MW"],
            })
        f.write(pd.DataFrame(pp_rows).round(3).to_markdown(index=False) + "\n\n")

    print("\nWrote reports/MULTI_HORIZON_REPORT.md")


if __name__ == "__main__":
    main()
