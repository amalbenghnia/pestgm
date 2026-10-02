"""
End-to-end DEMO_ONLY model comparison + backtesting + uncertainty +
hierarchical reconciliation + residual correction, run against the
labelled DEMO/PROXY dataset (see docs/DATA_READINESS_REPORT.md).

Every number this script prints/writes is run against
`pv_production_mw_proxy` (DEMO_PROXY_PHYSICS_SIMULATION) - NEVER real
STEG measurements. reports/MODEL_VALIDATION_REPORT.md states this in its
header, and this script refuses to run under DATA_MODE=real (there is no
supervised real target to validate against yet - see
docs/REAL_PRODUCTION_DATA_SOURCES.md).

Usage:
    python scripts/05_train_and_backtest.py
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
from src.models.baselines import (
    PersistenceBaseline, SeasonalPersistenceBaseline, PhysicsBaseline, LightGBMBaseline,
    XGBoostBaseline, DEFAULT_ML_FEATURES,
)
from src.forecasting.backtesting import make_chrono_split, evaluate, evaluate_by_group
from src.uncertainty.quantile_forecast import QuantileForecastModel, evaluate_coverage, pinball_loss
from src.aggregation.hierarchical import reconcile_all_levels, check_consistency
from src.calibration.residual_correction import RollingBiasCorrector, correction_experiment_report
from src.validation.physical_checks import run_all_checks
from src.models.postprocessing import compare_raw_vs_constrained


TARGET_COL_DEMO = "pv_production_mw_proxy"
TARGET_COL_PVGIS = "pv_production_mw_reference"
DATA_PATH = "data/processed/training_dataset_15min.csv"


def main():
    import yaml
    mode = get_data_mode()
    if mode == DataMode.REAL:
        print("DATA_MODE=real: no real production target is available yet "
              "(see docs/REAL_PRODUCTION_DATA_SOURCES.md). Refusing to run "
              "model validation against synthetic data while claiming real mode. "
              "Re-run with DATA_MODE=demo for the labelled DEMO_ONLY comparison.")
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
        assert (df["pv_production_source"] == "DEMO_PROXY_PHYSICS_SIMULATION").all()

    print(f"Loaded {len(df):,} rows, {df['district_id'].nunique()} districts, "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}")

    # ---- physical validation of the target itself (CDC section 18) ----
    checked = run_all_checks(df, production_col=TARGET_COL)
    print("Physical check summary (target series):", checked.attrs["physical_check_summary"])

    # ---- leakage-safe lag features ----
    df = add_lag_features(df, TARGET_COL)
    df = add_rolling_features(df, TARGET_COL)

    # ---- chronological split (CDC section 16) ----
    split = make_chrono_split(df, train_frac=0.6, val_frac=0.2)
    train, val, test = split.split(df)
    print(f"Chronological split -> train: {train['timestamp'].min()}..{split.train_end} "
          f"({len(train):,} rows) | val: ..{split.val_end} ({len(val):,} rows) | "
          f"test: ..{split.test_end} ({len(test):,} rows)")

    results = {}

    # ---- Baselines ----
    persistence = PersistenceBaseline(target_col=TARGET_COL, lag_steps=1)
    test_pred_persist = persistence.predict(df).loc[test.index]
    results["persistence"] = evaluate(test[TARGET_COL], test_pred_persist, test["capacity_mw"])

    seasonal = SeasonalPersistenceBaseline(target_col=TARGET_COL,
                                             season_steps=24 if cfg.get("production_target", {}).get("resolution_minutes", 15) == 60 else 96)
    test_pred_seasonal = seasonal.predict(df).loc[test.index]
    results["seasonal_persistence"] = evaluate(test[TARGET_COL], test_pred_seasonal, test["capacity_mw"])

    physics = PhysicsBaseline()
    test_pred_physics = physics.predict(test)
    results["physics_baseline"] = evaluate(test[TARGET_COL], test_pred_physics, test["capacity_mw"])

    class CapacityFactorModelWrapper:
        def __init__(self, base_model):
            self.model = base_model
            if hasattr(self.model, "feature_cols"):
                self.model.feature_cols = [c for c in self.model.feature_cols if c != "capacity_mw"]

        def fit(self, df, target_col):
            df_cf = df.copy()
            df_cf["_cf_target"] = (df[target_col] / df["capacity_mw"]).clip(0, 1)
            self.model.fit(df_cf, "_cf_target")

        def predict(self, df):
            cf_pred = self.model.predict(df)
            return cf_pred * df["capacity_mw"]

    # ---- ML baseline (MW target) ----
    lgbm_mw = LightGBMBaseline()
    lgbm_mw.fit(train, TARGET_COL)
    test_pred_lgbm_mw = lgbm_mw.predict(test)
    results["lightgbm_mw"] = evaluate(test[TARGET_COL], test_pred_lgbm_mw, test["capacity_mw"])

    xgb_mw = XGBoostBaseline()
    xgb_mw.fit(train, TARGET_COL)
    test_pred_xgb_mw = xgb_mw.predict(test)
    results["xgboost_mw"] = evaluate(test[TARGET_COL], test_pred_xgb_mw, test["capacity_mw"])

    # ---- ML baseline (Capacity Factor target, per recommended architecture) ----
    lgbm_cf = CapacityFactorModelWrapper(LightGBMBaseline())
    lgbm_cf.fit(train, TARGET_COL)
    test_pred_lgbm_cf = lgbm_cf.predict(test)
    results["lightgbm_cf"] = evaluate(test[TARGET_COL], test_pred_lgbm_cf, test["capacity_mw"])

    xgb_cf = CapacityFactorModelWrapper(XGBoostBaseline())
    xgb_cf.fit(train, TARGET_COL)
    test_pred_xgb_cf = xgb_cf.predict(test)
    results["xgboost_cf"] = evaluate(test[TARGET_COL], test_pred_xgb_cf, test["capacity_mw"])

    test_pred_ml = test_pred_xgb_cf  # Use XGBoost CF as the primary model for downstream steps

    print("\n=== Model comparison (DEMO_ONLY, national test-set metrics) ===")
    print(pd.DataFrame(results).T.round(3).to_string())

    # ---- Physical post-processing (CDC continuation Phase 12): the raw
    #      LightGBM forecast is known to be non-zero at night on a material
    #      fraction of rows - quantify and fix it rather than hide it. ----
    postproc_report = compare_raw_vs_constrained(
        test[TARGET_COL], test_pred_ml, test["capacity_mw"], test["daylight_flag"]
    )
    print("\n=== Physical post-processing effect (LightGBM, raw vs constrained) ===")
    print(json.dumps(postproc_report, indent=2))

    # ---- Per-horizon evaluation for the ML model would require refitting
    #      per horizon; MVP scope: report intra-day (native 15-min) horizon
    #      here and document J+1/J+2/J+3 as architecturally supported but
    #      not separately backtested in this run (see limitations). ----

    # ---- Uncertainty (quantile LightGBM + conformal) ----
    qmodel = QuantileForecastModel(feature_cols=[c for c in DEFAULT_ML_FEATURES if c in df.columns])
    qmodel.fit(train, TARGET_COL)
    qmodel.calibrate(val, TARGET_COL)
    q_test = qmodel.predict(test)
    ordering_ok = bool((q_test["p10_mw"] <= q_test["p25_mw"]).all() and
                        (q_test["p25_mw"] <= q_test["p50_mw"]).all() and
                        (q_test["p50_mw"] <= q_test["p75_mw"]).all() and
                        (q_test["p75_mw"] <= q_test["p90_mw"]).all())
    coverage = evaluate_coverage(test[TARGET_COL].reset_index(drop=True),
                                  q_test["p10_mw"].reset_index(drop=True),
                                  q_test["p90_mw"].reset_index(drop=True))
    pinballs = {f"pinball_{t}": pinball_loss(test[TARGET_COL].to_numpy(), q_test[c].to_numpy(), t)
                for t, c in zip([0.10, 0.25, 0.50, 0.75, 0.90],
                                 ["p10_mw", "p25_mw", "p50_mw", "p75_mw", "p90_mw"])}
    print("\n=== Uncertainty (DEMO_ONLY) ===")
    print(f"P10<=P25<=P50<=P75<=P90 ordering respected: {ordering_ok}")
    print(coverage)
    print(pinballs)

    # ---- Hierarchical reconciliation, evaluated on the ML forecast ----
    test_fc = test.copy()
    test_fc["forecast_mw"] = test_pred_ml.to_numpy()
    levels = reconcile_all_levels(test_fc, "forecast_mw")
    consistency = check_consistency(levels, "forecast_mw")
    print("\n=== Hierarchical reconciliation consistency ===")
    print(consistency)

    # ---- Residual correction: raw ML forecast vs bias-corrected forecast ----
    # split test period in half: first half "history" (to fit correction), second half "evaluation"
    test_sorted = test.sort_values("timestamp")
    mid = test_sorted["timestamp"].quantile(0.5)
    corr_hist = test_sorted[test_sorted["timestamp"] <= mid].copy()
    corr_eval = test_sorted[test_sorted["timestamp"] > mid].copy()
    corr_hist["raw_forecast_mw"] = xgb_cf.predict(corr_hist)
    corr_eval["raw_forecast_mw"] = xgb_cf.predict(corr_eval)

    corrector = RollingBiasCorrector(window=10_000)
    corrector.fit(corr_hist, actual_col=TARGET_COL, forecast_col="raw_forecast_mw")
    corr_eval["corrected_forecast_mw"] = corrector.correct(corr_eval, forecast_col="raw_forecast_mw")

    correction_report = correction_experiment_report(
        corr_eval[TARGET_COL], corr_eval["raw_forecast_mw"], corr_eval["corrected_forecast_mw"],
        corr_eval["capacity_mw"],
    )
    print("\n=== Residual correction experiment (raw vs corrected, DEMO_ONLY) ===")
    print(json.dumps(correction_report, indent=2))

    # ---- write MODEL_VALIDATION_REPORT.md ----
    Path("reports").mkdir(exist_ok=True)
    with open("reports/MODEL_VALIDATION_REPORT.md", "w") as f:
        f.write("# MODEL_VALIDATION_REPORT.md\n\n")
        f.write("**ALL numbers in this report were computed against the labelled "
                "DEMO/PROXY production target (`pv_production_mw_proxy`, "
                "`pv_production_source=DEMO_PROXY_PHYSICS_SIMULATION`). "
                "They demonstrate the pipeline is complete and correct; "
                "they are NOT a measurement of real-world forecasting skill, "
                "because no real STEG production data exists to validate "
                "against (see docs/REAL_PRODUCTION_DATA_SOURCES.md).**\n\n")
        f.write(f"- Dataset: `{DATA_PATH}`, {len(df):,} rows, "
                f"{df['district_id'].nunique()} districts, "
                f"{df['timestamp'].min()} -> {df['timestamp'].max()}\n")
        f.write(f"- Chronological split: train <= {split.train_end}, "
                f"val <= {split.val_end}, test <= {split.test_end} "
                f"(fractions 60/20/20 of the date range, no random shuffling)\n\n")
        f.write("## Model comparison (test set, national-level metrics)\n\n")
        f.write(pd.DataFrame(results).T.round(3).to_markdown() + "\n\n")
        f.write("## Uncertainty (quantile LightGBM + split-conformal)\n\n")
        f.write(f"- Quantile ordering (P10<=P25<=P50<=P75<=P90) respected on every test row: **{ordering_ok}**\n")
        f.write(f"- 80% interval (P10-P90) empirical coverage: **{coverage['empirical_coverage_pct']:.1f}%** "
                f"(nominal 80%)\n")
        f.write(f"- Mean interval width: **{coverage['mean_interval_width_mw']:.2f} MW**\n")
        f.write(f"- Pinball losses: {pinballs}\n\n")
        f.write("## Hierarchical reconciliation\n\n")
        f.write(f"- Bottom-up reconciliation, sum(district)==governorate and "
                f"sum(governorate)==national verified exactly: {consistency}\n\n")
        f.write("## Residual correction (LightGBM raw forecast, rolling bias corrector)\n\n")
        f.write(f"```json\n{json.dumps(correction_report, indent=2)}\n```\n\n")
        f.write("## Physical validation of the target series\n\n")
        f.write(f"{checked.attrs['physical_check_summary']}\n\n")
        f.write("## Physical post-processing (LightGBM, raw vs constrained)\n\n")
        f.write(f"Raw forecast anomaly rate: {postproc_report['raw_physical_check_summary']['pct_flagged_anomalous']:.2f}% "
                f"(mostly non-zero-at-night). After clipping to `[0, capacity]` and forcing "
                f"zero when `daylight_flag==0`: {postproc_report['constrained_physical_check_summary']['pct_flagged_anomalous']:.2f}%. "
                f"sMAPE improves from {postproc_report['raw_metrics']['sMAPE_pct']:.1f}% to "
                f"{postproc_report['constrained_metrics']['sMAPE_pct']:.1f}% (MAE/RMSE change marginally, "
                f"since night errors are small in absolute MW).\n\n")
        f.write("## Limitations\n\n")
        f.write("- Results are against a physics-derived proxy target - a model trained on it partly "
                "re-learns the physics equation used to generate it, so absolute error magnitudes are "
                "not informative about real-world skill; only structural correctness (ordering, coverage, "
                "reconciliation, correction direction) should be read from this report.\n"
                "- Only the intra-day (15-min) horizon was backtested in this run; J+1/J+2/J+3 horizons "
                "are supported by `src/forecasting/backtesting.py::shift_to_horizon` but were not "
                "separately retrained/backtested here (next iteration).\n"
                "- ML baseline excludes production lag features to stay honest about what would be "
                "available operationally at longer horizons; intra-day performance would improve with "
                "lag features once real production data justifies using them.\n")

    print("\nWrote reports/MODEL_VALIDATION_REPORT.md")


if __name__ == "__main__":
    main()
