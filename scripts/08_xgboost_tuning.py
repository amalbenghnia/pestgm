"""
Time-aware XGBoost hyperparameter search (CDC continuation Phase 10).

Process (documented, followed exactly):
    TRAIN -> time-based validation -> hyperparameter selection -> LOCK
    PARAMETERS -> FINAL TEST (touched only once, at the very end)

Never random k-fold CV on time series. Never any parameter chosen by looking
at the test split.

Usage:
    python scripts/08_xgboost_tuning.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.backtesting import make_chrono_split, evaluate
from src.models.baselines import DEFAULT_ML_FEATURES

DATA_PATH = "data/processed/training_dataset_15min_sample.csv"
TARGET_COL = "pv_production_mw_proxy"

# Search space (kept modest - this is a demo-scale dataset; a wider grid
# would overfit the small validation split it's tuned against)
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.08],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "min_child_weight": [1, 5],
    "reg_alpha": [0.0, 0.5],
    "reg_lambda": [1.0, 2.0],
    "gamma": [0.0, 0.5],
}


def random_search(train, val, feature_cols, target_col, n_trials=25, seed=42):
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    keys = list(PARAM_GRID.keys())
    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]
    valid_train = y_train.notna() & X_train.notna().all(axis=1)
    valid_val = y_val.notna() & X_val.notna().all(axis=1)

    results = []
    for trial in range(n_trials):
        params = {k: rng.choice(PARAM_GRID[k]).item() if hasattr(rng.choice(PARAM_GRID[k]), "item")
                   else rng.choice(PARAM_GRID[k]) for k in keys}
        params["random_state"] = seed
        params["objective"] = "reg:squarederror"
        params["n_jobs"] = -1
        model = xgb.XGBRegressor(**params)
        model.fit(X_train[valid_train], y_train[valid_train])
        pred = model.predict(X_val[valid_val])
        m = evaluate(y_val[valid_val], pred, val.loc[valid_val, "capacity_mw"])
        results.append({"trial": trial, **params, **{f"val_{k}": v for k, v in m.items()}})
        print(f"trial {trial}: val_MAE={m['MAE_MW']:.4f} params={params}")

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    split = make_chrono_split(df, train_frac=0.6, val_frac=0.2)
    train, val, test = split.split(df)

    feature_cols = [c for c in DEFAULT_ML_FEATURES if c in df.columns]
    results = random_search(train, val, feature_cols, TARGET_COL, n_trials=25)

    best_row = results.loc[results["val_MAE_MW"].idxmin()]
    best_params = {k: (int(best_row[k]) if k in ["n_estimators", "max_depth", "min_child_weight"] else float(best_row[k]))
                    for k in PARAM_GRID}
    best_params.update({"random_state": 42, "objective": "reg:squarederror", "n_jobs": -1})
    print(f"\nBest params (by validation MAE={best_row['val_MAE_MW']:.4f}): {best_params}")

    # LOCK parameters, then evaluate ONCE on the untouched test split
    import xgboost as xgb
    X_trainval = pd.concat([train, val])[feature_cols]
    y_trainval = pd.concat([train, val])[TARGET_COL]
    valid = y_trainval.notna() & X_trainval.notna().all(axis=1)
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_trainval[valid], y_trainval[valid])

    X_test, y_test = test[feature_cols], test[TARGET_COL]
    valid_test = y_test.notna() & X_test.notna().all(axis=1)
    test_pred = final_model.predict(X_test[valid_test])
    test_metrics = evaluate(y_test[valid_test], test_pred, test.loc[valid_test, "capacity_mw"])
    print(f"\nFinal (locked-params) TEST metrics (touched once): {test_metrics}")

    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/xgboost_tuning_trials.csv", index=False)
    with open("reports/xgboost_best_params.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "validation_metrics": {k: best_row[f"val_{k}"] for k in ["MAE_MW", "RMSE_MW", "nMAE_pct_of_capacity", "bias_MW"]},
            "final_test_metrics": test_metrics,
            "note": "Tuned by time-aware train/val split (60/20 of date range); "
                    "test split (remaining 20%) evaluated exactly once with locked parameters.",
        }, f, indent=2)
    print("\nWrote reports/xgboost_tuning_trials.csv and reports/xgboost_best_params.json")


if __name__ == "__main__":
    main()
