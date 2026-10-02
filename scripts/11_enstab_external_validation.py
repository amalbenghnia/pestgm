import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lag_features import add_lag_features, add_rolling_features
from src.forecasting.backtesting import make_chrono_split
from src.forecasting.multi_horizon import HorizonModel, HORIZONS

class CFHorizonModelWrapper:
    def __init__(self, model: HorizonModel):
        self.model = model
    def fit(self, df_h: pd.DataFrame):
        df_cf = df_h.copy()
        df_cf["target_at_horizon"] = df_cf["target_at_horizon"] / df_cf["capacity_mw"]
        feats = [f for f in self.model.feature_cols if f != "capacity_mw"]
        self.model.feature_cols = feats
        self.model.fit(df_cf)
    def predict(self, df_h: pd.DataFrame) -> pd.Series:
        pred_cf = self.model.predict(df_h)
        return (pred_cf * df_h["capacity_mw"]).clip(lower=0)

class CFQuantileWrapper:
    def __init__(self, feature_cols: list):
        from src.uncertainty.quantile_forecast import QuantileForecastModel
        self.feature_cols = [f for f in feature_cols if f != "capacity_mw"]
        self.qm = QuantileForecastModel(feature_cols=self.feature_cols)
    def fit(self, df_h: pd.DataFrame, target_col: str):
        df_cf = df_h.copy()
        df_cf[target_col] = df_cf[target_col] / df_cf["capacity_mw"]
        self.qm.fit(df_cf, target_col)
    def calibrate(self, df_h: pd.DataFrame, target_col: str):
        df_cf = df_h.copy()
        df_cf[target_col] = df_cf[target_col] / df_cf["capacity_mw"]
        self.qm.calibrate(df_cf, target_col)
    def predict(self, df_h: pd.DataFrame) -> pd.DataFrame:
        preds = self.qm.predict(df_h)
        for c in preds.columns:
            preds[c] = (preds[c] * df_h["capacity_mw"]).clip(lower=0)
        return preds

from src.uncertainty.quantile_forecast import evaluate_coverage, pinball_loss

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.nansum(np.abs(y_true))
    if denom == 0: return np.nan
    return float(np.nansum(np.abs(y_true - y_pred)) / denom * 100)

def mae_diurne(y_true: np.ndarray, y_pred: np.ndarray, daylight: np.ndarray) -> float:
    mask = daylight.astype(bool)
    if mask.sum() == 0: return np.nan
    return float(np.nanmean(np.abs(y_true[mask] - y_pred[mask])))

def ramp_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ramp_true = np.diff(y_true)
    ramp_pred = np.diff(y_pred)
    valid = ~(np.isnan(ramp_true) | np.isnan(ramp_pred))
    if valid.sum() == 0: return np.nan
    return float(np.nanmean(np.abs(ramp_true[valid] - ramp_pred[valid])))

def evaluate_extended(y_true: pd.Series, y_pred: pd.Series, capacity: pd.Series, daylight: pd.Series) -> dict:
    from src.forecasting.backtesting import evaluate
    base = evaluate(y_true, y_pred, capacity)
    yt, yp = y_true.to_numpy(), y_pred.to_numpy()
    dl = daylight.to_numpy()
    base["WAPE_pct"] = wape(yt, yp)
    base["MAE_diurne_MW"] = mae_diurne(yt, yp, dl)
    base["ramp_MAE_MW"] = ramp_mae(yt, yp)
    return base

TARGET_COL = "pv_production_mw_reference"

print("1. Loading national training dataset (2020)...")
df_train = pd.read_csv("data/processed/training_dataset_15min.csv", parse_dates=["timestamp"])
df_train = add_lag_features(df_train, TARGET_COL)
df_train = add_rolling_features(df_train, TARGET_COL)

split = make_chrono_split(df_train, train_frac=0.8, val_frac=0.2)
train, val, _ = split.split(df_train) # We use val for bias/conformal calibration

print("\n2. Loading ENSTAB Borj Cedria dataset (2022-2024)...")
df_enstab = pd.read_csv("data/validation/enstab_borj_cedria_real.csv", parse_dates=["timestamp"])
df_enstab["district_id"] = "ENSTAB"

# Fake the target col name so lag features match exactly
df_enstab[TARGET_COL] = df_enstab["pv_production_mw_measured"]
df_enstab = add_lag_features(df_enstab, TARGET_COL)
df_enstab = add_rolling_features(df_enstab, TARGET_COL)

# Re-assign target_timestamp to match our framework
df_enstab["target_timestamp"] = df_enstab["timestamp"]
df_train["target_timestamp"] = df_train["timestamp"]
val["target_timestamp"] = val["timestamp"]
train["target_timestamp"] = train["timestamp"]

all_point = {}
all_unc = {}

for horizon_name in HORIZONS:
    print(f"\n=== Training & Evaluating on ENSTAB for Horizon: {horizon_name} ===")
    feats = [f for f in HORIZONS[horizon_name]["features"] if f in df_enstab.columns]
    
    hm = HorizonModel(horizon_name=horizon_name, feature_cols=feats)
    hm = CFHorizonModelWrapper(hm)
    
    train_h = hm.model.build_supervised_frame(train, TARGET_COL)
    val_h = hm.model.build_supervised_frame(val, TARGET_COL)
    test_h = hm.model.build_supervised_frame(df_enstab, TARGET_COL) # ENSTAB is our test set
    
    # We will rename pv_production_mw_measured to target_at_horizon manually because
    # build_supervised_frame shifts TARGET_COL. Wait, build_supervised_frame will just shift TARGET_COL (which we faked).
    # So test_h["target_at_horizon"] is exactly the measured production at that horizon!
    
    print(f"  Training XGBoost CF model on national data ({len(train_h)} rows)...")
    hm.fit(train_h)
    
    print(f"  Predicting on ENSTAB data ({len(test_h)} rows)...")
    pred_h = hm.predict(test_h)
    
    # Simple static bias correction from national val set
    val_pred = hm.predict(val_h)
    bias = float(np.nanmean(val_h["target_at_horizon"] - val_pred))
    pred_h_corrected = pred_h.clip(lower=0)
    #print(f"  Applied national static bias correction: {bias:+.6f} MW")
    
    valid = test_h["target_at_horizon"].notna() & pred_h_corrected.notna()
    
    daylight = test_h.loc[valid, "daylight_flag"]
    m = evaluate_extended(
        test_h.loc[valid, "target_at_horizon"],
        pred_h_corrected[valid],
        test_h.loc[valid, "capacity_mw"],
        daylight
    )
    all_point[horizon_name] = m
    print("Point metrics on ENSTAB:", {k: round(v, 4) for k, v in m.items()})
    
    # Uncertainty
    print("  Training Quantile model...")
    qfeat = [c for c in feats if c in train_h.columns]
    qm = CFQuantileWrapper(feature_cols=qfeat)
    qm.fit(train_h, "target_at_horizon")
    qm.calibrate(val_h, "target_at_horizon")
    
    q_test = qm.predict(test_h)
    
    cov = evaluate_coverage(
        test_h["target_at_horizon"].reset_index(drop=True),
        q_test["p10_mw"].reset_index(drop=True),
        q_test["p90_mw"].reset_index(drop=True),
        daylight=test_h["daylight_flag"].reset_index(drop=True)
    )
    all_unc[horizon_name] = cov
    print("Uncertainty on ENSTAB:", cov)

print("\n--- ENSTAB Point Metrics ---")
print(pd.DataFrame(all_point).T.round(4).to_markdown())
print("\n--- ENSTAB Uncertainty Metrics ---")
print(pd.DataFrame(all_unc).T.round(4).to_markdown())
