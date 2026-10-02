"""
Probabilistic forecasting: quantile LightGBM + split-conformal calibration
(CDC continuation section 15).

Implemented using Conformalized Quantile Regression (CQR, Romano et al. 2019).
This ensures that the predicted intervals (e.g. P10-P90 for 80% coverage)
are rigorously calibrated on the hold-out validation set to guarantee
marginal coverage, by applying a single nonconformity score to the interval
bounds rather than independently shifting each quantile.

Design:
  1. Train 5 independent LightGBM quantile regressors (objective="quantile")
     for tau in {0.10, 0.25, 0.50, 0.75, 0.90}.
  2. Enforce monotonic ordering P10<=P25<=P50<=P75<=P90 by sorting each row's
     raw quantile predictions (a standard, simple non-crossing fix).
  3. Split-conformal calibration (CQR): for each nested interval
     ([P10, P90] and [P25, P75]), compute the nonconformity score
     E_i = max(q_low - y, y - q_high) on the calibration set. Shift the
     bounds symmetrically by the empirical quantile of E to guarantee coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_COL_NAMES = {0.10: "p10_mw", 0.25: "p25_mw", 0.50: "p50_mw", 0.75: "p75_mw", 0.90: "p90_mw"}


@dataclass
class QuantileForecastModel:
    feature_cols: list
    params_template: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    ))
    models: dict = field(default_factory=dict)         # tau -> fitted LightGBM
    conformal_offsets: dict = field(default_factory=dict)  # tau -> additive offset from calibration

    def fit(self, train_df: pd.DataFrame, target_col: str):
        import lightgbm as lgb
        X = train_df[self.feature_cols]
        y = train_df[target_col]
        valid = y.notna() & X.notna().all(axis=1)
        for tau in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **self.params_template)
            m.fit(X[valid], y[valid])
            self.models[tau] = m
        return self

    def _raw_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols]
        valid = X.notna().all(axis=1)
        out = pd.DataFrame(index=df.index)
        for tau in QUANTILES:
            col = QUANTILE_COL_NAMES[tau]
            pred = np.full(len(df), np.nan)
            pred[valid.to_numpy()] = self.models[tau].predict(X[valid])
            out[col] = pred
        # enforce monotonic ordering P10<=P25<=P50<=P75<=P90 row-wise
        cols = [QUANTILE_COL_NAMES[t] for t in QUANTILES]
        out[cols] = np.sort(out[cols].to_numpy(), axis=1)
        return out.clip(lower=0)

    def calibrate(self, calib_df: pd.DataFrame, target_col: str):
        """Conformalized Quantile Regression (CQR, Romano et al. 2019).
        Calibrates the nested intervals [P10, P90] and [P25, P75] to achieve
        valid marginal coverage, and centers P50."""
        preds = self._raw_predict(calib_df)
        y = calib_df[target_col].to_numpy()
        valid = ~np.isnan(y)
        y_valid = y[valid]
        n = valid.sum()

        # 1. P50 (median) - simple additive bias correction
        resid_50 = y_valid - preds[QUANTILE_COL_NAMES[0.50]].to_numpy()[valid]
        self.conformal_offsets[0.50] = float(np.median(resid_50))

        # 2. Intervals: (tau_low, tau_high, target_coverage)
        intervals = [
            (0.10, 0.90, 0.80),
            (0.25, 0.75, 0.50),
        ]
        
        # We will store a scaling factor 'q_val' (multiplicative) instead of additive offset
        # for adaptive conformal prediction.
        if not hasattr(self, "conformal_scales"):
            self.conformal_scales = {}

        epsilon = 1e-6  # prevent division by zero
        for tau_low, tau_high, coverage in intervals:
            q_low = preds[QUANTILE_COL_NAMES[tau_low]].to_numpy()[valid]
            q_high = preds[QUANTILE_COL_NAMES[tau_high]].to_numpy()[valid]
            
            # Locally Adaptive CQR nonconformity score: 
            # E_i = max(q_low - y, y - q_high) / (q_high - q_low + epsilon)
            width = q_high - q_low + epsilon
            scores = np.maximum(q_low - y_valid, y_valid - q_high) / width
            
            # Find the empirical quantile of the scores
            q_level = min(coverage * (1 + 1.0 / n), 1.0)
            q_val = float(np.quantile(scores, q_level))
            
            self.conformal_scales[(tau_low, tau_high)] = q_val
            
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        preds = self._raw_predict(df)
        
        # apply P50 shift
        preds[QUANTILE_COL_NAMES[0.50]] = (
            preds[QUANTILE_COL_NAMES[0.50]] + self.conformal_offsets.get(0.50, 0.0)
        ).clip(lower=0)
        
        epsilon = 1e-6
        # apply locally adaptive interval scaling
        if hasattr(self, "conformal_scales"):
            for (tau_low, tau_high), q_val in self.conformal_scales.items():
                col_low = QUANTILE_COL_NAMES[tau_low]
                col_high = QUANTILE_COL_NAMES[tau_high]
                width = preds[col_high] - preds[col_low] + epsilon
                
                preds[col_low] = (preds[col_low] - q_val * width).clip(lower=0)
                preds[col_high] = (preds[col_high] + q_val * width).clip(lower=0)
                
        cols = [QUANTILE_COL_NAMES[t] for t in QUANTILES]
        preds[cols] = np.sort(preds[cols].to_numpy(), axis=1)  # re-sort after calibration
        return preds


def evaluate_coverage(y_true: pd.Series, p10: pd.Series, p90: pd.Series, daylight: pd.Series = None) -> dict:
    """80% interval (P10-P90) coverage + average width - CDC section 15/17."""
    covered = (y_true >= p10) & (y_true <= p90)
    res = {
        "nominal_coverage_pct": 80.0,
        "overall_coverage_pct": float(covered.mean() * 100),
        "mean_interval_width_mw": float((p90 - p10).mean()),
    }
    if daylight is not None:
        mask = daylight.astype(bool)
        if mask.sum() > 0:
            res["daytime_coverage_pct"] = float(covered[mask].mean() * 100)
    return res


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.nanmean(np.maximum(tau * diff, (tau - 1) * diff)))
