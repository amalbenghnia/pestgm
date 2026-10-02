import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines import LightGBMBaseline


def _toy_capacity_factor_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    capacity = rng.choice([0.5, 2.0, 10.0, 30.0], size=n)
    ghi = rng.uniform(0, 1000, n)
    cf = np.clip(ghi / 1000 + rng.normal(0, 0.02, n), 0, 1)
    production = cf * capacity
    return pd.DataFrame({
        "capacity_mw": capacity,
        "shortwave_radiation": ghi,
        "direct_radiation": ghi * 0.7,
        "diffuse_radiation": ghi * 0.3,
        "temperature_2m": rng.uniform(15, 35, n),
        "relative_humidity_2m": rng.uniform(20, 80, n),
        "cloud_cover": rng.uniform(0, 100, n),
        "wind_speed_10m": rng.uniform(0, 10, n),
        "solar_elevation_deg": rng.uniform(0, 80, n),
        "solar_azimuth_deg": rng.uniform(0, 360, n),
        "clearsky_ghi_wm2": ghi * 1.05,
        "daylight_flag": (ghi > 0).astype(int),
        "hour_sin": rng.uniform(-1, 1, n), "hour_cos": rng.uniform(-1, 1, n),
        "doy_sin": rng.uniform(-1, 1, n), "doy_cos": rng.uniform(-1, 1, n),
        "day_of_week": rng.integers(0, 7, n), "month": rng.integers(1, 13, n),
        "pv_production_mw": production,
        "capacity_factor": cf,
    })


def test_capacity_normalized_model_predicts_bounded_factor():
    """A model trained on capacity_factor (production/capacity, [0,1]) should
    predict values that, once rescaled by a NEW/unseen capacity, stay
    physically sane - this is the mechanism behind the ENSTAB capacity-
    normalization fix (scripts/09_enstab_external_validation.py)."""
    df = _toy_capacity_factor_df()
    train, test = df.iloc[:200], df.iloc[200:]

    model = LightGBMBaseline()
    model.feature_cols = [c for c in model.feature_cols if c != "capacity_mw"]
    model.fit(train, "capacity_factor")
    pred_cf = model.predict(test)

    # predicted capacity factor should be roughly bounded near [0,1] (small
    # overshoot from the regressor is fine before the downstream physical clip)
    assert pred_cf.min() >= -0.2
    assert pred_cf.max() <= 1.2

    # rescale by an UNSEEN small capacity (0.1 MW, never in training) and
    # confirm the scaling mechanism itself is sound
    unseen_capacity = 0.1
    pred_mw = pred_cf * unseen_capacity
    assert (pred_mw.dropna() <= unseen_capacity * 1.3).all()


def test_pvgis_reference_script_stops_on_first_failure(monkeypatch):
    """attempt_pvgis_for_all_districts must stop at the first unreachable
    district rather than silently substituting anything for the rest."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib
    mod = importlib.import_module("10_build_pv_reference_dataset")

    class FailingProvider:
        def get_reference_series(self, *a, **k):
            raise RuntimeError("PVGIS request failed: 403 Forbidden")

    monkeypatch.setattr(mod, "PVGISProvider", lambda: FailingProvider())

    districts = pd.DataFrame({
        "district_id": ["D01", "D02", "D03"],
        "district": ["A", "B", "C"],
        "governorate": ["G1", "G1", "G2"],
        "latitude": [36.8, 35.8, 34.8],
        "longitude": [10.1, 10.6, 10.7],
    })
    frames, status = mod.attempt_pvgis_for_all_districts(districts, 2023, 2023)
    assert frames == []
    assert status["attempted"] == 1  # stopped immediately, did not try D02/D03
    assert status["succeeded"] == 0
    assert status["first_failure"]["district_id"] == "D01"
