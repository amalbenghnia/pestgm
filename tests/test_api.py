import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_districts_returns_50():
    r = client.get("/districts")
    assert r.status_code == 200
    assert len(r.json()) == 50


def test_forecast_district_labelled_demo():
    r = client.get("/forecast/district/D01")
    assert r.status_code == 200
    body = r.json()
    assert len(body) > 0
    assert all(p["forecast_status"] == "DEMO_PROXY_PHYSICS_SIMULATION" for p in body)


def test_forecast_unknown_district_404():
    r = client.get("/forecast/district/NOPE")
    assert r.status_code == 404


def test_forecast_national_reconciled():
    r = client.get("/forecast/national")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_post_forecast_dispatches():
    r = client.post("/forecast", json={"spatial_level": "national", "entity": "TUNISIA"})
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_post_forecast_invalid_level():
    r = client.post("/forecast", json={"spatial_level": "planet", "entity": "TUNISIA"})
    assert r.status_code == 400


def test_uncertainty_endpoint():
    r = client.get("/uncertainty", params={"district_id": "D01"})
    assert r.status_code == 200
    body = r.json()
    assert len(body) > 0
    assert all(row["p10_mw"] <= row["p90_mw"] + 1e-9 for row in body)


def test_errors_endpoint():
    r = client.get("/errors", params={"district_id": "D01"})
    assert r.status_code == 200
    assert "metrics" in r.json()


def test_anomalies_endpoint():
    r = client.get("/anomalies", params={"district_id": "D01"})
    assert r.status_code == 200
    assert "pct_flagged" in r.json()


def test_drift_weather_endpoint():
    r = client.get("/drift/weather")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_export_csv():
    r = client.get("/export", params={"spatial_level": "national", "fmt": "csv"})
    assert r.status_code == 200
    assert "forecast_mw" in r.text
