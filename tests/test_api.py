"""API smoke, privacy, and validation tests using explicit synthetic data."""
import os
from pathlib import Path

os.environ["DEMO_MODE"] = "true"
os.environ["CORS_ORIGINS"] = "http://localhost:5173"
from fastapi.testclient import TestClient
from backend.main import app
from backend.models import AssessmentInput

def test_health_and_overview():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "synthetic_demo"
        overview = client.get("/api/overview")
        assert overview.status_code == 200
        data = overview.json()
        assert data["mode"] == "synthetic_demo"
        assert data["sample_size"] == 520
        assert sum(entry["count"] for entry in data["distribution"]) == 520
        assert set(data) == {"mode", "sample_size", "average_social_density", "average_hours_alone", "high_loneliness_share", "distribution", "baseline"}
        assert all(set(row) == {"name", "count"} for row in data["distribution"])
        assert overview.headers["cache-control"] == "no-store"

def test_predict_private_no_persistence():
    log_path = Path(__file__).resolve().parents[1] / "backend" / "training_data.csv"
    assert not log_path.exists()
    payload = AssessmentInput.model_json_schema()["examples"][0]
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["risk_category"]
        assert len(data["user_id"]) == 12
        assert "Social Density" in data["user_scores"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
    assert not log_path.exists()

def test_reject_bad_input():
    payload = AssessmentInput.model_json_schema()["examples"][0]
    payload["cgpa"] = 100
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

def test_reject_oversized_body():
    with TestClient(app) as client:
        response = client.post("/predict", content=b"x"*9000, headers={"content-type":"application/json"})
        assert response.status_code == 413

def test_disallow_random_cors_origin():
    with TestClient(app) as client:
        response = client.options("/predict", headers={"origin":"https://attacker.example","access-control-request-method":"POST"})
        assert response.headers.get("access-control-allow-origin") != "https://attacker.example"

def test_private_dataset_preprocessing_drops_identifiers(tmp_path):
    """Exercise the normal CSV startup path with generated, never real, records."""
    from backend.ml_engine import MLEngine, COLUMN_MAP
    demo = MLEngine._demo_dataframe()
    reverse = {v: k for k, v in COLUMN_MAP.items()}
    raw = demo.rename(columns=reverse)
    raw["Name:"] = "synthetic-person"
    raw["Timestamp"] = "2026-01-01"
    raw.loc[0, reverse["assignments_due"]] = 100000.0
    csv = tmp_path / "authorized_test.csv"
    raw.to_csv(csv, index=False)
    engine = MLEngine(str(csv))
    assert engine.demo_mode is False
    assert len(engine.df) == 520
    assert not {"name", "timestamp", "university"}.intersection(engine.df.columns)
    assert engine.df["assignments_due"].max() <= 40
    assert engine.overview()["mode"] == "survey"
