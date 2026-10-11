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
        assert "name" not in str(data).lower()
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
