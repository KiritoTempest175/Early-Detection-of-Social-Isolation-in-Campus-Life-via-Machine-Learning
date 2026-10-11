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

def test_socially_active_anomaly_is_not_called_isolation():
    """A high-engagement synthetic profile must not be labeled a risk based solely on rarity."""
    profile = AssessmentInput.model_json_schema()["examples"][0]
    profile.update({
        "friend_group_size": 9, "meals_with_friends": 12,
        "social_outings": 5, "society_memberships": 3,
        "society_events_attended": 10, "common_room_frequency": 5,
        "hours_alone_weekly": 10, "screen_time": 3,
        "social_wellness_rating": 5, "loneliness_score": 1,
        "peer_satisfaction": 9, "mood_rating": 9,
        "workload_stress": 3
    })
    with TestClient(app) as client:
        response = client.post("/predict", json=profile)
    assert response.status_code == 200
    result = response.json()
    assert result["risk_category"] == "Socially Active"
    assert result["cluster_label"] == "Socially Active"
    assert result["anomaly_score"] is not None


def test_low_engagement_with_self_reported_distress_gets_context():
    profile = AssessmentInput.model_json_schema()["examples"][0]
    profile.update({
        "living_status": "Day Scholar", "friend_group_size": 0,
        "meals_with_friends": 0, "social_outings": 0,
        "society_memberships": 0, "society_events_attended": 0,
        "common_room_frequency": 1, "hours_alone_weekly": 80,
        "loneliness_score": 5, "workload_stress": 9,
        "mood_rating": 2, "peer_satisfaction": 2, "screen_time": 10,
    })
    with TestClient(app) as client:
        response = client.post("/predict", json=profile)
    assert response.status_code == 200
    assert response.json()["risk_category"] == "Elevated Signals"


def test_analytics_visualizes_all_synthetic_rows_without_identifiers():
    with TestClient(app) as client:
        response = client.get("/api/analytics")
    assert response.status_code == 200
    a = response.json()
    assert a["mode"] == "synthetic_demo"
    assert len(a["points"]) == a["sample_size"] == 520
    assert sum(row["count"] for row in a["density_histogram"]) == 520
    assert sum(row["count"] for row in a["density_wellbeing_heatmap"]) == 520
    assert sum(row["count"] for row in a["centroids"]) == 520
    assert len(a["correlation"]["labels"]) == 7
    assert len(a["correlation"]["values"]) == 7
    assert all(set(point) == {"id", "x", "y", "cluster"} for point in a["points"])
    assert all(0 <= point["x"] <= 100 and 0 <= point["y"] <= 100 for point in a["points"])


def test_real_survey_analytics_uses_aggregates_only(tmp_path):
    from backend.ml_engine import MLEngine, COLUMN_MAP
    demo = MLEngine._demo_dataframe()
    reverse = {v: k for k, v in COLUMN_MAP.items()}
    raw = demo.rename(columns=reverse)
    raw["Name:"] = ["PRIVATE-NAME-" + str(i) for i in range(len(raw))]
    csv = tmp_path / "test.csv"
    raw.to_csv(csv, index=False)
    a = MLEngine(str(csv)).analytics()
    assert a["mode"] == "survey_aggregates"
    assert a["points"] == []
    assert sum(cell["count"] for cell in a["density_wellbeing_heatmap"]) == 520
    assert "PRIVATE-NAME-" not in str(a)


def test_prediction_percentile_and_specific_improvement_ideas():
    example = AssessmentInput.model_json_schema()["examples"][0]
    example.update({"loneliness_score": 5, "workload_stress": 9, "sleep_hours": 4})
    with TestClient(app) as client:
        response = client.post("/predict", json=example)
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["density_percentile"] <= 100
    assert 1 <= len(result["insights"]) <= 3
    assert all(set(i) == {"title", "detail", "action"} for i in result["insights"])
