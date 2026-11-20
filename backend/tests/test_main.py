from fastapi.testclient import TestClient
import pytest
from pathlib import Path

from app.main import app
from app.database import configure_database, dispose_database, save_assessment


client = TestClient(app)

SAMPLE_READING = {
    "product_type": "M",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 100.0,
}


def test_explanation_endpoint_matches_selected_model_prediction() -> None:
    response = client.post("/api/explain", json=SAMPLE_READING)

    assert response.status_code == 200
    explanation = response.json()
    assert explanation["model"]
    assert explanation["failure_score"] == client.post(
        "/api/predict", json=SAMPLE_READING
    ).json()["failure_score"]
    assert explanation["baseline_value"] + sum(
        item["shap_value"] for item in explanation["contributions"]
    ) == pytest.approx(explanation["failure_score"], abs=1e-6, rel=0)
    assert "Positive SHAP values increase" in explanation["interpretation"]


def test_existing_prediction_and_anomaly_endpoints_still_work() -> None:
    prediction = client.post("/api/predict", json=SAMPLE_READING)
    anomaly = client.post("/api/anomaly", json=SAMPLE_READING)

    assert prediction.status_code == 200
    assert {"model", "failure_score", "predicted_failure", "priority"} <= set(
        prediction.json()
    )
    assert anomaly.status_code == 200
    assert {"is_anomaly", "anomaly_score", "threshold_assumption"} <= set(
        anomaly.json()
    )


def test_zscore_endpoint_and_invalid_input() -> None:
    response = client.post("/api/zscore", json=SAMPLE_READING)

    assert response.status_code == 200
    payload = response.json()
    assert {"is_anomaly", "feature_zscores", "exceeded_features", "threshold", "max_absolute_zscore"} <= set(payload)
    assert payload["threshold"] == 3.0
    assert len(payload["feature_zscores"]) == 5

    anomalous = dict(SAMPLE_READING)
    anomalous["torque"] = 1000.0
    anomalous_response = client.post("/api/zscore", json=anomalous)
    assert anomalous_response.status_code == 200
    assert "Torque" in anomalous_response.json()["exceeded_features"]

    invalid = dict(SAMPLE_READING)
    invalid["air_temperature"] = "not-a-number"
    invalid_response = client.post("/api/zscore", json=invalid)
    assert invalid_response.status_code == 422


def test_assessment_history_detail_asset_and_pagination_with_sqlite() -> None:
    dispose_database()
    test_db = Path(__file__).resolve().parent / "phase5-test.db"
    if test_db.exists():
        test_db.unlink()
    assert configure_database(f"sqlite+pysqlite:///{test_db.as_posix()}")

    first = save_assessment(
        source="manual",
        asset_id="asset-one",
        kind="machine_assessment",
        reading={"torque": 40.0},
        prediction={"failure_score": 0.2},
        anomaly={"is_anomaly": False},
        zscore={"is_anomaly": False},
        simulated=False,
    )
    second = save_assessment(
        source="manual",
        asset_id="asset-two",
        kind="machine_assessment",
        reading={"torque": 50.0},
        prediction={"failure_score": 0.3},
        anomaly={"is_anomaly": True},
        zscore={"is_anomaly": True},
        simulated=False,
    )
    assert first is not None
    assert second is not None

    history = client.get("/api/assessments?limit=1&offset=0")
    assert history.status_code == 200
    payload = history.json()
    assert payload["database_connected"] is True
    assert payload["total"] == 2
    assert len(payload["items"]) == 1

    asset_history = client.get("/api/assessments?limit=50&asset_id=asset-one")
    assert asset_history.status_code == 200
    assert all(
        item["asset_id"] == "asset-one"
        for item in asset_history.json()["items"]
    )

    detail = client.get(f"/api/assessments/{first['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == first["id"]
    assert detail.json()["zscore"]["is_anomaly"] is False

    missing = client.get("/api/assessments/999999")
    assert missing.status_code == 404

    dispose_database()
    if test_db.exists():
        test_db.unlink()


def test_assessment_history_unavailable_is_graceful() -> None:
    dispose_database()
    assert configure_database("postgresql+psycopg://invalid:invalid@127.0.0.1:1/invalid") is False

    response = client.get("/api/assessments?limit=50")
    assert response.status_code == 200
    assert response.json()["database_connected"] is False
    assert response.json()["items"] == []

    detail = client.get("/api/assessments/1")
    assert detail.status_code == 503
    dispose_database()


def test_assessment_history_date_range_and_validation() -> None:
    dispose_database()
    test_db = Path(__file__).resolve().parent / "phase5-date-test.db"
    if test_db.exists():
        test_db.unlink()
    assert configure_database(f"sqlite+pysqlite:///{test_db.as_posix()}")

    first = save_assessment(
        source="manual",
        asset_id="date-asset",
        kind="machine_assessment",
        reading={"torque": 40.0},
    )
    assert first is not None

    valid = client.get(
        "/api/assessments",
        params={
            "start": "2000-01-01T00:00:00Z",
            "end": "2100-01-01T00:00:00Z",
            "asset_id": "date-asset",
        },
    )
    assert valid.status_code == 200
    assert valid.json()["database_connected"] is True
    assert valid.json()["total"] == 1

    invalid = client.get("/api/assessments", params={"start": "not-a-date"})
    assert invalid.status_code == 422

    reversed_range = client.get(
        "/api/assessments",
        params={
            "start": "2100-01-01T00:00:00Z",
            "end": "2000-01-01T00:00:00Z",
        },
    )
    assert reversed_range.status_code == 422


def test_analytics_endpoint_empty_and_filtered_sqlite() -> None:
    dispose_database()
    test_db = Path(__file__).resolve().parent / "phase7-analytics-test.db"
    if test_db.exists():
        test_db.unlink()
    assert configure_database(f"sqlite+pysqlite:///{test_db.as_posix()}")

    empty = client.get("/api/analytics")
    assert empty.status_code == 200
    assert empty.json()["database_connected"] is True
    assert empty.json()["analytics"]["total_assessments"] == 0

    save_assessment(
        source="manual",
        asset_id="analytics-a",
        kind="machine_assessment",
        reading={"torque": 40.0},
        prediction={"predicted_failure": True},
        anomaly={"is_anomaly": True},
        zscore={"is_anomaly": False},
        recommendation={"severity": "critical"},
        simulated=False,
    )
    save_assessment(
        source="simulator",
        asset_id="analytics-b",
        kind="machine_assessment",
        reading={"torque": 20.0},
        prediction={"predicted_failure": False},
        anomaly={"is_anomaly": False},
        zscore={"is_anomaly": True},
        recommendation={"severity": "warning"},
        simulated=True,
    )

    aggregate = client.get("/api/analytics")
    assert aggregate.status_code == 200
    analytics = aggregate.json()["analytics"]
    assert analytics["total_assessments"] == 2
    assert analytics["predicted_failures"] == 1
    assert analytics["normal_assessments"] == 1
    assert analytics["anomaly_assessments"] == 2
    assert analytics["urgent_critical"] == 1
    assert analytics["monitored_assets"] == 2
    assert analytics["simulated_assessments"] == 1

    filtered = client.get("/api/analytics", params={"asset_id": "analytics-a"})
    assert filtered.status_code == 200
    assert filtered.json()["analytics"]["total_assessments"] == 1
    assert filtered.json()["analytics"]["monitored_assets"] == 1

    invalid = client.get("/api/analytics", params={"start": "bad-date"})
    assert invalid.status_code == 422

    dispose_database()
    if test_db.exists():
        test_db.unlink()


def test_analytics_endpoint_database_unavailable() -> None:
    dispose_database()
    assert configure_database("postgresql+psycopg://invalid:invalid@127.0.0.1:1/invalid") is False

    response = client.get("/api/analytics")
    assert response.status_code == 200
    assert response.json()["database_connected"] is False
    assert response.json()["analytics"] is None
    dispose_database()


def test_simulation_generate_history_reset_and_invalid_mode() -> None:
    reset = client.post("/api/simulation/reset?seed=42")
    assert reset.status_code == 200
    assert reset.json()["reset"] is True

    generated = client.post("/api/simulation/generate?mode=normal")
    assert generated.status_code == 200
    payload = generated.json()
    assert payload["simulated"] is True
    assert payload["simulation_mode"] == "normal"
    assert {"prediction", "anomaly", "zscore", "reading"} <= set(payload)

    history = client.get("/api/simulation/history?limit=20")
    assert history.status_code == 200
    assert history.json()["simulated"] is True
    assert history.json()["count"] >= 1

    latest = client.get("/api/simulation/latest")
    assert latest.status_code == 200
    assert latest.json()["available"] is True

    reset_again = client.post("/api/simulation/reset?seed=7")
    assert reset_again.status_code == 200
    assert client.get("/api/simulation/history").json()["count"] == 0

    invalid = client.post("/api/simulation/generate?mode=invalid")
    assert invalid.status_code == 422
