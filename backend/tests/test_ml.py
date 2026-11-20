from pathlib import Path
from types import SimpleNamespace

from app.ml import (
    DATA_PATH,
    SHAP_ADDITIVITY_TOLERANCE,
    make_anomaly_prediction,
    make_explanation,
    make_prediction,
    make_zscore_prediction,
    train_system,
)


def _sample_reading() -> dict[str, object]:
    return {
        "product_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }


def test_training_uses_only_sensor_features_and_returns_holdout_metrics() -> None:
    system = train_system(DATA_PATH)

    assert system.dataset_summary["rows"] == 10_000
    assert system.dataset_summary["failure_cases"] > 0
    assert len(system.metrics) == 4
    assert {item["model"] for item in system.metrics} == {
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "XGBoost",
    }
    assert set(system.cross_validation_f1) == {
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "XGBoost",
    }
    assert system.dataset_summary["test_rows"] == 2_000
    assert all(0 <= item["f1"] <= 1 for item in system.metrics)
    assert system.selected_model_name in system.models


def test_prediction_and_anomaly_detection_return_bounded_scores() -> None:
    system = train_system(DATA_PATH)

    prediction = make_prediction(system, _sample_reading())
    anomaly = make_anomaly_prediction(system, _sample_reading())

    assert 0 <= prediction["failure_score"] <= 1
    assert prediction["priority"] in {"Low", "Moderate", "High", "Critical"}
    assert isinstance(anomaly["is_anomaly"], bool)
    assert isinstance(anomaly["anomaly_score"], float)


def test_explanation_adds_to_selected_model_prediction() -> None:
    system = train_system(DATA_PATH)
    reading = _sample_reading()

    prediction = make_prediction(system, reading)
    explanation = make_explanation(system, reading)
    explained_score = explanation["baseline_value"] + sum(
        item["shap_value"] for item in explanation["contributions"]
    )

    assert explanation["model"] == prediction["model"]
    assert explanation["failure_score"] == prediction["failure_score"]
    assert abs(explained_score - prediction["failure_score"]) <= (
        SHAP_ADDITIVITY_TOLERANCE
    )
    assert len(explanation["contributions"]) == 6
    assert all(
        item["direction"]
        == (
            "increases"
            if item["shap_value"] > 0
            else "decreases"
            if item["shap_value"] < 0
            else "no change"
        )
        for item in explanation["contributions"]
    )


def test_zscore_normal_and_clearly_anomalous_observations() -> None:
    system = train_system(DATA_PATH)

    normal = make_zscore_prediction(system, _sample_reading())
    assert normal["threshold"] == 3.0
    assert isinstance(normal["is_anomaly"], bool)
    assert 0 <= normal["max_absolute_zscore"]
    assert set(normal["feature_zscores"]) == {
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    }

    anomalous_reading = _sample_reading()
    anomalous_reading["torque"] = 1000.0
    anomalous = make_zscore_prediction(system, anomalous_reading)
    assert anomalous["is_anomaly"]
    assert "Torque" in anomalous["exceeded_features"]


def test_zscore_individual_calculation_and_threshold_boundary() -> None:
    system = SimpleNamespace(
        zscore_baseline={
            "Air temperature [K]": {"mean": 300.0, "std": 10.0},
            "Process temperature [K]": {"mean": 310.0, "std": 10.0},
            "Rotational speed [rpm]": {"mean": 1500.0, "std": 100.0},
            "Torque [Nm]": {"mean": 40.0, "std": 10.0},
            "Tool wear [min]": {"mean": 100.0, "std": 10.0},
        }
    )
    reading = _sample_reading()
    reading["air_temperature"] = 330.0
    reading["torque"] = 70.0

    result = make_zscore_prediction(system, reading)

    assert result["feature_zscores"]["Air temperature"] == 3.0
    assert result["feature_zscores"]["Torque"] == 3.0
    assert "Air temperature" in result["exceeded_features"]
    assert "Torque" in result["exceeded_features"]
    assert result["is_anomaly"]


def test_zscore_zero_std_and_invalid_values() -> None:
    system = SimpleNamespace(
        zscore_baseline={
            "Air temperature [K]": {"mean": 300.0, "std": 0.0},
            "Process temperature [K]": {"mean": 310.0, "std": 10.0},
            "Rotational speed [rpm]": {"mean": 1500.0, "std": 100.0},
            "Torque [Nm]": {"mean": 40.0, "std": 10.0},
            "Tool wear [min]": {"mean": 100.0, "std": 10.0},
        }
    )
    reading = _sample_reading()

    reading["air_temperature"] = 300.0
    equal_result = make_zscore_prediction(system, reading)
    assert equal_result["feature_zscores"]["Air temperature"] == 0.0
    assert "Air temperature" not in equal_result["exceeded_features"]
    assert "Air temperature" in equal_result["zero_variance_features"]
    assert equal_result["is_anomaly"] is False
    assert equal_result["max_absolute_zscore"] == 0.0

    reading["air_temperature"] = 300.0000000005
    equal_tolerance_result = make_zscore_prediction(system, reading)
    assert equal_tolerance_result["feature_zscores"]["Air temperature"] == 0.0
    assert "Air temperature" not in equal_tolerance_result["exceeded_features"]

    reading["air_temperature"] = 301.0
    above_result = make_zscore_prediction(system, reading)
    assert above_result["feature_zscores"]["Air temperature"] == 3.0
    assert "Air temperature" in above_result["exceeded_features"]
    assert "Air temperature" in above_result["zero_variance_features"]
    assert above_result["is_anomaly"] is True
    assert above_result["max_absolute_zscore"] == 3.0

    reading["air_temperature"] = 299.0
    below_result = make_zscore_prediction(system, reading)
    assert below_result["feature_zscores"]["Air temperature"] == -3.0
    assert "Air temperature" in below_result["exceeded_features"]
    assert "Air temperature" in below_result["zero_variance_features"]
    assert below_result["is_anomaly"] is True
    assert below_result["max_absolute_zscore"] == 3.0

    reading["rotational_speed"] = float("nan")
    try:
        make_zscore_prediction(system, reading)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("Non-finite Z-score input should raise ValueError.")


def test_missing_dataset_is_reported() -> None:
    missing_path = Path(DATA_PATH).with_name("missing-ai4i.csv")

    try:
        train_system(missing_path)
    except FileNotFoundError as error:
        assert "AI4I dataset was not found" in str(error)
    else:
        raise AssertionError("Training should report a missing dataset.")
