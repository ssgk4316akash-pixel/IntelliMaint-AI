import pytest

from app.recommendations import build_recommendation


READING = {
    "product_type": "M",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 100.0,
}


def _prediction(priority="Low", predicted=False, score=0.1):
    return {
        "model": "Test Model",
        "failure_score": score,
        "predicted_failure": predicted,
        "priority": priority,
    }


def _anomaly(flag=False):
    return {
        "is_anomaly": flag,
        "anomaly_score": 0.1,
        "threshold_assumption": 0.03,
    }


def _zscore(flag=False, features=None, maximum=0.5):
    return {
        "is_anomaly": flag,
        "feature_zscores": {"Torque": maximum},
        "exceeded_features": features or [],
        "threshold": 3.0,
        "max_absolute_zscore": maximum,
    }


def test_normal_assessment_returns_routine_recommendation() -> None:
    result = build_recommendation(
        READING,
        _prediction(),
        _anomaly(),
        _zscore(),
    )

    assert result["severity"] == "routine"
    assert result["actions"][0]["code"] == "routine"
    assert result["reasons"]


def test_failure_prediction_without_anomaly_is_urgent() -> None:
    result = build_recommendation(
        READING,
        _prediction(priority="Critical", predicted=True, score=0.9),
        _anomaly(),
        _zscore(),
    )

    assert result["severity"] == "urgent"
    assert any(reason["code"] == "model_score" for reason in result["reasons"])


def test_single_anomaly_is_warning() -> None:
    result = build_recommendation(
        READING,
        _prediction(),
        _anomaly(flag=True),
        _zscore(),
    )

    assert result["severity"] == "warning"
    assert any(reason["code"] == "isolation_forest" for reason in result["reasons"])


def test_both_anomaly_detectors_is_urgent() -> None:
    result = build_recommendation(
        READING,
        _prediction(),
        _anomaly(flag=True),
        _zscore(flag=True, features=["Torque"], maximum=4.0),
    )

    assert result["severity"] == "urgent"
    assert any(action["code"] == "zscore" for action in result["actions"])
    assert any(reason["code"] == "zscore" for reason in result["reasons"])


def test_combined_failure_and_anomaly_is_critical() -> None:
    result = build_recommendation(
        READING,
        _prediction(priority="Critical", predicted=True, score=0.95),
        _anomaly(flag=True),
        _zscore(flag=True, features=["Torque"], maximum=4.0),
    )

    assert result["severity"] == "critical"
    assert len(result["actions"]) >= 3
    assert len(result["reasons"]) >= 3


def test_invalid_or_nonfinite_recommendation_input_is_rejected() -> None:
    invalid = dict(READING)
    invalid["torque"] = float("nan")

    with pytest.raises(ValueError):
        build_recommendation(
            invalid,
            _prediction(),
            _anomaly(),
            _zscore(),
        )
