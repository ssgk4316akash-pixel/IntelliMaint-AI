from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.main import app
from app.rul import (
    RUL_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    _read_official_rul,
    construct_test_observation_targets,
    construct_training_targets,
    get_last_test_rows,
    nasa_scoring_function,
    select_validation_snapshot,
    train_rul_system,
)
from sklearn.model_selection import GroupKFold

import app.rul as rul_module


client = TestClient(app)


def test_training_targets_use_engine_maximum_and_125_cycle_cap() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 200, 10, 20],
        }
    )

    targets = construct_training_targets(frame)

    assert targets.tolist() == [125, 125, 0, 10, 0]


def test_official_test_labels_remain_uncapped() -> None:
    raw_labels = _read_official_rul(RUL_DATA_PATH)

    assert len(raw_labels) == 100
    assert raw_labels.max() > 125
    assert raw_labels.tolist()[0] == 112.0


def test_test_observation_targets_use_official_rul_and_last_cycle() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 2, 2],
            "time_in_cycles": [1, 4, 2, 5],
        }
    )

    targets = construct_test_observation_targets(frame, np.array([10.0, 20.0]))

    assert targets.tolist() == [13.0, 10.0, 23.0, 20.0]


def test_last_test_rows_select_one_maximum_cycle_per_engine() -> None:
    frame = pd.DataFrame(
        {
            "unit_number": [2, 1, 2, 1],
            "time_in_cycles": [2, 5, 8, 7],
            "sensor_1": [20.0, 10.0, 21.0, 11.0],
        }
    )

    last_rows = get_last_test_rows(frame)

    assert last_rows["unit_number"].tolist() == [1, 2]
    assert last_rows["time_in_cycles"].tolist() == [7, 8]


def test_validation_engine_split_cutoff_and_one_prediction_per_engine() -> None:
    frame = pd.read_csv(
        TRAIN_DATA_PATH,
        sep=r"\s+",
        header=None,
        names=[
            "unit_number",
            "time_in_cycles",
            "operational_setting_1",
            "operational_setting_2",
            "operational_setting_3",
            *[f"sensor_{index}" for index in range(1, 22)],
        ],
    )
    rng = np.random.default_rng(42)
    cv = GroupKFold(n_splits=5)
    total_validation_predictions = 0

    for train_indices, validation_indices in cv.split(
        frame,
        groups=frame["unit_number"],
    ):
        train_ids = set(frame.iloc[train_indices]["unit_number"].unique())
        validation_ids = set(frame.iloc[validation_indices]["unit_number"].unique())
        assert train_ids.isdisjoint(validation_ids)

        validation_frame = frame.iloc[validation_indices]
        for engine_id in sorted(validation_ids):
            snapshot, actual_rul, cutoff, final_cycle = select_validation_snapshot(
                validation_frame,
                int(engine_id),
                rng,
            )
            assert snapshot.shape[0] == 24
            assert 1 <= cutoff < final_cycle
            assert 0 <= actual_rul <= 125
            total_validation_predictions += 1

    assert total_validation_predictions == 100


def test_nasa_scoring_function_matches_asymmetric_formula() -> None:
    assert nasa_scoring_function(np.array([50.0]), np.array([50.0])) == 0.0

    underestimation = nasa_scoring_function(
        np.array([13.0]),
        np.array([0.0]),
    )
    assert np.isclose(
        underestimation,
        np.expm1(1.0),
    )

    overestimation = nasa_scoring_function(
        np.array([0.0]),
        np.array([10.0]),
    )
    assert np.isclose(
        overestimation,
        np.expm1(1.0),
    )

    unequal = nasa_scoring_function(
        np.array([13.0, 50.0]),
        np.array([0.0, 70.0]),
    )
    expected = np.expm1(1.0) + np.expm1(2.0)
    assert np.isclose(unequal, expected)


def test_nasa_scoring_rejects_invalid_and_extreme_inputs() -> None:
    try:
        nasa_scoring_function(np.array([1.0]), np.array([2.0, 3.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Shape mismatch should raise ValueError.")

    try:
        nasa_scoring_function(np.array([np.nan]), np.array([1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite values should raise ValueError.")

    try:
        nasa_scoring_function(np.array([0.0]), np.array([100000.0]))
    except OverflowError:
        pass
    else:
        raise AssertionError("Representability overflow should raise OverflowError.")


def test_nasa_scoring_rejects_finite_score_aggregate_overflow() -> None:
    individual_score = np.expm1(709.5)
    assert np.isfinite(individual_score)

    try:
        nasa_scoring_function(
            np.zeros(2, dtype=float),
            np.full(2, 7095.0, dtype=float),
        )
    except OverflowError as error:
        assert "aggregate" in str(error)
    else:
        raise AssertionError("Aggregate overflow should raise OverflowError.")


def test_fd001_training_reports_all_three_models_and_required_counts(monkeypatch) -> None:
    fit_counts: dict[str, int] = {}
    original_make_models = rul_module._make_models

    def make_counted_models(feature_columns: list[str]):
        models = original_make_models(feature_columns)
        for name, model in models.items():
            original_fit = model.fit

            def counted_fit(*args, _name=name, _fit=original_fit, **kwargs):
                fit_counts[_name] = fit_counts.get(_name, 0) + 1
                return _fit(*args, **kwargs)

            model.fit = counted_fit
        return models

    monkeypatch.setattr(rul_module, "_make_models", make_counted_models)
    system = train_rul_system(TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH)

    assert fit_counts == {
        "Linear Regression": 6,
        "Random Forest": 6,
        "XGBoost": 6,
    }
    expected_metrics = {
        "Linear Regression": (22.964, 18.014, 1326.988, 21.678, 17.435, 1292.231),
        "Random Forest": (19.881, 14.362, 1149.371, 17.928, 13.129, 869.977),
        "XGBoost": (20.776, 14.579, 1473.723, 18.661, 13.575, 1012.637),
    }
    validation = {item["model"]: item for item in system.validation_metrics}
    test = {item["model"]: item for item in system.test_metrics}
    for model_name, expected in expected_metrics.items():
        validation_row = validation[model_name]
        test_row = test[model_name]
        for key, value in zip(("rmse", "mae", "nasa_score"), expected[:3]):
            assert validation_row[key] == pytest.approx(value, abs=0.0005)
        for key, value in zip(("rmse", "mae", "nasa_score"), expected[3:]):
            assert test_row[key] == pytest.approx(value, abs=0.0005)
    assert system.dataset_summary["training_engines"] == 100
    assert system.dataset_summary["test_engines"] == 100
    assert system.dataset_summary["training_observations"] == 20631
    assert system.dataset_summary["test_observations"] == 13096
    assert {item["model"] for item in system.validation_metrics} == {
        "Linear Regression",
        "Random Forest",
        "XGBoost",
    }
    assert {item["model"] for item in system.test_metrics} == {
        "Linear Regression",
        "Random Forest",
        "XGBoost",
    }
    assert system.selected_model_name in system.models
    assert len(system.raw_official_test_rul) == 100
    assert system.raw_official_test_rul.max() > 125

    test_frame = pd.read_csv(
        TEST_DATA_PATH,
        sep=r"\s+",
        header=None,
        names=[
            "unit_number",
            "time_in_cycles",
            "operational_setting_1",
            "operational_setting_2",
            "operational_setting_3",
            *[f"sensor_{index}" for index in range(1, 22)],
        ],
    )
    last_rows = get_last_test_rows(test_frame)
    assert len(last_rows) == 100
    assert last_rows["unit_number"].nunique() == 100
    assert last_rows["unit_number"].tolist() == list(range(1, 101))

    for model in system.models.values():
        predictions = model.predict(last_rows[
            [
                "operational_setting_1",
                "operational_setting_2",
                "operational_setting_3",
                *[f"sensor_{index}" for index in range(1, 22)],
            ]
        ])
        assert len(predictions) == 100

    selected_predictions = np.clip(
        system.selected_model.predict(last_rows[
            [
                "operational_setting_1",
                "operational_setting_2",
                "operational_setting_3",
                *[f"sensor_{index}" for index in range(1, 22)],
            ]
        ]),
        0.0,
        125,
    )
    independent_rmse = float(
        np.sqrt(mean_squared_error(system.raw_official_test_rul, selected_predictions))
    )
    independent_mae = float(
        mean_absolute_error(system.raw_official_test_rul, selected_predictions)
    )
    independent_nasa = nasa_scoring_function(
        system.raw_official_test_rul,
        selected_predictions,
    )
    selected_test_metric = next(
        item
        for item in system.test_metrics
        if item["model"] == system.selected_model_name
    )
    assert np.isclose(selected_test_metric["rmse"], independent_rmse)
    assert np.isclose(selected_test_metric["mae"], independent_mae)
    assert np.isclose(selected_test_metric["nasa_score"], independent_nasa)


def test_rul_overview_and_prediction_endpoints() -> None:
    overview = client.get("/api/rul/overview")
    assert overview.status_code == 200
    payload = overview.json()
    assert payload["dataset"]["sensor_measurements"] == 21
    assert payload["dataset"]["operational_settings"] == 3
    assert len(payload["validation_metrics"]) == 3
    assert len(payload["test_metrics"]) == 3

    reading = {
        "operational_setting_1": -0.0007,
        "operational_setting_2": -0.0004,
        "operational_setting_3": 100.0,
        **{f"sensor_{index}": 1.0 for index in range(1, 22)},
    }
    prediction = client.post("/api/rul/predict", json=reading)
    assert prediction.status_code == 200
    result = prediction.json()
    assert 0 <= result["predicted_rul"] <= 125
    assert result["model"]
    assert result["units"] == "cycles"
    assert result["interpretation"]


def test_rul_prediction_rejects_invalid_reading() -> None:
    response = client.post("/api/rul/predict", json={"operational_setting_1": 0.0})

    assert response.status_code == 422
