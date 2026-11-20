from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from app.recommendations import build_rul_recommendation


RUL_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nasa"
TRAIN_DATA_PATH = RUL_DATA_DIR / "train_FD001.txt"
TEST_DATA_PATH = RUL_DATA_DIR / "test_FD001.txt"
RUL_DATA_PATH = RUL_DATA_DIR / "RUL_FD001.txt"
RUL_CAP = 125
RANDOM_STATE = 42

COLUMN_NAMES = [
    "unit_number",
    "time_in_cycles",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    *[f"sensor_{index}" for index in range(1, 22)],
]
FEATURE_COLUMNS = COLUMN_NAMES[2:]


@dataclass
class RulSystem:
    models: dict[str, Pipeline]
    selected_model_name: str
    selected_model: Pipeline
    validation_metrics: list[dict[str, Any]]
    test_metrics: list[dict[str, Any]]
    dataset_summary: dict[str, Any]
    train_feature_columns: list[str]
    raw_official_test_rul: np.ndarray


def _read_fd001(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"FD001 dataset was not found at {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    if frame.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"FD001 dataset must contain {len(COLUMN_NAMES)} columns: {path}"
        )
    if frame.isna().any().any():
        raise ValueError(f"FD001 dataset contains missing values: {path}")
    if (frame["time_in_cycles"] <= 0).any():
        raise ValueError("FD001 cycle values must be positive.")
    return frame


def construct_training_targets(frame: pd.DataFrame) -> pd.Series:
    maximum_cycles = frame.groupby("unit_number")["time_in_cycles"].transform("max")
    return (maximum_cycles - frame["time_in_cycles"]).clip(upper=RUL_CAP)


def construct_test_observation_targets(
    test_frame: pd.DataFrame, official_rul: np.ndarray
) -> pd.Series:
    engine_ids = sorted(test_frame["unit_number"].unique())
    if len(engine_ids) != len(official_rul):
        raise ValueError(
            "Official RUL_FD001.txt row count must match the number of test engines."
        )

    targets = []
    for engine_id, final_rul in zip(engine_ids, official_rul, strict=True):
        engine_rows = test_frame[test_frame["unit_number"] == engine_id]
        last_cycle = int(engine_rows["time_in_cycles"].max())
        for cycle in engine_rows["time_in_cycles"]:
            rul = float(final_rul) + (last_cycle - int(cycle))
            targets.append(rul)

    return pd.Series(targets, dtype=float)


def _read_official_rul(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Official RUL file was not found at {path}")
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Official RUL_FD001.txt contains invalid values.")
    if (values < 0).any():
        raise ValueError("Official RUL_FD001.txt values must be non-negative.")
    return values


def nasa_scoring_function(actual_rul: np.ndarray, predicted_rul: np.ndarray) -> float:
    actual = np.asarray(actual_rul, dtype=float)
    predicted = np.asarray(predicted_rul, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted RUL arrays must have matching shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted RUL arrays must contain finite values.")
    if actual.size == 0:
        return 0.0

    errors = predicted - actual
    score = np.empty_like(errors)
    negative_mask = errors < 0
    positive_mask = ~negative_mask

    with np.errstate(over="ignore", invalid="ignore"):
        score[negative_mask] = np.expm1(-errors[negative_mask] / 13.0)
        score[positive_mask] = np.expm1(errors[positive_mask] / 10.0)

    if not np.isfinite(score).all():
        raise OverflowError(
            "NASA scoring exponent exceeded the representable floating-point range."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        aggregate = float(score.sum(dtype=np.float64))
    if not np.isfinite(aggregate):
        raise OverflowError(
            "NASA scoring aggregate exceeded the representable floating-point range."
        )
    return aggregate


def _make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), feature_columns),
        ],
        remainder="drop",
    )


def _make_models(feature_columns: list[str]) -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            [
                ("preprocessor", _make_preprocessor(feature_columns)),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", _make_preprocessor(feature_columns)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=160,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", _make_preprocessor(feature_columns)),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=160,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    clipped = np.clip(predicted, 0.0, RUL_CAP)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, clipped))),
        "mae": float(mean_absolute_error(actual, clipped)),
        "nasa_score": nasa_scoring_function(actual, clipped),
    }


def get_last_test_rows(test_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        test_frame.sort_values(["unit_number", "time_in_cycles"])
        .groupby("unit_number", sort=True)
        .tail(1)
        .reset_index(drop=True)
    )


def select_validation_snapshot(
    frame: pd.DataFrame,
    engine_id: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, float, int, int]:
    engine_rows = (
        frame[frame["unit_number"] == engine_id]
        .sort_values("time_in_cycles")
    )
    if engine_rows.empty:
        raise ValueError(f"Validation engine {engine_id} was not found.")

    final_cycle = int(engine_rows["time_in_cycles"].max())
    if final_cycle <= 1:
        raise ValueError(
            f"Validation engine {engine_id} has no strictly earlier cutoff cycle."
        )

    # A fixed seeded RNG makes fold cutoffs reproducible. The upper bound is
    # exclusive, so cutoff_cycle is always strictly less than final_cycle.
    cutoff_cycle = int(rng.integers(1, final_cycle))
    available = engine_rows[engine_rows["time_in_cycles"] <= cutoff_cycle]
    if available.empty:
        raise ValueError(
            f"Validation engine {engine_id} has no observation at or before cutoff."
        )
    snapshot = available.iloc[-1]
    actual_remaining_life = min(RUL_CAP, final_cycle - cutoff_cycle)
    return snapshot[FEATURE_COLUMNS].copy(), float(actual_remaining_life), cutoff_cycle, final_cycle


def _validate_test_engine_matching(
    test_frame: pd.DataFrame,
    last_test_rows: pd.DataFrame,
    official_rul: np.ndarray,
) -> None:
    test_engine_ids = sorted(test_frame["unit_number"].unique().tolist())
    unique_test_engines = int(test_frame["unit_number"].nunique())
    if unique_test_engines != 100:
        raise ValueError(f"FD001 must contain exactly 100 test engines; found {unique_test_engines}.")
    if len(last_test_rows) != 100:
        raise ValueError(
            f"FD001 must select exactly 100 final test observations; found {len(last_test_rows)}."
        )
    if len(official_rul) != 100:
        raise ValueError(
            f"RUL_FD001.txt must contain exactly 100 official values; found {len(official_rul)}."
        )

    selected_engine_ids = last_test_rows["unit_number"].tolist()
    if selected_engine_ids != test_engine_ids:
        raise ValueError(
            "Selected test engine order does not match sorted test engine order."
        )
    if len(set(selected_engine_ids)) != 100:
        raise ValueError("Final test observations do not contain exactly 100 unique engines.")
    if test_engine_ids != list(range(1, 101)):
        raise ValueError(
            "Test engine IDs must be 1-100 in official RUL_FD001.txt position order."
        )


def train_rul_system(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    official_rul_path: Path = RUL_DATA_PATH,
) -> RulSystem:
    train_frame = _read_fd001(train_path)
    test_frame = _read_fd001(test_path)
    official_rul = _read_official_rul(official_rul_path)

    engine_groups = train_frame["unit_number"]

    models = _make_models(FEATURE_COLUMNS)
    cv = GroupKFold(n_splits=5)
    rng = np.random.default_rng(RANDOM_STATE)
    validation_actual: list[float] = []
    validation_predictions: dict[str, list[float]] = {
        name: [] for name in models
    }

    for train_indices, validation_indices in cv.split(
        train_frame,
        groups=engine_groups,
    ):
        fold_train_frame = train_frame.iloc[train_indices].copy()
        fold_validation_frame = train_frame.iloc[validation_indices].copy()
        fold_train_ids = set(fold_train_frame["unit_number"].unique())
        fold_validation_ids = set(fold_validation_frame["unit_number"].unique())
        if fold_train_ids & fold_validation_ids:
            raise RuntimeError("Engine-wise validation overlap detected.")

        fold_train_targets = construct_training_targets(fold_train_frame)
        fold_train_features = fold_train_frame[FEATURE_COLUMNS]
        validation_snapshots = []

        for validation_engine_id in sorted(fold_validation_ids):
            snapshot, actual_rul, cutoff_cycle, final_cycle = (
                select_validation_snapshot(
                    fold_validation_frame,
                    validation_engine_id,
                    rng,
                )
            )
            if cutoff_cycle >= final_cycle:
                raise RuntimeError(
                    "Validation cutoff was not strictly before the engine final cycle."
                )
            validation_actual.append(actual_rul)
            validation_snapshots.append(snapshot)

        snapshot_frame = pd.DataFrame(validation_snapshots, columns=FEATURE_COLUMNS)
        for name, model in models.items():
            model.fit(fold_train_features, fold_train_targets)
            predictions = model.predict(snapshot_frame)
            validation_predictions[name].extend(float(value) for value in predictions)

    if len(validation_actual) != train_frame["unit_number"].nunique():
        raise RuntimeError(
            "Engine-wise validation did not produce one prediction per training engine."
        )

    validation_metrics: list[dict[str, Any]] = []
    validation_actual_array = np.asarray(validation_actual, dtype=float)
    for name in models:
        predicted_array = np.asarray(validation_predictions[name], dtype=float)
        if len(predicted_array) != len(validation_actual_array):
            raise RuntimeError(
                f"{name} validation did not produce one prediction per engine."
            )
        metrics = _regression_metrics(validation_actual_array, predicted_array)
        validation_metrics.append({"model": name, **metrics})

    selected_name = min(
        validation_metrics,
        key=lambda item: item["rmse"],
    )["model"]
    selected_model = models[selected_name]
    train_targets = construct_training_targets(train_frame)
    features = train_frame[FEATURE_COLUMNS]
    selected_model.fit(features, train_targets)

    last_test_rows = get_last_test_rows(test_frame)
    _validate_test_engine_matching(test_frame, last_test_rows, official_rul)

    test_metrics: list[dict[str, Any]] = []
    last_test_features = last_test_rows[FEATURE_COLUMNS]
    for name, model in models.items():
        if name == selected_name:
            model = selected_model
        else:
            model.fit(features, train_targets)
        predicted = model.predict(last_test_features)
        if len(predicted) != 100:
            raise RuntimeError(
                f"{name} produced {len(predicted)} final test predictions; expected exactly 100."
            )
        metrics = _regression_metrics(official_rul, predicted)
        test_metrics.append({"model": name, **metrics})

    train_engine_count = int(train_frame["unit_number"].nunique())
    test_engine_count = int(test_frame["unit_number"].nunique())
    dataset_summary = {
        "name": "NASA C-MAPSS FD001",
        "source": "NASA Prognostics Data Repository",
        "source_url": (
            "https://phm-datasets.s3.amazonaws.com/NASA/"
            "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
        ),
        "training_engines": train_engine_count,
        "test_engines": test_engine_count,
        "training_observations": int(len(train_frame)),
        "test_observations": int(len(test_frame)),
        "operational_settings": 3,
        "sensor_measurements": 21,
        "rul_cap": RUL_CAP,
        "validation": (
            "5-fold GroupKFold by engine ID; one seeded cutoff snapshot "
            "per held-out engine"
        ),
        "test_evaluation": (
            "Official RUL_FD001.txt targets matched to the last observed "
            "row of each test engine"
        ),
    }

    return RulSystem(
        models=models,
        selected_model_name=selected_name,
        selected_model=selected_model,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        dataset_summary=dataset_summary,
        train_feature_columns=FEATURE_COLUMNS,
        raw_official_test_rul=official_rul.copy(),
    )


def predict_rul(system: RulSystem, observation: dict[str, Any]) -> dict[str, Any]:
    features = pd.DataFrame(
        [{column: observation[column] for column in system.train_feature_columns}],
        columns=system.train_feature_columns,
    )
    raw_prediction = float(system.selected_model.predict(features)[0])
    predicted_rul = float(np.clip(raw_prediction, 0.0, RUL_CAP))
    if predicted_rul < 20:
        interpretation = "Low remaining-life estimate; prioritize immediate inspection."
    elif predicted_rul < 50:
        interpretation = "Moderate remaining-life estimate; plan maintenance review."
    else:
        interpretation = "Higher remaining-life estimate; continue routine monitoring."
    return {
        "predicted_rul": predicted_rul,
        "model": system.selected_model_name,
        "interpretation": interpretation,
        "units": "cycles",
        "capped_upper_bound": RUL_CAP,
        "recommendation": build_rul_recommendation(predicted_rul),
    }
