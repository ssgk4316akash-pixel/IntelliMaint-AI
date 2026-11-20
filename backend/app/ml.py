from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42
ZSCORE_THRESHOLD = 3.0
ZERO_VARIANCE_ATOL = 1e-9
SHAP_BACKGROUND_SIZE = 20
SHAP_PERMUTATIONS = 5
SHAP_MAX_EVALS = SHAP_PERMUTATIONS * (2 * len(FEATURE_COLUMNS) + 1)
SHAP_ADDITIVITY_TOLERANCE = 1e-6
PRODUCT_TYPE_CODES = {"L": 0.0, "M": 1.0, "H": 2.0}
PRODUCT_TYPES_BY_CODE = {
    code: product_type for product_type, code in PRODUCT_TYPE_CODES.items()
}

DISPLAY_NAMES = {
    "Air temperature [K]": "Air temperature",
    "Process temperature [K]": "Process temperature",
    "Rotational speed [rpm]": "Rotational speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool wear",
    "Type": "Product type",
}


@dataclass
class TrainedSystem:
    models: dict[str, Pipeline]
    cross_validation_f1: dict[str, float]
    metrics: list[dict[str, Any]]
    selected_model_name: str
    selected_model: Pipeline
    preprocessor: ColumnTransformer
    anomaly_model: IsolationForest
    anomaly_metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    dataset_summary: dict[str, Any]
    explanation_explainer: shap.Explainer
    zscore_baseline: dict[str, dict[str, float]]


def _read_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not data_path.is_file():
        raise FileNotFoundError(f"AI4I dataset was not found at {data_path}")

    frame = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"AI4I dataset is missing required columns: {sorted(missing)}")

    features = frame[FEATURE_COLUMNS].copy()
    targets = frame[TARGET_COLUMN].astype(int)
    if features.isna().any().any() or targets.isna().any():
        raise ValueError("AI4I data contains missing feature or target values.")
    if not set(targets.unique()).issubset({0, 1}):
        raise ValueError("Machine failure labels must be binary values 0 or 1.")
    return features, targets


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _make_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=160,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=160,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _classification_metrics(
    model: Pipeline, features: pd.DataFrame, targets: pd.Series
) -> dict[str, Any]:
    predicted = model.predict(features)
    scores = model.predict_proba(features)[:, 1]
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
    )
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "precision": float(precision_score(targets, predicted, zero_division=0)),
        "recall": float(recall_score(targets, predicted, zero_division=0)),
        "f1": float(f1_score(targets, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }


def _extract_feature_importance(model: Pipeline) -> list[dict[str, Any]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    encoded_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    else:
        values = np.abs(classifier.coef_[0])

    aggregated: dict[str, float] = {}
    for encoded_name, value in zip(encoded_names, values, strict=True):
        feature_name = encoded_name.split("__", 1)[-1]
        source_name = "Type" if feature_name.startswith("Type_") else feature_name
        aggregated[source_name] = aggregated.get(source_name, 0.0) + float(value)

    total = sum(aggregated.values())
    return [
        {"feature": DISPLAY_NAMES.get(name, name), "importance": value / total}
        for name, value in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]


@dataclass
class _FailureScorePredictor:
    model: Pipeline

    def __call__(self, values: np.ndarray) -> np.ndarray:
        features = pd.DataFrame(
            np.asarray(values, dtype=float), columns=FEATURE_COLUMNS
        )
        features["Type"] = features["Type"].map(PRODUCT_TYPES_BY_CODE)
        if features["Type"].isna().any():
            raise ValueError("SHAP produced an unsupported product type code.")
        return self.model.predict_proba(features)[:, 1]


def _make_explanation_explainer(
    model: Pipeline, training_features: pd.DataFrame
) -> shap.Explainer:
    background = training_features.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(training_features)),
        random_state=RANDOM_STATE,
    )
    background = background.copy()
    background["Type"] = background["Type"].map(PRODUCT_TYPE_CODES)
    if background["Type"].isna().any():
        raise ValueError("Training data contains an unsupported product type.")

    return shap.Explainer(
        _FailureScorePredictor(model),
        background.to_numpy(dtype=float),
        algorithm="permutation",
        seed=RANDOM_STATE,
    )


def train_system(data_path: Path = DATA_PATH) -> TrainedSystem:
    features, targets = _read_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    models = _make_models()
    cross_validation_f1: dict[str, float] = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cross_validation_f1[name] = float(scores.mean())
        model.fit(x_train, y_train)

    selected_model_name = max(cross_validation_f1, key=cross_validation_f1.get)
    selected_model = models[selected_model_name]
    metrics = [
        {"model": name, **_classification_metrics(model, x_test, y_test)}
        for name, model in models.items()
    ]

    preprocessor = _make_preprocessor()
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    anomaly_model = IsolationForest(
        n_estimators=160,
        contamination=0.03,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    anomaly_model.fit(transformed_train)
    anomaly_predictions = anomaly_model.predict(transformed_test) == -1
    anomaly_metrics = {
        "precision": float(
            precision_score(y_test, anomaly_predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, anomaly_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, anomaly_predictions, zero_division=0)),
        "flagged": int(anomaly_predictions.sum()),
        "test_rows": int(len(y_test)),
        "contamination_assumption": 0.03,
    }

    failure_count = int(targets.sum())
    dataset_summary = {
        "name": "AI4I 2020 Predictive Maintenance",
        "source": "UCI Machine Learning Repository",
        "rows": int(len(features)),
        "failure_cases": failure_count,
        "failure_rate": float(targets.mean()),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "split": "Stratified 80/20 holdout; random_state=42",
    }
    healthy_training = x_train.loc[y_train == 0, NUMERIC_FEATURES]
    zscore_baseline = {
        feature: {
            "mean": float(healthy_training[feature].mean()),
            "std": float(healthy_training[feature].std(ddof=0)),
        }
        for feature in NUMERIC_FEATURES
    }
    return TrainedSystem(
        models=models,
        cross_validation_f1=cross_validation_f1,
        metrics=metrics,
        selected_model_name=selected_model_name,
        selected_model=selected_model,
        preprocessor=preprocessor,
        anomaly_model=anomaly_model,
        anomaly_metrics=anomaly_metrics,
        feature_importance=_extract_feature_importance(selected_model),
        dataset_summary=dataset_summary,
        explanation_explainer=_make_explanation_explainer(selected_model, x_train),
        zscore_baseline=zscore_baseline,
    )


def _observation_features(observation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": observation["product_type"],
                "Air temperature [K]": observation["air_temperature"],
                "Process temperature [K]": observation["process_temperature"],
                "Rotational speed [rpm]": observation["rotational_speed"],
                "Torque [Nm]": observation["torque"],
                "Tool wear [min]": observation["tool_wear"],
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def _explanation_input(features: pd.DataFrame) -> np.ndarray:
    encoded = features.copy()
    encoded["Type"] = encoded["Type"].map(PRODUCT_TYPE_CODES)
    if encoded["Type"].isna().any():
        raise ValueError("Observation contains an unsupported product type.")
    return encoded.to_numpy(dtype=float)


def make_prediction(system: TrainedSystem, observation: dict[str, Any]) -> dict[str, Any]:
    features = _observation_features(observation)
    score = float(system.selected_model.predict_proba(features)[0, 1])
    if score >= 0.7:
        priority = "Critical"
    elif score >= 0.4:
        priority = "High"
    elif score >= 0.2:
        priority = "Moderate"
    else:
        priority = "Low"

    return {
        "model": system.selected_model_name,
        "failure_score": score,
        "predicted_failure": bool(score >= 0.5),
        "priority": priority,
        "feature_importance": system.feature_importance,
    }


def make_explanation(
    system: TrainedSystem, observation: dict[str, Any]
) -> dict[str, Any]:
    features = _observation_features(observation)
    failure_score = float(system.selected_model.predict_proba(features)[0, 1])
    explanation = system.explanation_explainer(
        _explanation_input(features),
        max_evals=SHAP_MAX_EVALS,
    )
    shap_values = np.asarray(explanation.values, dtype=float)
    if shap_values.shape != (1, len(FEATURE_COLUMNS)):
        raise RuntimeError(
            f"Unexpected SHAP value shape: {shap_values.shape}"
        )

    baseline_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if baseline_values.size != 1:
        raise RuntimeError(
            f"Unexpected SHAP baseline shape: {baseline_values.shape}"
        )
    baseline_value = float(baseline_values[0])
    contributions = shap_values[0]
    explained_score = baseline_value + float(contributions.sum())
    if not np.isclose(
        explained_score,
        failure_score,
        rtol=0.0,
        atol=SHAP_ADDITIVITY_TOLERANCE,
    ):
        raise RuntimeError(
            "SHAP contributions do not add up to the selected model's "
            f"class-1 failure score (difference={explained_score - failure_score})."
        )

    feature_contributions = []
    for column, value, contribution in zip(
        FEATURE_COLUMNS, features.iloc[0], contributions, strict=True
    ):
        if contribution > 0:
            direction = "increases"
        elif contribution < 0:
            direction = "decreases"
        else:
            direction = "no change"
        feature_value = value.item() if isinstance(value, np.generic) else value
        feature_contributions.append(
            {
                "feature": DISPLAY_NAMES.get(column, column),
                "value": feature_value,
                "shap_value": float(contribution),
                "direction": direction,
            }
        )

    feature_contributions.sort(
        key=lambda item: abs(item["shap_value"]), reverse=True
    )
    return {
        "model": system.selected_model_name,
        "failure_score": failure_score,
        "predicted_failure": bool(failure_score >= 0.5),
        "baseline_value": baseline_value,
        "contributions": feature_contributions,
        "interpretation": (
            "Positive SHAP values increase the uncalibrated class-1 failure score "
            "relative to the baseline; negative SHAP values decrease it."
        ),
    }


def make_anomaly_prediction(
    system: TrainedSystem, observation: dict[str, Any]
) -> dict[str, Any]:
    features = _observation_features(observation)
    transformed = system.preprocessor.transform(features)
    anomaly_score = float(-system.anomaly_model.decision_function(transformed)[0])
    is_anomaly = bool(system.anomaly_model.predict(transformed)[0] == -1)
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "threshold_assumption": system.anomaly_metrics["contamination_assumption"],
    }


def make_zscore_prediction(
    system: TrainedSystem, observation: dict[str, Any]
) -> dict[str, Any]:
    features = _observation_features(observation).iloc[0]
    feature_zscores: dict[str, float] = {}
    exceeded_features: list[str] = []
    zero_variance_features: list[str] = []

    for feature in NUMERIC_FEATURES:
        value = float(features[feature])
        if not np.isfinite(value):
            raise ValueError(f"{DISPLAY_NAMES[feature]} must be a finite number.")

        baseline = system.zscore_baseline[feature]
        mean = baseline["mean"]
        std = baseline["std"]
        display_name = DISPLAY_NAMES[feature]
        if std == 0.0:
            zero_variance_features.append(display_name)
            if np.isclose(value, mean, rtol=0.0, atol=ZERO_VARIANCE_ATOL):
                zscore = 0.0
            else:
                # Zero variance has no mathematical Z-score. These signed
                # threshold sentinels preserve the numeric API convention.
                zscore = ZSCORE_THRESHOLD if value > mean else -ZSCORE_THRESHOLD
        else:
            zscore = (value - mean) / std
        if not np.isfinite(zscore):
            raise ValueError(f"Could not calculate a finite Z-score for {DISPLAY_NAMES[feature]}.")

        feature_zscores[display_name] = float(zscore)
        if abs(zscore) >= ZSCORE_THRESHOLD:
            exceeded_features.append(display_name)

    absolute_zscores = [abs(value) for value in feature_zscores.values()]
    max_absolute_zscore = max(absolute_zscores, default=0.0)
    return {
        "is_anomaly": bool(exceeded_features),
        "feature_zscores": feature_zscores,
        "exceeded_features": exceeded_features,
        "threshold": ZSCORE_THRESHOLD,
        "max_absolute_zscore": float(max_absolute_zscore),
        "zero_variance_features": zero_variance_features,
    }
