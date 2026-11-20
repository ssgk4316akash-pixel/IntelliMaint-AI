import logging

import numpy as np

from app import runtime
from app.ml import TrainedSystem
from app.rul import RulSystem


def _trained_system() -> TrainedSystem:
    return TrainedSystem(
        models={},
        cross_validation_f1={},
        metrics=[],
        selected_model_name="test",
        selected_model=None,
        preprocessor=None,
        anomaly_model=None,
        anomaly_metrics={},
        feature_importance=[],
        dataset_summary={"rows": 10_000},
        explanation_explainer=None,
        zscore_baseline={},
    )


def _rul_system() -> RulSystem:
    return RulSystem(
        models={},
        selected_model_name="test",
        selected_model=None,
        validation_metrics=[],
        test_metrics=[],
        dataset_summary={"training_engines": 100},
        train_feature_columns=[],
        raw_official_test_rul=np.array([], dtype=float),
    )


def test_ai4i_system_loads_from_persistent_cache(tmp_path, monkeypatch) -> None:
    runtime.get_system.cache_clear()
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(runtime, "_cache_fingerprint", lambda *args: "same-inputs")
    trained = _trained_system()
    training_calls = 0

    def train(_data_path):
        nonlocal training_calls
        training_calls += 1
        return trained

    monkeypatch.setattr(runtime, "train_system", train)
    try:
        assert runtime.get_system() is trained
        runtime.get_system.cache_clear()

        loaded = runtime.get_system()

        assert isinstance(loaded, TrainedSystem)
        assert loaded.dataset_summary == trained.dataset_summary
        assert training_calls == 1
    finally:
        runtime.get_system.cache_clear()


def test_rul_system_retrains_when_cache_fingerprint_changes(
    tmp_path, monkeypatch
) -> None:
    runtime.get_rul_system.cache_clear()
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path)
    fingerprints = iter(("old-inputs", "new-inputs"))
    monkeypatch.setattr(
        runtime,
        "_cache_fingerprint",
        lambda *args: next(fingerprints),
    )
    systems = (_rul_system(), _rul_system())
    training_calls = 0

    def train():
        nonlocal training_calls
        system = systems[training_calls]
        training_calls += 1
        return system

    monkeypatch.setattr(runtime, "train_rul_system", train)
    try:
        assert runtime.get_rul_system() is systems[0]
        runtime.get_rul_system.cache_clear()

        loaded = runtime.get_rul_system()

        assert loaded is systems[1]
        assert training_calls == 2
    finally:
        runtime.get_rul_system.cache_clear()


def test_unreadable_cache_is_logged_and_rebuilt(tmp_path, monkeypatch, caplog) -> None:
    runtime.get_system.cache_clear()
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(runtime, "_cache_fingerprint", lambda *args: "same-inputs")
    (tmp_path / "ai4i.joblib").write_bytes(b"corrupt cache")
    trained = _trained_system()
    training_calls = 0

    def train(_data_path):
        nonlocal training_calls
        training_calls += 1
        return trained

    monkeypatch.setattr(runtime, "train_system", train)
    try:
        with caplog.at_level(logging.WARNING):
            assert runtime.get_system() is trained
        assert "Ignoring unreadable ai4i model cache" in caplog.text
        runtime.get_system.cache_clear()

        loaded = runtime.get_system()

        assert isinstance(loaded, TrainedSystem)
        assert training_calls == 1
    finally:
        runtime.get_system.cache_clear()
