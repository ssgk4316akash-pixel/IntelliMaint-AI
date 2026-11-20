from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import pickle
import sys
import tempfile
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

import joblib

from app.ml import DATA_PATH, TrainedSystem, train_system
from app.rul import (
    RUL_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    RulSystem,
    train_rul_system,
)


logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parents[1] / ".model_cache"
CACHE_FORMAT_VERSION = 1
T = TypeVar("T")


def _cache_fingerprint(
    cache_name: str,
    source_path: Path,
    data_paths: tuple[Path, ...],
    packages: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{CACHE_FORMAT_VERSION}:{cache_name}:{sys.version_info[:3]}".encode())

    for path in (source_path, *data_paths):
        digest.update(path.name.encode())
        if not path.is_file():
            digest.update(b":missing")
            continue
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)

    for package in packages:
        digest.update(package.encode())
        digest.update(importlib.metadata.version(package).encode())

    return digest.hexdigest()


def _load_or_train(
    cache_name: str,
    expected_type: type[T],
    source_path: Path,
    data_paths: tuple[Path, ...],
    packages: tuple[str, ...],
    trainer: Callable[[], T],
) -> T:
    cache_path = CACHE_DIR / f"{cache_name}.joblib"
    fingerprint = _cache_fingerprint(cache_name, source_path, data_paths, packages)

    if cache_path.is_file():
        try:
            cached = joblib.load(cache_path)
        except (
            AttributeError,
            EOFError,
            ImportError,
            IndexError,
            KeyError,
            OSError,
            pickle.PickleError,
            TypeError,
            ValueError,
            zlib.error,
        ) as error:
            logger.warning(
                "Ignoring unreadable %s model cache and retraining: %s",
                cache_name,
                error,
            )
        else:
            if (
                isinstance(cached, dict)
                and cached.get("format_version") == CACHE_FORMAT_VERSION
                and cached.get("fingerprint") == fingerprint
                and isinstance(cached.get("system"), expected_type)
            ):
                logger.info("Loaded cached %s model system.", cache_name)
                return cached["system"]
            logger.info("The %s model cache is stale; retraining.", cache_name)

    system = trainer()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=CACHE_DIR,
            prefix=f".{cache_name}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            joblib.dump(
                {
                    "format_version": CACHE_FORMAT_VERSION,
                    "fingerprint": fingerprint,
                    "system": system,
                },
                temporary_path,
                compress=3,
            )
            os.replace(temporary_path, cache_path)
            logger.info("Saved cached %s model system.", cache_name)
        finally:
            temporary_path.unlink(missing_ok=True)
    except (
        AttributeError,
        OSError,
        pickle.PickleError,
        TypeError,
        ValueError,
        zlib.error,
    ) as error:
        logger.warning(
            "Could not save the %s model cache; current run will continue: %s",
            cache_name,
            error,
        )
    return system


@lru_cache(maxsize=1)
def get_system() -> TrainedSystem:
    return _load_or_train(
        "ai4i",
        TrainedSystem,
        Path(__file__).with_name("ml.py"),
        (DATA_PATH,),
        ("joblib", "numpy", "pandas", "scikit-learn", "shap", "xgboost"),
        lambda: train_system(DATA_PATH),
    )


@lru_cache(maxsize=1)
def get_rul_system() -> RulSystem:
    return _load_or_train(
        "nasa-fd001",
        RulSystem,
        Path(__file__).with_name("rul.py"),
        (TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH),
        ("joblib", "numpy", "pandas", "scikit-learn", "xgboost"),
        train_rul_system,
    )
