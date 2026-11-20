from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.database import count_assessments, database_status, save_assessment
from app.ml import (
    make_anomaly_prediction,
    make_prediction,
    make_zscore_prediction,
)
from app.recommendations import build_recommendation
from app.rul import predict_rul
from app.runtime import get_rul_system, get_system
from app.schemas import RulReading, SensorReading
from app.simulator import SensorSimulator


SIMULATOR_ASSET_ID = "sim-machine-1"
logger = logging.getLogger(__name__)


class StreamHub:
    def __init__(self) -> None:
        self.interval_seconds = 2.0
        self.simulator = SensorSimulator()
        self.latest: dict[str, Any] | None = None
        self.published = 0
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._lock = asyncio.Lock()
        self.history: list[dict[str, Any]] = []
        self.max_history_size = 100

    @property
    def running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=20)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def start(self, interval_seconds: float) -> None:
        async with self._lock:
            self.interval_seconds = interval_seconds
            if self.running:
                self._running = True
                return
            self._running = True
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            task = self._task
            self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def publish(self, event: dict[str, Any]) -> None:
        self.latest = event
        self.published += 1
        self.history.append(event)
        if len(self.history) > self.max_history_size:
            del self.history[: len(self.history) - self.max_history_size]
        for queue in list(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self.history[-limit:]))

    def clear_history(self) -> None:
        self.history.clear()
        self.latest = None
        self.published = 0

    def reset(self, seed: int = 42) -> None:
        self.clear_history()
        self.simulator = SensorSimulator(seed)

    async def generate(self, mode: str = "mixed") -> dict[str, Any]:
        if mode not in {"normal", "abnormal", "mixed"}:
            raise ValueError("Simulation mode must be normal, abnormal, or mixed.")
        system = await asyncio.to_thread(get_system)
        self.simulator.zscore_baseline = system.zscore_baseline
        reading = self.simulator.next_reading(mode=mode)
        event = await asyncio.to_thread(
            assess_machine_reading,
            reading,
            "simulator",
            SIMULATOR_ASSET_ID,
            persist=False,
        )
        event["simulated"] = True
        event["simulation_mode"] = reading["simulation_mode"]
        await self.publish(event)
        return event

    def status(self) -> dict[str, Any]:
        database = database_status()
        stored = count_assessments() if database["connected"] else None
        return {
            "simulator_running": self.running,
            "interval_seconds": self.interval_seconds,
            "asset_id": SIMULATOR_ASSET_ID,
            "subscriber_count": len(self._subscribers),
            "samples_published": self.published,
            "stored_assessments": stored,
            "database": database,
            "latest": self.latest,
        }

    async def _loop(self) -> None:
        try:
            while self._running:
                try:
                    system = await asyncio.to_thread(get_system)
                    self.simulator.zscore_baseline = system.zscore_baseline
                    reading = self.simulator.next_reading()
                    try:
                        event = await asyncio.to_thread(
                            assess_machine_reading,
                            reading,
                            "simulator",
                            SIMULATOR_ASSET_ID,
                            False,
                        )
                    except (FileNotFoundError, ValueError) as error:
                        logger.exception("Simulation assessment failed.")
                        event = {
                            "simulated": True,
                            "source": "simulator",
                            "asset_id": SIMULATOR_ASSET_ID,
                            "persisted": False,
                            "error": str(error),
                        }
                    await self.publish(event)
                except Exception:
                    logger.exception(
                        "Unexpected simulation tick failure; retrying after the configured interval."
                    )
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False


hub = StreamHub()


def assess_machine_reading(
    raw_reading: dict[str, Any],
    source: str,
    asset_id: str,
    persist: bool = True,
) -> dict[str, Any]:
    reading = SensorReading.model_validate(raw_reading).model_dump()
    system = get_system()
    prediction = make_prediction(system, reading)
    anomaly = make_anomaly_prediction(system, reading)
    zscore = make_zscore_prediction(system, reading)
    recommendation = build_recommendation(reading, prediction, anomaly, zscore)
    simulated = raw_reading.get("simulated") is True
    simulation_mode = raw_reading.get("simulation_mode")
    stored = None
    if persist:
        stored = save_assessment(
            source=source,
            asset_id=asset_id,
            kind="machine_assessment",
            reading=reading,
            prediction=_public_prediction(prediction),
            anomaly=anomaly,
            recommendation=recommendation,
            zscore=zscore,
            simulated=simulated,
        )
    payload = {
        "source": source,
        "asset_id": asset_id,
        "kind": "machine_assessment",
        "reading": reading,
        "prediction": _public_prediction(prediction),
        "anomaly": anomaly,
        "zscore": zscore,
        "recommendation": recommendation,
        "simulated": simulated,
        "simulation_mode": simulation_mode,
    }
    if persist:
        return _with_persistence(
            stored,
            **payload,
        )
    return {
        "id": None,
        "created_at": None,
        "persisted": False,
        **payload,
    }


def assess_rul_reading(raw_reading: dict[str, Any], asset_id: str) -> dict[str, Any]:
    reading = RulReading.model_validate(raw_reading).model_dump()
    result = predict_rul(get_rul_system(), reading)
    recommendation = result["recommendation"]
    rul = {key: value for key, value in result.items() if key != "recommendation"}
    stored = save_assessment(
        source="manual",
        asset_id=asset_id,
        kind="rul_estimate",
        reading=reading,
        recommendation=recommendation,
        rul=rul,
    )
    result["persisted"] = stored is not None
    result["id"] = None if stored is None else stored["id"]
    return result


def record_failure_prediction(reading: dict[str, Any], prediction: dict[str, Any]) -> None:
    save_assessment(
        source="manual",
        asset_id="manual-entry",
        kind="failure_prediction",
        reading=reading,
        prediction=_public_prediction(prediction),
    )


def record_anomaly(reading: dict[str, Any], anomaly: dict[str, Any]) -> None:
    save_assessment(
        source="manual",
        asset_id="manual-entry",
        kind="anomaly_check",
        reading=reading,
        anomaly=anomaly,
    )


def encode_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def _public_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": prediction["model"],
        "failure_score": prediction["failure_score"],
        "predicted_failure": prediction["predicted_failure"],
        "priority": prediction["priority"],
    }


def _with_persistence(stored: dict[str, Any] | None, **payload: Any) -> dict[str, Any]:
    event = {
        "id": None if stored is None else stored["id"],
        "created_at": None if stored is None else stored["created_at"],
        "persisted": stored is not None,
        **payload,
    }
    return event
