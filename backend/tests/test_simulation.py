import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

from app.monitoring import StreamHub, assess_machine_reading
from app.runtime import get_system
from app.schemas import SensorReading
from app.simulator import SensorSimulator


def test_simulator_respects_constraints_and_fixed_seed_is_deterministic() -> None:
    first = SensorSimulator(seed=123)
    second = SensorSimulator(seed=123)

    first_normal = [first.next_reading("normal") for _ in range(5)]
    second_normal = [second.next_reading("normal") for _ in range(5)]

    assert first_normal == second_normal
    for reading in first_normal:
        validated = SensorReading.model_validate(reading)
        assert validated.air_temperature <= 500
        assert validated.process_temperature <= 500
        assert validated.rotational_speed <= 10000
        assert validated.torque <= 1000
        assert validated.tool_wear <= 1000
        assert reading["simulated"] is True
        assert reading["simulation_mode"] in {"normal", "abnormal"}


def test_simulator_normal_and_abnormal_modes() -> None:
    normal = SensorSimulator(seed=42).next_reading("normal")
    abnormal = SensorSimulator(seed=42).next_reading("abnormal")

    assert normal["simulation_mode"] == "normal"
    assert abnormal["simulation_mode"] == "abnormal"
    assert abnormal["torque"] != normal["torque"] or abnormal["speed"] != normal["speed"]


def test_simulation_history_is_bounded_resettable_and_retrievable() -> None:
    hub = StreamHub()

    async def exercise() -> None:
        for index in range(hub.max_history_size + 25):
            await hub.publish({
                "simulated": True,
                "index": index,
                "reading": {"tool_wear": 40},
            })

        assert len(hub.history) == hub.max_history_size
        assert hub.history[0]["index"] == 25
        assert hub.get_history(10)[0]["index"] == hub.max_history_size + 24

        hub.reset(seed=99)
        assert hub.history == []
        assert hub.latest is None
        assert hub.published == 0
        assert hub.simulator._rng is not None

    asyncio.run(exercise())


def test_assessment_integrates_prediction_anomaly_and_zscore() -> None:
    reading = SensorSimulator(seed=42).next_reading("normal")

    event = assess_machine_reading(
        reading,
        source="simulator",
        asset_id="test-sim-machine",
    )

    assert event["simulated"] is True
    assert event["reading"]["tool_wear"] == reading["tool_wear"]
    assert "prediction" in event
    assert "anomaly" in event
    assert "zscore" in event
    assert event["asset_id"] == "test-sim-machine"


def test_abnormal_reading_is_zscore_flagged() -> None:
    baseline = get_system().zscore_baseline
    simulator = SensorSimulator(seed=42, zscore_baseline=baseline)

    abnormal = simulator.next_reading("abnormal")
    event = assess_machine_reading(
        abnormal,
        source="simulator",
        asset_id="test-sim-machine",
        persist=False,
    )

    assert abnormal["simulation_mode"] == "abnormal"
    assert event["zscore"]["is_anomaly"] is True
    assert event["zscore"]["max_absolute_zscore"] >= 3.0
    assert event["zscore"]["exceeded_features"]


def test_simulation_generation_never_persists_assessment() -> None:
    hub = StreamHub()

    async def exercise() -> None:
        with (
            patch("app.monitoring.save_assessment") as save_assessment,
            patch(
                "app.monitoring.assess_machine_reading",
                wraps=assess_machine_reading,
            ) as assessment,
        ):
            event = await hub.generate("abnormal")

        save_assessment.assert_not_called()
        assessment.assert_called_once()
        assert assessment.call_args.kwargs["persist"] is False
        assert event["persisted"] is False
        assert event["simulated"] is True

    asyncio.run(exercise())


def test_simulation_generation_offloads_model_work_from_event_loop() -> None:
    hub = StreamHub()
    initialization_started = threading.Event()
    assessment_started = threading.Event()
    release_initialization = threading.Event()
    release_assessment = threading.Event()
    event_loop_thread_ids: list[int] = []

    def initialize_system() -> SimpleNamespace:
        assert threading.get_ident() != event_loop_thread_ids[0]
        initialization_started.set()
        if not release_initialization.wait(timeout=1):
            raise TimeoutError("Timed out waiting to release model initialization.")
        return SimpleNamespace(zscore_baseline={})

    def assess(
        reading: dict[str, object],
        source: str,
        asset_id: str,
        *,
        persist: bool,
    ) -> dict[str, object]:
        assert threading.get_ident() != event_loop_thread_ids[0]
        assert persist is False
        assessment_started.set()
        if not release_assessment.wait(timeout=1):
            raise TimeoutError("Timed out waiting to release assessment.")
        return {
            "persisted": False,
            "simulated": True,
            "source": source,
            "asset_id": asset_id,
            "reading": reading,
        }

    async def assert_event_loop_responsive(task: asyncio.Task[dict[str, object]]) -> None:
        pulse = asyncio.Event()
        asyncio.get_running_loop().call_soon(pulse.set)
        await asyncio.wait_for(pulse.wait(), timeout=0.1)
        assert not task.done()

    async def exercise() -> None:
        event_loop_thread_ids.append(threading.get_ident())
        task: asyncio.Task[dict[str, object]] | None = None
        try:
            with (
                patch("app.monitoring.get_system", side_effect=initialize_system),
                patch("app.monitoring.assess_machine_reading", side_effect=assess),
            ):
                task = asyncio.create_task(hub.generate("normal"))
                assert await asyncio.wait_for(
                    asyncio.to_thread(initialization_started.wait),
                    timeout=1,
                )
                await assert_event_loop_responsive(task)
                release_initialization.set()

                assert await asyncio.wait_for(
                    asyncio.to_thread(assessment_started.wait),
                    timeout=1,
                )
                await assert_event_loop_responsive(task)
                release_assessment.set()

                event = await asyncio.wait_for(task, timeout=1)

            assert event["simulated"] is True
            assert event["simulation_mode"] == "normal"
            assert hub.latest == event
            assert hub.get_history(1) == [event]
        finally:
            release_initialization.set()
            release_assessment.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_background_tick_logs_and_retries_after_unexpected_error(caplog) -> None:
    hub = StreamHub()
    assessment_started = threading.Event()

    def fail_assessment(*args: object, **kwargs: object) -> None:
        assessment_started.set()
        raise RuntimeError("Unexpected assessment failure.")

    async def exercise() -> None:
        with (
            patch(
                "app.monitoring.get_system",
                return_value=SimpleNamespace(zscore_baseline={}),
            ),
            patch(
                "app.monitoring.assess_machine_reading",
                side_effect=fail_assessment,
            ) as assessment,
        ):
            await hub.start(0.5)
            assert await asyncio.wait_for(
                asyncio.to_thread(assessment_started.wait),
                timeout=1,
            )
            await asyncio.sleep(0.02)

            assert hub.running
            assessment.assert_called_once()
            await hub.stop()

        assert hub._task is None
        assert not hub.running

    asyncio.run(exercise())
    assert "Unexpected simulation tick failure" in caplog.text
    assert "Unexpected assessment failure." in caplog.text


def test_stream_stop_cancels_pending_model_initialization() -> None:
    hub = StreamHub()
    initialization_started = threading.Event()
    release_initialization = threading.Event()

    def blocking_initialization() -> SimpleNamespace:
        initialization_started.set()
        release_initialization.wait(timeout=2)
        return SimpleNamespace(zscore_baseline={})

    async def exercise() -> None:
        try:
            with patch(
                "app.monitoring.get_system",
                side_effect=blocking_initialization,
            ):
                await hub.start(0.5)
                assert await asyncio.wait_for(
                    asyncio.to_thread(initialization_started.wait),
                    timeout=1,
                )
                task = hub._task
                assert task is not None

                await asyncio.wait_for(hub.stop(), timeout=1)

                assert task.done()
                assert hub._task is None
                assert not hub.running
        finally:
            release_initialization.set()

    asyncio.run(exercise())
