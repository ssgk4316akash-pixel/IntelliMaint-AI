import asyncio
from pathlib import Path
from datetime import datetime, timezone

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.database import (
    count_assessment_pages,
    database_status,
    get_assessment_analytics,
    get_assessment_by_id,
    list_assessment_pages,
    list_assessments,
)
from app.ml import (
    DATA_PATH,
    TrainedSystem,
    make_anomaly_prediction,
    make_explanation,
    make_prediction,
    make_zscore_prediction,
)
from app.monitoring import (
    assess_machine_reading,
    assess_rul_reading,
    encode_sse,
    hub,
    record_anomaly,
    record_failure_prediction,
)
from app.rul import RulSystem
from app.runtime import get_rul_system, get_system
from app.schemas import MachineAssessmentRequest, RulReading, SensorReading, StreamStartRequest


app = FastAPI(
    title="IntelliMaint AI",
    description="Benchmark-based predictive maintenance research prototype.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _system_or_http_error() -> TrainedSystem:
    try:
        return get_system()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _assess_or_http_error(
    reading: dict[str, object], source: str, asset_id: str
) -> dict[str, object]:
    _system_or_http_error()
    return assess_machine_reading(reading, source, asset_id)


def _rul_system_or_http_error() -> RulSystem:
    try:
        return get_rul_system()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, object]:
    database = database_status()
    return {
        "status": "ok",
        "dataset_available": Path(DATA_PATH).is_file(),
        "dataset": "AI4I 2020",
        "database_connected": database["connected"],
        "database_target": database["target"],
    }


@app.get("/api/overview")
def overview() -> dict[str, object]:
    system = _system_or_http_error()
    return {
        "dataset": system.dataset_summary,
        "selected_model": system.selected_model_name,
        "cross_validation_f1": system.cross_validation_f1,
        "classification_metrics": system.metrics,
        "anomaly_metrics": system.anomaly_metrics,
        "feature_importance": system.feature_importance,
    }


@app.post("/api/predict")
def predict(reading: SensorReading) -> dict[str, object]:
    system = _system_or_http_error()
    payload = reading.model_dump()
    result = make_prediction(system, payload)
    record_failure_prediction(payload, result)
    return result


@app.post(
    "/api/explain",
    description=(
        "Explain the selected classifier's uncalibrated class-1 score. Positive "
        "SHAP values increase it relative to the baseline; negative values decrease it."
    ),
)
def explain(reading: SensorReading) -> dict[str, object]:
    system = _system_or_http_error()
    return make_explanation(system, reading.model_dump())


@app.post("/api/anomaly")
def detect_anomaly(reading: SensorReading) -> dict[str, object]:
    system = _system_or_http_error()
    payload = reading.model_dump()
    result = make_anomaly_prediction(system, payload)
    record_anomaly(payload, result)
    return result


@app.post("/api/zscore")
def detect_zscore_anomaly(reading: SensorReading) -> dict[str, object]:
    system = _system_or_http_error()
    try:
        return make_zscore_prediction(system, reading.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/assess")
async def assess(reading: MachineAssessmentRequest) -> dict[str, object]:
    payload = reading.model_dump()
    asset_id = str(payload.pop("asset_id"))
    event = await asyncio.to_thread(_assess_or_http_error, payload, "manual", asset_id)
    await hub.publish(event)
    return event


@app.post("/api/stream/ingest")
async def ingest(reading: MachineAssessmentRequest) -> dict[str, object]:
    payload = reading.model_dump()
    asset_id = str(payload.pop("asset_id"))
    event = await asyncio.to_thread(_assess_or_http_error, payload, "ingest", asset_id)
    await hub.publish(event)
    return event


@app.get("/api/history")
def history(
    limit: int = Query(default=50, ge=1, le=200),
    kind: str | None = Query(default=None, max_length=32),
) -> dict[str, object]:
    database = database_status()
    items = list_assessments(limit=limit, kind=kind) if database["connected"] else None
    return {
        "database_connected": database["connected"],
        "database_target": database["target"],
        "database_error": database["error"],
        "items": items or [],
    }


def _parse_history_datetime(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be an ISO-8601 datetime.",
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@app.get("/api/assessments")
def assessment_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    asset_id: str | None = Query(default=None, max_length=64),
    kind: str | None = Query(default=None, max_length=32),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
) -> dict[str, object]:
    database = database_status()
    if not database["connected"]:
        return {
            "database_connected": False,
            "database_target": database["target"],
            "database_error": database["error"],
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
        }

    start_datetime = _parse_history_datetime(start, "start")
    end_datetime = _parse_history_datetime(end, "end")
    if start_datetime is not None and end_datetime is not None and start_datetime > end_datetime:
        raise HTTPException(
            status_code=422,
            detail="start must be earlier than or equal to end.",
        )

    items = list_assessment_pages(
        limit=limit,
        offset=offset,
        asset_id=asset_id,
        kind=kind,
        start=start_datetime,
        end=end_datetime,
    )
    total = count_assessment_pages(
        asset_id=asset_id,
        kind=kind,
        start=start_datetime,
        end=end_datetime,
    )
    if items is None or total is None:
        return {
            "database_connected": False,
            "database_target": database["target"],
            "database_error": database_status()["error"],
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
        }
    return {
        "database_connected": True,
        "database_target": database["target"],
        "database_error": None,
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/analytics")
def assessment_analytics(
    asset_id: str | None = Query(default=None, max_length=64),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
) -> dict[str, object]:
    database = database_status()
    if not database["connected"]:
        return {
            "database_connected": False,
            "database_target": database["target"],
            "database_error": database["error"],
            "analytics": None,
        }

    start_datetime = _parse_history_datetime(start, "start")
    end_datetime = _parse_history_datetime(end, "end")
    if start_datetime is not None and end_datetime is not None and start_datetime > end_datetime:
        raise HTTPException(
            status_code=422,
            detail="start must be earlier than or equal to end.",
        )

    analytics = get_assessment_analytics(
        asset_id=asset_id,
        start=start_datetime,
        end=end_datetime,
    )
    if analytics is None:
        return {
            "database_connected": False,
            "database_target": database["target"],
            "database_error": database_status()["error"],
            "analytics": None,
        }
    return {
        "database_connected": True,
        "database_target": database["target"],
        "database_error": None,
        "analytics": analytics,
    }


@app.get("/api/assessments/{assessment_id}")
def assessment_detail(assessment_id: int) -> dict[str, object]:
    database = database_status()
    if not database["connected"]:
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable.",
        )
    item = get_assessment_by_id(assessment_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return item


@app.get("/api/stream/status")
def stream_status() -> dict[str, object]:
    return hub.status()


@app.post("/api/stream/start")
async def stream_start(
    body: StreamStartRequest | None = Body(default=None),
) -> dict[str, object]:
    interval = 2.0 if body is None else body.interval_seconds
    await hub.start(interval)
    return hub.status()


@app.post("/api/stream/stop")
async def stream_stop() -> dict[str, object]:
    await hub.stop()
    return hub.status()


@app.post("/api/simulation/generate")
async def generate_simulation_reading(
    mode: str = Query(default="mixed", max_length=16),
) -> dict[str, object]:
    try:
        return await hub.generate(mode)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/simulation/latest")
def latest_simulation_reading() -> dict[str, object]:
    if hub.latest is None:
        return {"simulated": True, "available": False, "reading": None}
    return {"simulated": True, "available": True, "reading": hub.latest}


@app.get("/api/simulation/history")
def simulation_history(limit: int = Query(default=50, ge=1, le=100)) -> dict[str, object]:
    return {
        "simulated": True,
        "count": len(hub.get_history(limit)),
        "items": hub.get_history(limit),
    }


@app.post("/api/simulation/reset")
def reset_simulation(seed: int = Query(default=42, ge=0, le=2147483647)) -> dict[str, object]:
    hub.reset(seed)
    return {
        "simulated": True,
        "reset": True,
        "seed": seed,
        "history_count": 0,
    }


@app.get("/api/stream/events")
async def stream_events() -> StreamingResponse:
    queue = hub.subscribe()

    async def generate():
        try:
            if hub.latest is not None:
                yield encode_sse(hub.latest)
            while True:
                event = await queue.get()
                yield encode_sse(event)
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/rul/overview")
def rul_overview() -> dict[str, object]:
    system = _rul_system_or_http_error()
    return {
        "dataset": system.dataset_summary,
        "selected_model": system.selected_model_name,
        "validation_metrics": system.validation_metrics,
        "test_metrics": system.test_metrics,
        "methodology": {
            "training_target": (
                "Capped RUL = min(125, maximum observed cycle for engine "
                "- current cycle)"
            ),
            "test_target": (
                "Official RUL_FD001.txt value matched to each test engine's "
                "last observed cycle; labels remain uncapped"
            ),
            "test_evaluation": (
                "Only the last observed row of each test engine is used"
            ),
            "nasa_score": (
                "Sum of exp(-d/13)-1 for negative d and exp(d/10)-1 "
                "for positive d, where d = predicted RUL - actual RUL"
            ),
        },
    }


@app.post("/api/rul/predict")
def rul_predict(reading: RulReading) -> dict[str, object]:
    _rul_system_or_http_error()
    return assess_rul_reading(reading.model_dump(), "demo-engine")


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
