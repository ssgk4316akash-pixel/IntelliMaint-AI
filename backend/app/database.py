from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import Boolean, DateTime, Index, Integer, String, create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import JSON


logger = logging.getLogger(__name__)
RETRY_SECONDS = 15.0

_lock = threading.Lock()
_engine: Engine | None = None
_session_factory: sessionmaker | None = None
_connected = False
_last_error: str | None = None
_next_retry = 0.0
_logged_disconnect = False


class Base(DeclarativeBase):
    pass


class AssessmentRecord(Base):
    __tablename__ = "assessments"
    __table_args__ = (
        Index("ix_assessments_asset_created", "asset_id", "created_at"),
        Index("ix_assessments_kind_created", "kind", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    asset_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    reading: Mapped[dict[str, Any]] = mapped_column(JSON)
    prediction: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    anomaly: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    rul: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    zscore: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


def database_target(url: str | None = None) -> str:
    raw = url or os.environ.get("DATABASE_URL")  if not raw:     return "Database not configured"
    parts = urlsplit(raw)
    if parts.password is None:
        return raw
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    username = parts.username or ""
    netloc = f"{username}:***@{host}" if username else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def configure_database(url: str | None = None) -> bool:
    global _engine, _session_factory, _connected, _last_error, _next_retry, _logged_disconnect
    raw_url = url or os.environ.get("DATABASE_URL")  if not raw_url:     _mark_disconnected(         RuntimeError("DATABASE_URL is not configured")     )     return False
    engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if raw_url.startswith("postgresql"):
        engine_kwargs["connect_args"] = {"connect_timeout": 3}

    engine = None
    try:
        engine = create_engine(raw_url, **engine_kwargs)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        Base.metadata.create_all(engine)
        _ensure_assessment_columns(engine)
    except Exception as error:
        if engine is not None:
            engine.dispose()
        _mark_disconnected(error)
        _next_retry = time.monotonic() + RETRY_SECONDS
        return False

    with _lock:
        previous = _engine
        _engine = engine
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        _connected = True
        _last_error = None
        _next_retry = 0.0
        _logged_disconnect = False
    if previous is not None and previous is not engine:
        previous.dispose()
    return True


def _ensure_assessment_columns(engine: Engine) -> None:
    dialect = engine.dialect.name
    with engine.begin() as connection:
        if dialect == "sqlite":
            existing = {
                row[1]
                for row in connection.execute(text("PRAGMA table_info(assessments)")).all()
            }
            if "zscore" not in existing:
                connection.execute(text("ALTER TABLE assessments ADD COLUMN zscore JSON"))
            if "simulated" not in existing:
                connection.execute(
                    text("ALTER TABLE assessments ADD COLUMN simulated BOOLEAN DEFAULT 0")
                )
        elif dialect == "postgresql":
            existing = set(
                connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'assessments'"
                    )
                ).scalars()
            )
            if "zscore" not in existing:
                connection.execute(
                    text("ALTER TABLE assessments ADD COLUMN zscore JSON")
                )
            if "simulated" not in existing:
                connection.execute(
                    text(
                        "ALTER TABLE assessments "
                        "ADD COLUMN simulated BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                )


def dispose_database() -> None:
    global _engine, _session_factory, _connected, _last_error, _next_retry, _logged_disconnect
    with _lock:
        engine = _engine
        _engine = None
        _session_factory = None
        _connected = False
        _last_error = None
        _next_retry = 0.0
        _logged_disconnect = False
    if engine is not None:
        engine.dispose()


def database_status() -> dict[str, Any]:
    ensure_database()
    with _lock:
        connected = _connected
        error = _last_error
    return {
        "connected": connected,
        "target": database_target(),
        "error": None if connected else error,
    }


def ensure_database() -> bool:
    with _lock:
        if _connected and _session_factory is not None:
            return True
        if time.monotonic() < _next_retry:
            return False
    return configure_database()


def save_assessment(
    *,
    source: str,
    asset_id: str,
    kind: str,
    reading: dict[str, Any],
    prediction: dict[str, Any] | None = None,
    anomaly: dict[str, Any] | None = None,
    recommendation: dict[str, Any] | None = None,
    rul: dict[str, Any] | None = None,
    zscore: dict[str, Any] | None = None,
    simulated: bool = False,
) -> dict[str, Any] | None:
    if not ensure_database():
        return None
    with _lock:
        factory = _session_factory
    if factory is None:
        return None

    session = factory()
    try:
        row = AssessmentRecord(
            created_at=datetime.now(timezone.utc),
            source=source,
            asset_id=asset_id,
            kind=kind,
            reading=reading,
            prediction=prediction,
            anomaly=anomaly,
            recommendation=recommendation,
            rul=rul,
            zscore=zscore,
            simulated=simulated,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _row_to_dict(row)
    except Exception as error:
        session.rollback()
        _mark_disconnected(error)
        return None
    finally:
        session.close()


def list_assessments(
    *,
    limit: int = 50,
    kind: str | None = None,
) -> list[dict[str, Any]] | None:
    if not ensure_database():
        return None
    with _lock:
        factory = _session_factory
    if factory is None:
        return None

    session = factory()
    try:
        query = select(AssessmentRecord).order_by(AssessmentRecord.id.desc()).limit(limit)
        if kind is not None:
            query = (
                select(AssessmentRecord)
                .where(AssessmentRecord.kind == kind)
                .order_by(AssessmentRecord.id.desc())
                .limit(limit)
            )
        rows = session.scalars(query).all()
        return [_row_to_dict(row) for row in rows]
    except Exception as error:
        session.rollback()
        _mark_disconnected(error)
        return None
    finally:
        session.close()


def get_assessment_by_id(assessment_id: int) -> dict[str, Any] | None:
    if not ensure_database():
        return None
    with _lock:
        factory = _session_factory
    if factory is None:
        return None

    session = factory()
    try:
        row = session.scalar(
            select(AssessmentRecord).where(AssessmentRecord.id == assessment_id)
        )
        return _row_to_dict(row) if row is not None else None
    except Exception as error:
        session.rollback()
        _mark_disconnected(error)
        return None
    finally:
        session.close()


def list_assessment_pages(
    *,
    limit: int = 50,
    offset: int = 0,
    asset_id: str | None = None,
    kind: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict[str, Any]] | None:
    if not ensure_database():
        return None
    with _lock:
        factory = _session_factory
    if factory is None:
        return None

    session = factory()
    try:
        query = select(AssessmentRecord)
        if asset_id is not None:
            query = query.where(AssessmentRecord.asset_id == asset_id)
        if kind is not None:
            query = query.where(AssessmentRecord.kind == kind)
        if start is not None:
            query = query.where(AssessmentRecord.created_at >= start)
        if end is not None:
            query = query.where(AssessmentRecord.created_at <= end)
        query = (
            query.order_by(AssessmentRecord.created_at.desc(), AssessmentRecord.id.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = session.scalars(query).all()
        return [_row_to_dict(row) for row in rows]
    except Exception as error:
        session.rollback()
        _mark_disconnected(error)
        return None
    finally:
        session.close()


def count_assessment_pages(
    *,
    asset_id: str | None = None,
    kind: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> int | None:
    if not ensure_database():
        return None
    with _lock:
        factory = _session_factory
    if factory is None:
        return None

    session = factory()
    try:
        query = select(func.count()).select_from(AssessmentRecord)
        if asset_id is not None:
            query = query.where(AssessmentRecord.asset_id == asset_id)
        if kind is not None:
            query = query.where(AssessmentRecord.kind == kind)
        if start is not None:
            query = query.where(AssessmentRecord.created_at >= start)
        if end is not None:
            query = query.where(AssessmentRecord.created_at <= end)
        return int(session.scalar(query) or 0)
    except Exception as error:
        session.rollback()
        _mark_disconnected(error)
        return None
    finally:
        session.close()


def get_assessment_analytics(
    *,
    asset_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any] | None:
    if not ensure_database():
        return None
    with _lock:
        factory = _session_factory
    if factory is None:
        return None

    session = factory()
    try:
        query = select(AssessmentRecord)
        if asset_id is not None:
            query = query.where(AssessmentRecord.asset_id == asset_id)
        if start is not None:
            query = query.where(AssessmentRecord.created_at >= start)
        if end is not None:
            query = query.where(AssessmentRecord.created_at <= end)

        rows = session.scalars(query).all()
        if not rows:
            return {
                "total_assessments": 0,
                "predicted_failures": 0,
                "normal_assessments": 0,
                "anomaly_assessments": 0,
                "urgent_critical": 0,
                "monitored_assets": 0,
                "simulated_assessments": 0,
                "persisted_assessments": len(rows),
                "activity_by_day": [],
                "failure_distribution": {"predicted_failure": 0, "normal": 0},
                "anomaly_distribution": {
                    "any_anomaly": 0,
                    "isolation_forest": 0,
                    "zscore": 0,
                    "normal": 0,
                },
                "severity_distribution": {
                    "routine": 0,
                    "warning": 0,
                    "urgent": 0,
                    "critical": 0,
                },
                "asset_comparison": [],
            }

        predicted_failures = 0
        normal_assessments = 0
        anomaly_assessments = 0
        urgent_critical = 0
        simulated_assessments = 0
        isolation_count = 0
        zscore_count = 0
        severity_counts = {"routine": 0, "warning": 0, "urgent": 0, "critical": 0}
        activity: dict[str, int] = {}
        asset_counts: dict[str, int] = {}

        for row in rows:
            prediction = row.prediction or {}
            isolation = row.anomaly or {}
            zscore = row.zscore or {}
            recommendation = row.recommendation or {}

            predicted_failure = bool(prediction.get("predicted_failure", False))
            isolation_flag = bool(isolation.get("is_anomaly", False))
            zscore_flag = bool(zscore.get("is_anomaly", False))
            severity = str(recommendation.get("severity", "routine")).lower()
            if severity not in severity_counts:
                severity = "routine"

            if predicted_failure:
                predicted_failures += 1
            else:
                normal_assessments += 1
            if isolation_flag or zscore_flag:
                anomaly_assessments += 1
            if isolation_flag:
                isolation_count += 1
            if zscore_flag:
                zscore_count += 1
            if severity in {"urgent", "critical"}:
                urgent_critical += 1
            severity_counts[severity] += 1
            if row.simulated:
                simulated_assessments += 1

            day = row.created_at.date().isoformat()
            activity[day] = activity.get(day, 0) + 1
            asset_counts[row.asset_id] = asset_counts.get(row.asset_id, 0) + 1

        return {
            "total_assessments": len(rows),
            "predicted_failures": predicted_failures,
            "normal_assessments": normal_assessments,
            "anomaly_assessments": anomaly_assessments,
            "urgent_critical": urgent_critical,
            "monitored_assets": len(asset_counts),
            "simulated_assessments": simulated_assessments,
            "persisted_assessments": len(rows),
            "activity_by_day": [
                {"date": day, "count": activity[day]}
                for day in sorted(activity)
            ],
            "failure_distribution": {
                "predicted_failure": predicted_failures,
                "normal": normal_assessments,
            },
            "anomaly_distribution": {
                "any_anomaly": anomaly_assessments,
                "isolation_forest": isolation_count,
                "zscore": zscore_count,
                "normal": len(rows) - anomaly_assessments,
            },
            "severity_distribution": severity_counts,
            "asset_comparison": [
                {"asset_id": asset_id, "count": asset_counts[asset_id]}
                for asset_id in sorted(asset_counts)
            ],
        }
    except Exception as error:
        session.rollback()
        _mark_disconnected(error)
        return None
    finally:
        session.close()


def count_assessments() -> int | None:
    if not ensure_database():
        return None
    with _lock:
        factory = _session_factory
    if factory is None:
        return None
    session = factory()
    try:
        return int(session.scalar(select(func.count()).select_from(AssessmentRecord)) or 0)
    except Exception as error:
        session.rollback()
        _mark_disconnected(error)
        return None
    finally:
        session.close()


def _mark_disconnected(error: Exception) -> None:
    global _connected, _last_error, _logged_disconnect
    message = str(error).strip() or error.__class__.__name__
    message = re.sub(r"(postgresql(?:\+\w+)?://)[^@\s]+@", r"\1***@", message)
    message = re.sub(r"password=[^;&,\s]+", "password=***", message, flags=re.I)
    if len(message) > 500:
        message = message[:500] + "..."
    with _lock:
        _connected = False
        _last_error = message
        should_log = not _logged_disconnect
        _logged_disconnect = True
    if should_log:
        logger.warning("PostgreSQL persistence is unavailable: %s", message)


def _row_to_dict(row: AssessmentRecord) -> dict[str, Any]:
    created_at = row.created_at
    return {
        "id": row.id,
        "created_at": created_at.isoformat() if created_at is not None else None,
        "source": row.source,
        "asset_id": row.asset_id,
        "kind": row.kind,
        "reading": row.reading,
        "prediction": row.prediction,
        "anomaly": row.anomaly,
        "recommendation": row.recommendation,
        "rul": row.rul,
        "zscore": row.zscore,
        "simulated": row.simulated,
        "persisted": True,
    }
