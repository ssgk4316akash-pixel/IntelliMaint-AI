from datetime import datetime, timedelta, timezone

from app.database import (
    AssessmentRecord,
    Base,
    configure_database,
    dispose_database,
    get_assessment_by_id,
    list_assessment_pages,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


def test_database_model_creation_and_assessment_persistence_with_sqlite() -> None:
    dispose_database()


def test_assessment_analytics_empty_database() -> None:
    dispose_database()
    assert configure_database("sqlite+pysqlite:///:memory:")

    from app.database import get_assessment_analytics

    analytics = get_assessment_analytics()
    assert analytics is not None
    assert analytics["total_assessments"] == 0
    assert analytics["predicted_failures"] == 0
    assert analytics["monitored_assets"] == 0
    assert analytics["activity_by_day"] == []
    assert analytics["asset_comparison"] == []

    dispose_database()


def test_assessment_analytics_aggregate_filters_and_distributions() -> None:
    dispose_database()
    assert configure_database("sqlite+pysqlite:///:memory:")

    from app.database import get_assessment_analytics, save_assessment

    save_assessment(
        source="manual",
        asset_id="asset-a",
        kind="machine_assessment",
        reading={"torque": 40.0},
        prediction={"predicted_failure": True},
        anomaly={"is_anomaly": True},
        zscore={"is_anomaly": False},
        recommendation={"severity": "critical"},
        simulated=False,
    )
    save_assessment(
        source="simulator",
        asset_id="asset-a",
        kind="machine_assessment",
        reading={"torque": 20.0},
        prediction={"predicted_failure": False},
        anomaly={"is_anomaly": False},
        zscore={"is_anomaly": True},
        recommendation={"severity": "warning"},
        simulated=True,
    )
    save_assessment(
        source="manual",
        asset_id="asset-b",
        kind="machine_assessment",
        reading={"torque": 30.0},
        prediction={"predicted_failure": False},
        anomaly={"is_anomaly": False},
        zscore={"is_anomaly": False},
        recommendation={"severity": "routine"},
        simulated=False,
    )

    analytics = get_assessment_analytics()
    assert analytics is not None
    assert analytics["total_assessments"] == 3
    assert analytics["predicted_failures"] == 1
    assert analytics["normal_assessments"] == 2
    assert analytics["anomaly_assessments"] == 2
    assert analytics["urgent_critical"] == 1
    assert analytics["monitored_assets"] == 2
    assert analytics["simulated_assessments"] == 1
    assert analytics["severity_distribution"]["critical"] == 1
    assert analytics["severity_distribution"]["warning"] == 1

    asset_a = get_assessment_analytics(asset_id="asset-a")
    assert asset_a is not None
    assert asset_a["total_assessments"] == 2
    assert asset_a["monitored_assets"] == 1

    dispose_database()
    assert configure_database("sqlite+pysqlite:///:memory:")

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    with factory() as session:
        row = AssessmentRecord(
            created_at=now,
            source="manual",
            asset_id="pump-1",
            kind="machine_assessment",
            reading={"torque": 40.0},
            prediction={"failure_score": 0.2},
            anomaly={"is_anomaly": False},
            zscore={"is_anomaly": False},
            recommendation={"severity": "low"},
            simulated=False,
        )
        session.add(row)
        session.commit()
        assessment_id = row.id

        assert assessment_id is not None
        assert session.scalar(
            select(AssessmentRecord).where(AssessmentRecord.asset_id == "pump-1")
        ).asset_id == "pump-1"

    dispose_database()


def test_database_configuration_handles_unavailable_postgresql() -> None:
    dispose_database()
    assert configure_database("postgresql+psycopg://invalid:invalid@127.0.0.1:1/invalid") is False
    dispose_database()


def test_assessment_page_query_helpers() -> None:
    dispose_database()
    assert configure_database("sqlite+pysqlite:///:memory:")

    now = datetime.now(timezone.utc)
    saved = []
    for index in range(5):
        from app.database import save_assessment

        saved.append(save_assessment(
            source="manual",
            asset_id=f"asset-{index % 2}",
            kind="machine_assessment",
            reading={"torque": float(index)},
            prediction={"failure_score": float(index) / 10},
            zscore={"is_anomaly": False},
            simulated=False,
        ))

    assert all(item is not None for item in saved)
    first_id = saved[0]["id"]
    assert get_assessment_by_id(first_id)["id"] == first_id

    recent = list_assessment_pages(limit=2, offset=0)
    assert recent is not None
    assert len(recent) == 2

    asset_rows = list_assessment_pages(limit=20, asset_id="asset-0")
    assert asset_rows is not None
    assert all(row["asset_id"] == "asset-0" for row in asset_rows)

    ranged = list_assessment_pages(
        limit=20,
        start=now - timedelta(minutes=1),
        end=now + timedelta(minutes=1),
    )
    assert ranged is not None

    dispose_database()
