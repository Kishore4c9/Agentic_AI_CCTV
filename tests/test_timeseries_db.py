"""Unit tests for the TimeSeriesDB and TimeSeriesDBSubscriber."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    HeartbeatMessage,
    StructuredEvent,
)
from agentic_cctv.timeseries_db import TimeSeriesDB, TimeSeriesDBSubscriber


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> TimeSeriesDB:
    """Create an in-memory TimeSeriesDB for testing."""
    tsdb = TimeSeriesDB(":memory:")
    yield tsdb
    tsdb.close()


def _make_event(
    event_id: str = "evt-001",
    camera_id: str = "cam-lobby-01",
    tenant_id: str = "tenant-acme",
    site_id: str = "site-hq",
    object_type: str = "person",
    confidence: float = 0.92,
) -> StructuredEvent:
    return StructuredEvent(
        event_id=event_id,
        camera_id=camera_id,
        tenant_id=tenant_id,
        site_id=site_id,
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        object_type=object_type,
        track_id="trk-a1b2c3d4",
        bounding_box=BoundingBox(x=120, y=80, width=200, height=400),
        confidence=confidence,
        frame_crop="base64data",
    )


def _make_alert(
    alert_id: str = "alert-001",
    event_id: str = "evt-001",
) -> AlertPayload:
    return AlertPayload(
        alert_id=alert_id,
        event_id=event_id,
        camera_id="cam-lobby-01",
        tenant_id="tenant-acme",
        site_id="site-hq",
        timestamp=datetime(2025, 1, 15, 14, 30, 1, tzinfo=timezone.utc),
        alert_type="intrusion",
        description="Person detected in restricted area",
        threat_level="medium",
        frame_crop_url=None,
    )


def _make_heartbeat() -> HeartbeatMessage:
    return HeartbeatMessage(
        camera_id="cam-lobby-01",
        tenant_id="tenant-acme",
        site_id="site-hq",
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        cpu_percent=45.2,
        memory_percent=62.1,
        temperature_celsius=68.5,
        inference_latency_ms=35.2,
        gpu_utilization_percent=78.0,
    )


# ---------------------------------------------------------------------------
# Schema / Table creation tests
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    """Verify that all tables and indexes are created."""

    def test_events_table_exists(self, db: TimeSeriesDB) -> None:
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        )
        assert cursor.fetchone() is not None

    def test_alerts_table_exists(self, db: TimeSeriesDB) -> None:
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
        )
        assert cursor.fetchone() is not None

    def test_heartbeats_table_exists(self, db: TimeSeriesDB) -> None:
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='heartbeats'"
        )
        assert cursor.fetchone() is not None

    def test_rule_sets_table_exists(self, db: TimeSeriesDB) -> None:
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rule_sets'"
        )
        assert cursor.fetchone() is not None

    def test_indexes_created(self, db: TimeSeriesDB) -> None:
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        index_names = {row["name"] for row in cursor.fetchall()}
        expected = {
            "idx_events_camera_time",
            "idx_events_tenant",
            "idx_alerts_camera_time",
            "idx_heartbeats_camera_time",
            "idx_rule_sets_camera",
        }
        assert expected.issubset(index_names)


# ---------------------------------------------------------------------------
# Insert / query round-trip tests
# ---------------------------------------------------------------------------


class TestEventInsertAndQuery:
    """Test event insert and retrieval."""

    def test_insert_and_get_event(self, db: TimeSeriesDB) -> None:
        event = _make_event()
        db.insert_event(event)

        rows = db.get_events()
        assert len(rows) == 1
        row = rows[0]
        assert row["event_id"] == "evt-001"
        assert row["camera_id"] == "cam-lobby-01"
        assert row["tenant_id"] == "tenant-acme"
        assert row["object_type"] == "person"
        assert row["confidence"] == pytest.approx(0.92)
        assert row["detection_gate_passed"] == 1
        assert row["context_gate_passed"] == 0

    def test_bounding_box_stored_as_json(self, db: TimeSeriesDB) -> None:
        event = _make_event()
        db.insert_event(event)

        rows = db.get_events()
        bbox = json.loads(rows[0]["bounding_box"])
        assert bbox == {"x": 120, "y": 80, "width": 200, "height": 400}

    def test_gate_flags_persisted(self, db: TimeSeriesDB) -> None:
        event = _make_event()
        db.insert_event(event, detection_gate_passed=True, context_gate_passed=True)

        rows = db.get_events()
        assert rows[0]["detection_gate_passed"] == 1
        assert rows[0]["context_gate_passed"] == 1

    def test_filter_by_camera_id(self, db: TimeSeriesDB) -> None:
        db.insert_event(_make_event(event_id="e1", camera_id="cam-a"))
        db.insert_event(_make_event(event_id="e2", camera_id="cam-b"))

        rows = db.get_events(camera_id="cam-a")
        assert len(rows) == 1
        assert rows[0]["camera_id"] == "cam-a"

    def test_filter_by_tenant_id(self, db: TimeSeriesDB) -> None:
        db.insert_event(_make_event(event_id="e1", tenant_id="t1"))
        db.insert_event(_make_event(event_id="e2", tenant_id="t2"))

        rows = db.get_events(tenant_id="t1")
        assert len(rows) == 1
        assert rows[0]["tenant_id"] == "t1"

    def test_limit_parameter(self, db: TimeSeriesDB) -> None:
        for i in range(5):
            db.insert_event(_make_event(event_id=f"e{i}"))

        rows = db.get_events(limit=3)
        assert len(rows) == 3

    def test_combined_camera_and_tenant_filter(self, db: TimeSeriesDB) -> None:
        db.insert_event(_make_event(event_id="e1", camera_id="cam-a", tenant_id="t1"))
        db.insert_event(_make_event(event_id="e2", camera_id="cam-a", tenant_id="t2"))
        db.insert_event(_make_event(event_id="e3", camera_id="cam-b", tenant_id="t1"))

        rows = db.get_events(camera_id="cam-a", tenant_id="t1")
        assert len(rows) == 1
        assert rows[0]["event_id"] == "e1"


class TestAlertInsertAndQuery:
    """Test alert insert and retrieval."""

    def test_insert_and_get_alert(self, db: TimeSeriesDB) -> None:
        # Insert the referenced event first (foreign key)
        db.insert_event(_make_event())
        alert = _make_alert()
        db.insert_alert(alert, delivered_channels=["push", "webhook"])

        rows = db.get_alerts()
        assert len(rows) == 1
        row = rows[0]
        assert row["alert_id"] == "alert-001"
        assert row["event_id"] == "evt-001"
        assert row["alert_type"] == "intrusion"
        assert row["threat_level"] == "medium"
        channels = json.loads(row["delivered_channels"])
        assert channels == ["push", "webhook"]

    def test_cooldown_suppressed_count(self, db: TimeSeriesDB) -> None:
        db.insert_event(_make_event())
        alert = _make_alert()
        db.insert_alert(alert, delivered_channels=["push"], cooldown_suppressed_count=5)

        rows = db.get_alerts()
        assert rows[0]["cooldown_suppressed_count"] == 5

    def test_filter_alerts_by_camera(self, db: TimeSeriesDB) -> None:
        db.insert_event(_make_event(event_id="e1", camera_id="cam-a"))
        db.insert_event(_make_event(event_id="e2", camera_id="cam-b"))

        # Insert alert for cam-a
        db._conn.execute(
            """INSERT INTO alerts (alert_id, event_id, camera_id, tenant_id,
               alert_type, threat_level, delivered_channels)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("a1", "e1", "cam-a", "tenant-acme", "intrusion", "high", '["push"]'),
        )
        # Insert alert for cam-b
        db._conn.execute(
            """INSERT INTO alerts (alert_id, event_id, camera_id, tenant_id,
               alert_type, threat_level, delivered_channels)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("a2", "e2", "cam-b", "tenant-acme", "intrusion", "low", "[]"),
        )
        db._conn.commit()

        rows = db.get_alerts(camera_id="cam-a")
        assert len(rows) == 1
        assert rows[0]["alert_id"] == "a1"


class TestHeartbeatInsert:
    """Test heartbeat insert."""

    def test_insert_heartbeat(self, db: TimeSeriesDB) -> None:
        hb = _make_heartbeat()
        db.insert_heartbeat(hb)

        cursor = db._conn.execute("SELECT * FROM heartbeats")
        rows = [dict(r) for r in cursor.fetchall()]
        assert len(rows) == 1
        row = rows[0]
        assert row["camera_id"] == "cam-lobby-01"
        assert row["cpu_percent"] == pytest.approx(45.2)
        assert row["memory_percent"] == pytest.approx(62.1)
        assert row["temperature_celsius"] == pytest.approx(68.5)
        assert row["inference_latency_ms"] == pytest.approx(35.2)
        assert row["gpu_utilization_percent"] == pytest.approx(78.0)

    def test_heartbeat_nullable_fields(self, db: TimeSeriesDB) -> None:
        hb = HeartbeatMessage(
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, tzinfo=timezone.utc),
            cpu_percent=10.0,
            memory_percent=20.0,
            temperature_celsius=None,
            inference_latency_ms=5.0,
            gpu_utilization_percent=None,
        )
        db.insert_heartbeat(hb)

        cursor = db._conn.execute("SELECT * FROM heartbeats")
        row = dict(cursor.fetchone())
        assert row["temperature_celsius"] is None
        assert row["gpu_utilization_percent"] is None


# ---------------------------------------------------------------------------
# TimeSeriesDBSubscriber tests
# ---------------------------------------------------------------------------


class TestTimeSeriesDBSubscriber:
    """Test the MQTT subscriber callback."""

    def test_subscriber_persists_valid_event(self, db: TimeSeriesDB) -> None:
        subscriber = TimeSeriesDBSubscriber(db)
        payload = json.dumps(
            {
                "event_id": "evt-sub-001",
                "camera_id": "cam-lobby-01",
                "tenant_id": "tenant-acme",
                "site_id": "site-hq",
                "timestamp": "2025-01-15T14:30:00+00:00",
                "object_type": "person",
                "track_id": "trk-001",
                "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 200},
                "confidence": 0.85,
                "frame_crop": "base64data",
            }
        ).encode()

        subscriber("tenant-acme/site-hq/cam-lobby-01/events", payload, 1)

        rows = db.get_events()
        assert len(rows) == 1
        assert rows[0]["event_id"] == "evt-sub-001"
        assert rows[0]["confidence"] == pytest.approx(0.85)

    def test_subscriber_handles_z_suffix_timestamp(self, db: TimeSeriesDB) -> None:
        subscriber = TimeSeriesDBSubscriber(db)
        payload = json.dumps(
            {
                "event_id": "evt-z-001",
                "camera_id": "cam-01",
                "tenant_id": "t1",
                "site_id": "s1",
                "timestamp": "2025-01-15T14:30:00.123Z",
                "object_type": "vehicle",
                "track_id": "trk-002",
                "bounding_box": {"x": 0, "y": 0, "width": 50, "height": 50},
                "confidence": 0.7,
                "frame_crop": "",
            }
        ).encode()

        subscriber("t1/s1/cam-01/events", payload, 1)

        rows = db.get_events()
        assert len(rows) == 1
        assert rows[0]["event_id"] == "evt-z-001"

    def test_subscriber_ignores_invalid_json(self, db: TimeSeriesDB) -> None:
        subscriber = TimeSeriesDBSubscriber(db)
        subscriber("topic", b"not-json", 1)

        rows = db.get_events()
        assert len(rows) == 0

    def test_subscriber_ignores_missing_fields(self, db: TimeSeriesDB) -> None:
        subscriber = TimeSeriesDBSubscriber(db)
        payload = json.dumps({"event_id": "partial"}).encode()
        subscriber("topic", payload, 1)

        rows = db.get_events()
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Close / lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Test database lifecycle."""

    def test_close_prevents_further_operations(self) -> None:
        tsdb = TimeSeriesDB(":memory:")
        tsdb.close()
        with pytest.raises(Exception):
            tsdb.get_events()
