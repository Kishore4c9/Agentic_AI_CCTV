"""Unit tests for the RetentionScheduler and retention methods.

Tests cover:
- RetentionScheduler initialization with default and custom parameters
- run_retention() method: aggregation, purging raw events, purging aggregated data,
  purging orphaned embeddings
- Start/stop lifecycle of the background task
- Error handling (individual step failures don't block other steps)
- Integration between TimeSeriesDB retention methods and VectorDB purge
- 90-day raw event retention cutoff (Req 12.1)
- 365-day aggregated data retention cutoff (Req 12.2)
- VectorDB lifecycle-linking (Req 12.3)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from agentic_cctv.models import BoundingBox, StructuredEvent
from agentic_cctv.retention_scheduler import (
    RetentionScheduler,
    _DEFAULT_AGGREGATED_EVENTS_DAYS,
    _DEFAULT_INTERVAL_SECONDS,
    _DEFAULT_RAW_EVENTS_DAYS,
)
from agentic_cctv.timeseries_db import TimeSeriesDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_id: str = "evt-001",
    camera_id: str = "cam-lobby-01",
    tenant_id: str = "tenant-acme",
    site_id: str = "site-hq",
    object_type: str = "person",
    confidence: float = 0.92,
    timestamp: Optional[datetime] = None,
) -> StructuredEvent:
    if timestamp is None:
        timestamp = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    return StructuredEvent(
        event_id=event_id,
        camera_id=camera_id,
        tenant_id=tenant_id,
        site_id=site_id,
        timestamp=timestamp,
        object_type=object_type,
        track_id="trk-a1b2c3d4",
        bounding_box=BoundingBox(x=120, y=80, width=200, height=400),
        confidence=confidence,
        frame_crop="base64data",
    )


@pytest.fixture
def db() -> TimeSeriesDB:
    """Create an in-memory TimeSeriesDB for testing."""
    tsdb = TimeSeriesDB(":memory:")
    yield tsdb
    tsdb.close()


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestRetentionSchedulerInit:
    """Test RetentionScheduler initialization with default and custom params."""

    def test_default_parameters(self, db: TimeSeriesDB) -> None:
        scheduler = RetentionScheduler(timeseries_db=db)
        assert scheduler._raw_events_days == _DEFAULT_RAW_EVENTS_DAYS
        assert scheduler._aggregated_events_days == _DEFAULT_AGGREGATED_EVENTS_DAYS
        assert scheduler._interval_seconds == _DEFAULT_INTERVAL_SECONDS
        assert scheduler._tsdb is db
        assert scheduler._vdb is None
        assert scheduler._task is None
        assert scheduler.running is False

    def test_custom_parameters(self, db: TimeSeriesDB) -> None:
        mock_vdb = MagicMock()
        scheduler = RetentionScheduler(
            timeseries_db=db,
            vector_db=mock_vdb,
            raw_events_days=30,
            aggregated_events_days=180,
            interval_seconds=3600,
        )
        assert scheduler._raw_events_days == 30
        assert scheduler._aggregated_events_days == 180
        assert scheduler._interval_seconds == 3600
        assert scheduler._vdb is mock_vdb

    def test_default_raw_events_days_is_90(self) -> None:
        """Validates: Requirement 12.1 — raw events retained for 90 days."""
        assert _DEFAULT_RAW_EVENTS_DAYS == 90

    def test_default_aggregated_events_days_is_365(self) -> None:
        """Validates: Requirement 12.2 — aggregated data retained for 1 year."""
        assert _DEFAULT_AGGREGATED_EVENTS_DAYS == 365


# ---------------------------------------------------------------------------
# run_retention() tests
# ---------------------------------------------------------------------------


class TestRunRetention:
    """Test the run_retention() method end-to-end."""

    async def test_run_retention_returns_summary_dict(self, db: TimeSeriesDB) -> None:
        scheduler = RetentionScheduler(timeseries_db=db)
        results = await scheduler.run_retention()
        assert isinstance(results, dict)
        assert "aggregated_groups" in results
        assert "purged_raw_events" in results
        assert "purged_aggregated" in results
        assert "purged_embeddings" in results

    async def test_run_retention_no_data(self, db: TimeSeriesDB) -> None:
        """With no data, all counts should be zero."""
        scheduler = RetentionScheduler(timeseries_db=db)
        results = await scheduler.run_retention()
        assert results["aggregated_groups"] == 0
        assert results["purged_raw_events"] == 0
        assert results["purged_aggregated"] == 0
        assert results["purged_embeddings"] == 0

    async def test_aggregates_old_events(self, db: TimeSeriesDB) -> None:
        """Events older than 90 days should be aggregated into daily summaries."""
        old_ts = datetime.utcnow() - timedelta(days=100)
        db.insert_event(_make_event(event_id="old-1", timestamp=old_ts, object_type="person"))
        db.insert_event(_make_event(event_id="old-2", timestamp=old_ts, object_type="person"))

        scheduler = RetentionScheduler(timeseries_db=db)
        results = await scheduler.run_retention()

        assert results["aggregated_groups"] >= 1

        # Verify aggregated data exists
        agg_rows = db.get_aggregated_events()
        assert len(agg_rows) >= 1
        # The aggregated row should have event_count of 2
        assert agg_rows[0]["event_count"] == 2

    async def test_purges_old_raw_events(self, db: TimeSeriesDB) -> None:
        """Raw events older than 90 days should be purged after aggregation."""
        old_ts = datetime.utcnow() - timedelta(days=100)
        recent_ts = datetime.utcnow() - timedelta(days=10)

        db.insert_event(_make_event(event_id="old-1", timestamp=old_ts))
        db.insert_event(_make_event(event_id="recent-1", timestamp=recent_ts))

        scheduler = RetentionScheduler(timeseries_db=db)
        results = await scheduler.run_retention()

        assert results["purged_raw_events"] == 1

        # Only the recent event should remain
        remaining = db.get_events()
        assert len(remaining) == 1
        assert remaining[0]["event_id"] == "recent-1"

    async def test_purges_old_aggregated_data(self, db: TimeSeriesDB) -> None:
        """Aggregated data older than 365 days should be purged."""
        # Insert aggregated data directly — one old, one recent
        old_date = (datetime.utcnow() - timedelta(days=400)).strftime("%Y-%m-%d")
        recent_date = (datetime.utcnow() - timedelta(days=100)).strftime("%Y-%m-%d")

        db._conn.execute(
            """INSERT INTO aggregated_events
               (date, camera_id, tenant_id, site_id, object_type, event_count, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (old_date, "cam-1", "t1", "s1", "person", 50, 0.85),
        )
        db._conn.execute(
            """INSERT INTO aggregated_events
               (date, camera_id, tenant_id, site_id, object_type, event_count, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (recent_date, "cam-1", "t1", "s1", "person", 30, 0.90),
        )
        db._conn.commit()

        scheduler = RetentionScheduler(timeseries_db=db)
        results = await scheduler.run_retention()

        assert results["purged_aggregated"] == 1

        # Only the recent aggregated row should remain
        agg_rows = db.get_aggregated_events()
        assert len(agg_rows) == 1
        assert agg_rows[0]["date"] == recent_date

    async def test_purges_orphaned_embeddings(self, db: TimeSeriesDB) -> None:
        """VectorDB embeddings without matching events should be purged."""
        # Insert a recent event so it stays
        recent_ts = datetime.utcnow() - timedelta(days=10)
        db.insert_event(_make_event(event_id="recent-1", timestamp=recent_ts))

        mock_vdb = MagicMock()
        mock_vdb.purge_orphaned_embeddings.return_value = 3

        scheduler = RetentionScheduler(timeseries_db=db, vector_db=mock_vdb)
        results = await scheduler.run_retention()

        # VectorDB purge should have been called with valid IDs
        mock_vdb.purge_orphaned_embeddings.assert_called_once()
        call_args = mock_vdb.purge_orphaned_embeddings.call_args[0][0]
        assert "recent-1" in call_args
        assert results["purged_embeddings"] == 3

    async def test_skips_vector_db_when_none(self, db: TimeSeriesDB) -> None:
        """When vector_db is None, VectorDB retention is skipped."""
        scheduler = RetentionScheduler(timeseries_db=db, vector_db=None)
        results = await scheduler.run_retention()
        assert results["purged_embeddings"] == 0

    async def test_90_day_raw_event_cutoff(self, db: TimeSeriesDB) -> None:
        """Validates: Requirement 12.1 — events exactly at 90 days boundary.

        Events at exactly 90 days should NOT be purged (cutoff is strictly less than).
        Events at 91 days should be purged.
        """
        now = datetime.utcnow()
        # Event at exactly 89 days ago — should be kept
        ts_89 = now - timedelta(days=89)
        db.insert_event(_make_event(event_id="evt-89d", timestamp=ts_89))

        # Event at 91 days ago — should be purged
        ts_91 = now - timedelta(days=91)
        db.insert_event(_make_event(event_id="evt-91d", timestamp=ts_91))

        scheduler = RetentionScheduler(timeseries_db=db, raw_events_days=90)
        results = await scheduler.run_retention()

        assert results["purged_raw_events"] == 1
        remaining = db.get_events()
        assert len(remaining) == 1
        assert remaining[0]["event_id"] == "evt-89d"

    async def test_365_day_aggregated_cutoff(self, db: TimeSeriesDB) -> None:
        """Validates: Requirement 12.2 — aggregated data at 365-day boundary.

        Aggregated data at 364 days should be kept.
        Aggregated data at 366 days should be purged.
        """
        now = datetime.utcnow()
        date_364 = (now - timedelta(days=364)).strftime("%Y-%m-%d")
        date_366 = (now - timedelta(days=366)).strftime("%Y-%m-%d")

        db._conn.execute(
            """INSERT INTO aggregated_events
               (date, camera_id, tenant_id, site_id, object_type, event_count, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (date_364, "cam-1", "t1", "s1", "person", 10, 0.8),
        )
        db._conn.execute(
            """INSERT INTO aggregated_events
               (date, camera_id, tenant_id, site_id, object_type, event_count, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (date_366, "cam-1", "t1", "s1", "vehicle", 20, 0.7),
        )
        db._conn.commit()

        scheduler = RetentionScheduler(timeseries_db=db, aggregated_events_days=365)
        results = await scheduler.run_retention()

        assert results["purged_aggregated"] == 1
        agg_rows = db.get_aggregated_events()
        assert len(agg_rows) == 1
        assert agg_rows[0]["date"] == date_364


# ---------------------------------------------------------------------------
# VectorDB lifecycle-linking tests
# ---------------------------------------------------------------------------


class TestVectorDBLifecycleLinking:
    """Validates: Requirement 12.3 — VectorDB embeddings lifecycle-linked to TSDB."""

    async def test_embeddings_purged_when_events_purged(self, db: TimeSeriesDB) -> None:
        """When raw events are purged, their VectorDB embeddings should also be removed."""
        old_ts = datetime.utcnow() - timedelta(days=100)
        recent_ts = datetime.utcnow() - timedelta(days=10)

        db.insert_event(_make_event(event_id="old-evt", timestamp=old_ts))
        db.insert_event(_make_event(event_id="recent-evt", timestamp=recent_ts))

        mock_vdb = MagicMock()
        mock_vdb.purge_orphaned_embeddings.return_value = 1

        scheduler = RetentionScheduler(timeseries_db=db, vector_db=mock_vdb)
        await scheduler.run_retention()

        # After purging old events, only recent-evt should be valid
        mock_vdb.purge_orphaned_embeddings.assert_called_once()
        valid_ids = mock_vdb.purge_orphaned_embeddings.call_args[0][0]
        assert "recent-evt" in valid_ids
        assert "old-evt" not in valid_ids

    async def test_valid_ids_passed_to_vector_db(self, db: TimeSeriesDB) -> None:
        """The set of valid event IDs passed to VectorDB should match remaining events."""
        recent_ts = datetime.utcnow() - timedelta(days=5)
        db.insert_event(_make_event(event_id="evt-a", timestamp=recent_ts))
        db.insert_event(_make_event(event_id="evt-b", timestamp=recent_ts))

        mock_vdb = MagicMock()
        mock_vdb.purge_orphaned_embeddings.return_value = 0

        scheduler = RetentionScheduler(timeseries_db=db, vector_db=mock_vdb)
        await scheduler.run_retention()

        valid_ids = mock_vdb.purge_orphaned_embeddings.call_args[0][0]
        assert valid_ids == {"evt-a", "evt-b"}


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test that individual step failures don't block other steps."""

    async def test_aggregation_failure_doesnt_block_purge(self, db: TimeSeriesDB) -> None:
        """If aggregation fails, purging should still proceed."""
        old_ts = datetime.utcnow() - timedelta(days=100)
        db.insert_event(_make_event(event_id="old-1", timestamp=old_ts))

        scheduler = RetentionScheduler(timeseries_db=db)

        with patch.object(db, "aggregate_events", side_effect=RuntimeError("DB error")):
            results = await scheduler.run_retention()

        # Aggregation failed, but purge should still have run
        assert results["aggregated_groups"] == 0
        assert results["purged_raw_events"] == 1

    async def test_raw_purge_failure_doesnt_block_aggregated_purge(
        self, db: TimeSeriesDB
    ) -> None:
        """If raw event purge fails, aggregated purge should still proceed."""
        old_date = (datetime.utcnow() - timedelta(days=400)).strftime("%Y-%m-%d")
        db._conn.execute(
            """INSERT INTO aggregated_events
               (date, camera_id, tenant_id, site_id, object_type, event_count, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (old_date, "cam-1", "t1", "s1", "person", 10, 0.8),
        )
        db._conn.commit()

        scheduler = RetentionScheduler(timeseries_db=db)

        with patch.object(db, "purge_raw_events", side_effect=RuntimeError("DB error")):
            results = await scheduler.run_retention()

        assert results["purged_raw_events"] == 0
        assert results["purged_aggregated"] == 1

    async def test_aggregated_purge_failure_doesnt_block_vector_purge(
        self, db: TimeSeriesDB
    ) -> None:
        """If aggregated purge fails, VectorDB purge should still proceed."""
        mock_vdb = MagicMock()
        mock_vdb.purge_orphaned_embeddings.return_value = 2

        scheduler = RetentionScheduler(timeseries_db=db, vector_db=mock_vdb)

        with patch.object(
            db, "purge_aggregated_events", side_effect=RuntimeError("DB error")
        ):
            results = await scheduler.run_retention()

        assert results["purged_aggregated"] == 0
        assert results["purged_embeddings"] == 2
        mock_vdb.purge_orphaned_embeddings.assert_called_once()

    async def test_vector_purge_failure_logged(self, db: TimeSeriesDB) -> None:
        """If VectorDB purge fails, it should be logged but not raise."""
        mock_vdb = MagicMock()
        mock_vdb.purge_orphaned_embeddings.side_effect = RuntimeError("ChromaDB error")

        scheduler = RetentionScheduler(timeseries_db=db, vector_db=mock_vdb)
        results = await scheduler.run_retention()

        assert results["purged_embeddings"] == 0

    async def test_all_steps_fail_gracefully(self, db: TimeSeriesDB) -> None:
        """If every step fails, run_retention should still return without raising."""
        mock_vdb = MagicMock()
        mock_vdb.purge_orphaned_embeddings.side_effect = RuntimeError("fail")

        scheduler = RetentionScheduler(timeseries_db=db, vector_db=mock_vdb)

        with patch.object(db, "aggregate_events", side_effect=RuntimeError("fail")), \
             patch.object(db, "purge_raw_events", side_effect=RuntimeError("fail")), \
             patch.object(db, "purge_aggregated_events", side_effect=RuntimeError("fail")):
            results = await scheduler.run_retention()

        assert results == {
            "aggregated_groups": 0,
            "purged_raw_events": 0,
            "purged_aggregated": 0,
            "purged_embeddings": 0,
        }


# ---------------------------------------------------------------------------
# Start / stop lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Test start/stop lifecycle of the background task."""

    async def test_start_sets_running(self, db: TimeSeriesDB) -> None:
        scheduler = RetentionScheduler(timeseries_db=db, interval_seconds=3600)
        assert scheduler.running is False

        await scheduler.start()
        assert scheduler.running is True
        assert scheduler._task is not None

        await scheduler.stop()

    async def test_stop_clears_running(self, db: TimeSeriesDB) -> None:
        scheduler = RetentionScheduler(timeseries_db=db, interval_seconds=3600)
        await scheduler.start()
        await scheduler.stop()

        assert scheduler.running is False
        assert scheduler._task is None

    async def test_double_start_is_safe(self, db: TimeSeriesDB) -> None:
        """Starting twice should not create a second task."""
        scheduler = RetentionScheduler(timeseries_db=db, interval_seconds=3600)
        await scheduler.start()
        first_task = scheduler._task

        await scheduler.start()  # second start — should be no-op
        assert scheduler._task is first_task

        await scheduler.stop()

    async def test_stop_without_start_is_safe(self, db: TimeSeriesDB) -> None:
        """Stopping without starting should not raise."""
        scheduler = RetentionScheduler(timeseries_db=db)
        await scheduler.stop()  # should not raise
        assert scheduler.running is False

    async def test_background_task_runs_retention(self, db: TimeSeriesDB) -> None:
        """The background task should call run_retention at least once."""
        old_ts = datetime.utcnow() - timedelta(days=100)
        db.insert_event(_make_event(event_id="old-bg", timestamp=old_ts))

        scheduler = RetentionScheduler(
            timeseries_db=db, interval_seconds=3600
        )
        await scheduler.start()

        # Give the background task a moment to run
        await asyncio.sleep(0.1)

        await scheduler.stop()

        # The old event should have been purged by the background task
        remaining = db.get_events()
        assert len(remaining) == 0


# ---------------------------------------------------------------------------
# Integration: TimeSeriesDB retention methods
# ---------------------------------------------------------------------------


class TestTimeSeriesDBRetentionMethods:
    """Test the TimeSeriesDB retention methods directly."""

    def test_aggregate_events_groups_by_date_camera_type(self, db: TimeSeriesDB) -> None:
        """Aggregation should group by date, camera, tenant, site, and object_type."""
        ts = datetime(2024, 6, 15, 10, 0, 0)
        cutoff = datetime(2024, 7, 1).isoformat()

        db.insert_event(_make_event(event_id="e1", timestamp=ts, object_type="person", camera_id="cam-1"))
        db.insert_event(_make_event(event_id="e2", timestamp=ts, object_type="person", camera_id="cam-1"))
        db.insert_event(_make_event(event_id="e3", timestamp=ts, object_type="vehicle", camera_id="cam-1"))

        count = db.aggregate_events(cutoff)
        assert count == 2  # two groups: person and vehicle

        agg = db.get_aggregated_events()
        assert len(agg) == 2

        # Find the person group
        person_row = next(r for r in agg if r["object_type"] == "person")
        assert person_row["event_count"] == 2
        assert person_row["camera_id"] == "cam-1"

        vehicle_row = next(r for r in agg if r["object_type"] == "vehicle")
        assert vehicle_row["event_count"] == 1

    def test_purge_raw_events_deletes_related_alerts(self, db: TimeSeriesDB) -> None:
        """Purging raw events should also delete their related alerts."""
        old_ts = datetime(2024, 1, 1, 10, 0, 0)
        db.insert_event(_make_event(event_id="old-evt", timestamp=old_ts))

        # Insert an alert referencing the old event
        db._conn.execute(
            """INSERT INTO alerts (alert_id, event_id, camera_id, tenant_id,
               alert_type, threat_level, delivered_channels)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("alert-old", "old-evt", "cam-lobby-01", "tenant-acme", "intrusion", "high", "[]"),
        )
        db._conn.commit()

        cutoff = datetime(2024, 6, 1).isoformat()
        db.purge_raw_events(cutoff)

        # Both event and alert should be gone
        assert len(db.get_events()) == 0
        assert len(db.get_alerts()) == 0

    def test_purge_raw_events_deletes_old_heartbeats(self, db: TimeSeriesDB) -> None:
        """Purging raw events should also delete heartbeats older than the cutoff."""
        from agentic_cctv.models import HeartbeatMessage

        old_ts = datetime(2024, 1, 1, 10, 0, 0)
        recent_ts = datetime(2025, 1, 1, 10, 0, 0)

        db.insert_heartbeat(HeartbeatMessage(
            camera_id="cam-1", tenant_id="t1", site_id="s1",
            timestamp=old_ts, cpu_percent=10.0, memory_percent=20.0,
            temperature_celsius=None, inference_latency_ms=5.0,
            gpu_utilization_percent=None,
        ))
        db.insert_heartbeat(HeartbeatMessage(
            camera_id="cam-1", tenant_id="t1", site_id="s1",
            timestamp=recent_ts, cpu_percent=15.0, memory_percent=25.0,
            temperature_celsius=None, inference_latency_ms=6.0,
            gpu_utilization_percent=None,
        ))

        cutoff = datetime(2024, 6, 1).isoformat()
        db.purge_raw_events(cutoff)

        heartbeats = db.get_heartbeats(tenant_id="t1")
        assert len(heartbeats) == 1

    def test_get_all_event_ids_returns_correct_set(self, db: TimeSeriesDB) -> None:
        """get_all_event_ids should return all event IDs in the events table."""
        ts = datetime.utcnow()
        db.insert_event(_make_event(event_id="e1", timestamp=ts))
        db.insert_event(_make_event(event_id="e2", timestamp=ts))
        db.insert_event(_make_event(event_id="e3", timestamp=ts))

        ids = db.get_all_event_ids()
        assert ids == {"e1", "e2", "e3"}

    def test_get_all_event_ids_empty_db(self, db: TimeSeriesDB) -> None:
        ids = db.get_all_event_ids()
        assert ids == set()

    def test_purge_aggregated_events(self, db: TimeSeriesDB) -> None:
        """purge_aggregated_events should delete rows with date < cutoff."""
        db._conn.execute(
            """INSERT INTO aggregated_events
               (date, camera_id, tenant_id, site_id, object_type, event_count, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("2023-01-01", "cam-1", "t1", "s1", "person", 100, 0.8),
        )
        db._conn.execute(
            """INSERT INTO aggregated_events
               (date, camera_id, tenant_id, site_id, object_type, event_count, avg_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("2025-01-01", "cam-1", "t1", "s1", "person", 50, 0.9),
        )
        db._conn.commit()

        count = db.purge_aggregated_events("2024-01-01")
        assert count == 1

        agg = db.get_aggregated_events()
        assert len(agg) == 1
        assert agg[0]["date"] == "2025-01-01"

    def test_aggregated_events_table_has_unique_constraint(self, db: TimeSeriesDB) -> None:
        """The aggregated_events table should have a unique constraint on
        (date, camera_id, tenant_id, site_id, object_type)."""
        ts = datetime(2024, 6, 15, 10, 0, 0)
        cutoff = datetime(2024, 7, 1).isoformat()

        db.insert_event(_make_event(event_id="e1", timestamp=ts, object_type="person"))
        db.aggregate_events(cutoff)

        # Aggregate again — should update existing row, not create duplicate
        db.insert_event(_make_event(event_id="e2", timestamp=ts, object_type="person"))
        db.aggregate_events(cutoff)

        agg = db.get_aggregated_events()
        person_rows = [r for r in agg if r["object_type"] == "person"]
        assert len(person_rows) == 1
        # The count should reflect both aggregation runs
        assert person_rows[0]["event_count"] >= 2
