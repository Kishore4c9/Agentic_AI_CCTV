"""Unit tests for the EventSummarizer.

Tests:
- Hourly summary generation with mocked DB and LLM
- Daily summary generation
- Empty event windows (no events to summarize)
- Summary delivery via alert system
- Scheduler start/stop lifecycle

Requirements: 8.5, 17.2
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from agentic_cctv.alert_system import AlertSystem, PushNotificationChannel
from agentic_cctv.event_summarizer import EventSummarizer
from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    CooldownConfig,
    StructuredEvent,
)
from agentic_cctv.timeseries_db import TimeSeriesDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_id: str,
    camera_id: str = "cam-01",
    tenant_id: str = "tenant-a",
    site_id: str = "site-hq",
    object_type: str = "person",
    confidence: float = 0.9,
    timestamp: Optional[datetime] = None,
) -> StructuredEvent:
    """Create a StructuredEvent for testing."""
    return StructuredEvent(
        event_id=event_id,
        camera_id=camera_id,
        tenant_id=tenant_id,
        site_id=site_id,
        timestamp=timestamp or datetime(2025, 1, 15, 14, 30, 0),
        object_type=object_type,
        track_id=f"trk-{event_id}",
        bounding_box=BoundingBox(x=10, y=20, width=100, height=200),
        confidence=confidence,
        frame_crop="base64data",
    )


def _make_alert(
    alert_id: str,
    event_id: str,
    camera_id: str = "cam-01",
    tenant_id: str = "tenant-a",
    threat_level: str = "high",
) -> AlertPayload:
    """Create an AlertPayload for testing."""
    return AlertPayload(
        alert_id=alert_id,
        event_id=event_id,
        camera_id=camera_id,
        tenant_id=tenant_id,
        site_id="site-hq",
        timestamp=datetime(2025, 1, 15, 14, 30, 0),
        alert_type="intrusion",
        description="Test alert",
        threat_level=threat_level,
        frame_crop_url=None,
    )


class MockLLMBackend:
    """Mock LLM backend that returns a canned summary."""

    def __init__(self, response: Optional[Dict[str, Any]] = None) -> None:
        self._response = response or {
            "scene_description": "LLM-generated summary of events."
        }
        self.call_count = 0

    async def analyze(self, image_b64: str, event_context: dict) -> dict:
        self.call_count += 1
        return self._response


class FailingLLMBackend:
    """Mock LLM backend that always raises an exception."""

    async def analyze(self, image_b64: str, event_context: dict) -> dict:
        raise RuntimeError("LLM service unavailable")


class EmptyResponseLLMBackend:
    """Mock LLM backend that returns an empty response."""

    async def analyze(self, image_b64: str, event_context: dict) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_db() -> TimeSeriesDB:
    """Create an in-memory TimeSeriesDB."""
    return TimeSeriesDB(":memory:")


@pytest.fixture
def alert_system() -> AlertSystem:
    """Create an AlertSystem with a push channel and no cooldown."""
    return AlertSystem(
        channels=[PushNotificationChannel()],
        cooldown_config=CooldownConfig(default_seconds=0),
    )


@pytest.fixture
def summarizer(ts_db: TimeSeriesDB, alert_system: AlertSystem) -> EventSummarizer:
    """Create an EventSummarizer without LLM backend."""
    return EventSummarizer(
        timeseries_db=ts_db,
        alert_system=alert_system,
        llm_backend=None,
        tenant_id="tenant-a",
        site_id="site-hq",
    )


@pytest.fixture
def summarizer_with_llm(
    ts_db: TimeSeriesDB, alert_system: AlertSystem
) -> EventSummarizer:
    """Create an EventSummarizer with a mock LLM backend."""
    return EventSummarizer(
        timeseries_db=ts_db,
        alert_system=alert_system,
        llm_backend=MockLLMBackend(),
        tenant_id="tenant-a",
        site_id="site-hq",
    )


def _insert_test_events(
    ts_db: TimeSeriesDB,
    count: int = 5,
    base_time: Optional[datetime] = None,
    tenant_id: str = "tenant-a",
    camera_id: str = "cam-01",
    object_type: str = "person",
) -> List[StructuredEvent]:
    """Insert test events into the DB and return them."""
    base = base_time or datetime(2025, 1, 15, 14, 0, 0)
    events = []
    for i in range(count):
        # Use unique event IDs incorporating tenant and camera to avoid collisions
        ev = _make_event(
            event_id=f"evt-{tenant_id}-{camera_id}-{i:03d}",
            camera_id=camera_id,
            tenant_id=tenant_id,
            object_type=object_type,
            timestamp=base + timedelta(minutes=i * 5),
        )
        ts_db.insert_event(ev, detection_gate_passed=True, context_gate_passed=True)
        events.append(ev)
    return events


def _insert_test_alerts(
    ts_db: TimeSeriesDB,
    events: List[StructuredEvent],
    tenant_id: str = "tenant-a",
) -> List[AlertPayload]:
    """Insert test alerts into the DB referencing existing events."""
    alerts = []
    for i, ev in enumerate(events):
        al = _make_alert(
            alert_id=f"alert-{tenant_id}-{i:03d}",
            event_id=ev.event_id,
            tenant_id=tenant_id,
        )
        ts_db.insert_alert(al, delivered_channels=["push"])
        alerts.append(al)
    return alerts


# ---------------------------------------------------------------------------
# Tests: Query events and alerts in time window
# ---------------------------------------------------------------------------


class TestQueryEventsInWindow:
    """Test querying events within a time window."""

    def test_query_returns_events_in_window(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Events within the window should be returned."""
        _insert_test_events(ts_db, count=5)
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        events = summarizer.query_events_in_window(start, end)
        assert len(events) == 5

    def test_query_excludes_events_outside_window(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Events outside the window should not be returned."""
        _insert_test_events(ts_db, count=3)
        # Query a window that doesn't overlap
        start = datetime(2025, 1, 16, 0, 0, 0)
        end = datetime(2025, 1, 16, 1, 0, 0)
        events = summarizer.query_events_in_window(start, end)
        assert len(events) == 0

    def test_query_filters_by_tenant(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Events from other tenants should not be returned."""
        base_time = datetime(2025, 1, 15, 14, 0, 0)
        _insert_test_events(ts_db, count=3, tenant_id="tenant-a", base_time=base_time)
        _insert_test_events(ts_db, count=2, tenant_id="tenant-b", base_time=base_time)
        start = datetime(2025, 1, 15, 13, 0, 0)
        end = datetime(2025, 1, 15, 16, 0, 0)
        events = summarizer.query_events_in_window(start, end, tenant_id="tenant-a")
        assert len(events) == 3


class TestQueryAlertsInWindow:
    """Test querying alerts within a time window."""

    def test_query_returns_alerts_in_window(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Alerts within the window should be returned."""
        # Need events first (foreign key)
        events = _insert_test_events(ts_db, count=2)
        _insert_test_alerts(ts_db, events=events)
        # Alerts use created_at which defaults to CURRENT_TIMESTAMP (now),
        # so query a window that includes the current time
        now = datetime.utcnow()
        start = now - timedelta(minutes=5)
        end = now + timedelta(minutes=5)
        alerts = summarizer.query_alerts_in_window(start, end)
        assert len(alerts) == 2

    def test_query_returns_empty_for_no_alerts(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Empty result when no alerts exist in the window."""
        start = datetime(2025, 1, 15, 0, 0, 0)
        end = datetime(2025, 1, 16, 0, 0, 0)
        alerts = summarizer.query_alerts_in_window(start, end)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Tests: Statistical summary building
# ---------------------------------------------------------------------------


class TestBuildStatisticalSummary:
    """Test building statistical summaries from events and alerts."""

    def test_summary_with_events_and_alerts(
        self, summarizer: EventSummarizer
    ) -> None:
        """Summary should include event and alert counts."""
        events = [
            {"object_type": "person", "camera_id": "cam-01"},
            {"object_type": "person", "camera_id": "cam-01"},
            {"object_type": "vehicle", "camera_id": "cam-02"},
        ]
        alerts = [
            {"threat_level": "high"},
            {"threat_level": "medium"},
        ]
        summary = summarizer.build_statistical_summary(events, alerts, "Hourly")
        assert "Hourly Summary:" in summary
        assert "Total events: 3" in summary
        assert "Total alerts: 2" in summary
        assert "person: 2" in summary
        assert "vehicle: 1" in summary
        assert "cam-01: 2" in summary
        assert "high: 1" in summary

    def test_summary_with_no_events(self, summarizer: EventSummarizer) -> None:
        """Summary for empty window should indicate no events."""
        summary = summarizer.build_statistical_summary([], [], "Daily")
        assert "No events or alerts" in summary

    def test_summary_label_is_included(self, summarizer: EventSummarizer) -> None:
        """The window label should appear in the summary."""
        events = [{"object_type": "person", "camera_id": "cam-01"}]
        summary = summarizer.build_statistical_summary(events, [], "Daily")
        assert "Daily Summary:" in summary


# ---------------------------------------------------------------------------
# Tests: Hourly summary generation
# ---------------------------------------------------------------------------


class TestHourlySummaryGeneration:
    """Test hourly summary generation with mocked DB and LLM."""

    @pytest.mark.asyncio
    async def test_hourly_summary_with_events(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Hourly summary should include events from the last hour."""
        _insert_test_events(ts_db, count=3)
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Hourly")
        assert "Hourly Summary:" in summary
        assert "Total events: 3" in summary

    @pytest.mark.asyncio
    async def test_hourly_summary_with_llm(
        self, ts_db: TimeSeriesDB, summarizer_with_llm: EventSummarizer
    ) -> None:
        """Hourly summary with LLM should use LLM-generated text."""
        _insert_test_events(ts_db, count=3)
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer_with_llm.generate_summary(start, end, "Hourly")
        assert "LLM-generated summary" in summary


# ---------------------------------------------------------------------------
# Tests: Daily summary generation
# ---------------------------------------------------------------------------


class TestDailySummaryGeneration:
    """Test daily summary generation."""

    @pytest.mark.asyncio
    async def test_daily_summary_with_events(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Daily summary should include events from the last 24 hours."""
        _insert_test_events(ts_db, count=5)
        start = datetime(2025, 1, 15, 0, 0, 0)
        end = datetime(2025, 1, 16, 0, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Daily")
        assert "Daily Summary:" in summary
        assert "Total events: 5" in summary

    @pytest.mark.asyncio
    async def test_daily_summary_multiple_cameras(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """Daily summary should aggregate across multiple cameras."""
        base_time = datetime(2025, 1, 15, 10, 0, 0)
        _insert_test_events(ts_db, count=3, camera_id="cam-01", base_time=base_time)
        _insert_test_events(ts_db, count=2, camera_id="cam-02", base_time=base_time)
        start = datetime(2025, 1, 15, 0, 0, 0)
        end = datetime(2025, 1, 16, 0, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Daily")
        assert "Total events: 5" in summary
        assert "cam-01" in summary
        assert "cam-02" in summary


# ---------------------------------------------------------------------------
# Tests: Empty event windows
# ---------------------------------------------------------------------------


class TestEmptyEventWindows:
    """Test summary generation when no events exist in the window."""

    @pytest.mark.asyncio
    async def test_empty_window_returns_no_events_message(
        self, summarizer: EventSummarizer
    ) -> None:
        """Empty window should produce a 'no events' summary."""
        start = datetime(2025, 1, 15, 0, 0, 0)
        end = datetime(2025, 1, 16, 0, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Hourly")
        assert "No events or alerts" in summary

    @pytest.mark.asyncio
    async def test_empty_window_skips_llm(
        self, ts_db: TimeSeriesDB, alert_system: AlertSystem
    ) -> None:
        """When no events exist, LLM should not be called."""
        mock_llm = MockLLMBackend()
        summarizer = EventSummarizer(
            timeseries_db=ts_db,
            alert_system=alert_system,
            llm_backend=mock_llm,
            tenant_id="tenant-a",
        )
        start = datetime(2025, 1, 15, 0, 0, 0)
        end = datetime(2025, 1, 16, 0, 0, 0)
        await summarizer.generate_summary(start, end, "Hourly")
        assert mock_llm.call_count == 0


# ---------------------------------------------------------------------------
# Tests: LLM failure fallback
# ---------------------------------------------------------------------------


class TestLLMFailureFallback:
    """Test fallback to statistical summary when LLM fails."""

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_statistical(
        self, ts_db: TimeSeriesDB, alert_system: AlertSystem
    ) -> None:
        """When LLM fails, should fall back to statistical summary."""
        _insert_test_events(ts_db, count=3)
        summarizer = EventSummarizer(
            timeseries_db=ts_db,
            alert_system=alert_system,
            llm_backend=FailingLLMBackend(),
            tenant_id="tenant-a",
        )
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Hourly")
        # Should fall back to statistical summary
        assert "Hourly Summary:" in summary
        assert "Total events: 3" in summary

    @pytest.mark.asyncio
    async def test_llm_empty_response_falls_back(
        self, ts_db: TimeSeriesDB, alert_system: AlertSystem
    ) -> None:
        """When LLM returns empty response, should fall back to statistical."""
        _insert_test_events(ts_db, count=2)
        summarizer = EventSummarizer(
            timeseries_db=ts_db,
            alert_system=alert_system,
            llm_backend=EmptyResponseLLMBackend(),
            tenant_id="tenant-a",
        )
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Hourly")
        assert "Hourly Summary:" in summary
        assert "Total events: 2" in summary


# ---------------------------------------------------------------------------
# Tests: Summary delivery via alert system
# ---------------------------------------------------------------------------


class TestSummaryDelivery:
    """Test summary delivery through the alert system."""

    @pytest.mark.asyncio
    async def test_deliver_summary_returns_true(
        self, summarizer: EventSummarizer
    ) -> None:
        """Delivering a summary should return True on success."""
        result = await summarizer.deliver_summary(
            "Test summary text", "Hourly"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_generate_and_deliver(
        self, ts_db: TimeSeriesDB, summarizer: EventSummarizer
    ) -> None:
        """generate_and_deliver should generate and deliver the summary."""
        _insert_test_events(ts_db, count=3)
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer.generate_and_deliver(start, end, "Hourly")
        assert "Total events: 3" in summary

    @pytest.mark.asyncio
    async def test_deliver_creates_alert_payload(
        self, ts_db: TimeSeriesDB, alert_system: AlertSystem
    ) -> None:
        """The delivered summary should be an AlertPayload with correct fields."""
        delivered_payloads: list = []
        original_send = alert_system.send_alert

        async def capture_send(payload: AlertPayload):
            delivered_payloads.append(payload)
            return await original_send(payload)

        alert_system.send_alert = capture_send  # type: ignore[assignment]

        summarizer = EventSummarizer(
            timeseries_db=ts_db,
            alert_system=alert_system,
            tenant_id="tenant-a",
            site_id="site-hq",
        )
        await summarizer.deliver_summary("Test summary", "Hourly")

        assert len(delivered_payloads) == 1
        payload = delivered_payloads[0]
        assert payload.alert_type == "hourly_summary"
        assert payload.description == "Test summary"
        assert payload.camera_id == "system"
        assert payload.threat_level == "none"
        assert payload.tenant_id == "tenant-a"


# ---------------------------------------------------------------------------
# Tests: Scheduler start/stop lifecycle
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    """Test the background scheduler start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_scheduler(self, summarizer: EventSummarizer) -> None:
        """Starting the scheduler should set is_running to True."""
        assert summarizer.is_running is False
        await summarizer.start_scheduler()
        assert summarizer.is_running is True
        await summarizer.stop_scheduler()
        assert summarizer.is_running is False

    @pytest.mark.asyncio
    async def test_stop_scheduler_cancels_tasks(
        self, summarizer: EventSummarizer
    ) -> None:
        """Stopping the scheduler should cancel background tasks."""
        await summarizer.start_scheduler()
        assert summarizer._hourly_task is not None
        assert summarizer._daily_task is not None
        await summarizer.stop_scheduler()
        assert summarizer._hourly_task is None
        assert summarizer._daily_task is None

    @pytest.mark.asyncio
    async def test_double_start_is_safe(
        self, summarizer: EventSummarizer
    ) -> None:
        """Starting the scheduler twice should not create duplicate tasks."""
        await summarizer.start_scheduler()
        hourly_task = summarizer._hourly_task
        daily_task = summarizer._daily_task
        # Second start should be a no-op
        await summarizer.start_scheduler()
        assert summarizer._hourly_task is hourly_task
        assert summarizer._daily_task is daily_task
        await summarizer.stop_scheduler()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(
        self, summarizer: EventSummarizer
    ) -> None:
        """Stopping the scheduler without starting should not raise."""
        await summarizer.stop_scheduler()
        assert summarizer.is_running is False
