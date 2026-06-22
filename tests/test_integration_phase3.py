"""Integration tests for Phase 3 pipeline.

Validates:
- Multi-camera coordination via shared MCPContextServer
- Multi-camera coordination via shared A2ACommHub
- Watchdog offline/restored alerts
- Frame Crop Store encryption round-trip + retention purge
- Tenant isolation across TimeSeriesDB, FrameCropStore, and RuleStore
- RetentionScheduler data purge

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cctv.a2a_comm import A2ACommHub
from agentic_cctv.alert_system import AlertSystem, PushNotificationChannel
from agentic_cctv.frame_crop_store import FrameCropStore
from agentic_cctv.mcp_server import MCPContextServer
from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    CooldownConfig,
    Rule,
    RuleSet,
    SceneUnderstanding,
    StructuredEvent,
)
from agentic_cctv.orchestration_agent import (
    AlertTool,
    A2ACommTool,
    LogTool,
    MCPContextTool,
    OrchestrationAgent,
)
from agentic_cctv.retention_scheduler import RetentionScheduler
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB
from agentic_cctv.watchdog import Watchdog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structured_event(**overrides) -> StructuredEvent:
    """Create a StructuredEvent with sensible defaults."""
    defaults = dict(
        event_id=str(uuid.uuid4()),
        camera_id="cam-lobby-01",
        tenant_id="tenant-acme",
        site_id="site-hq",
        timestamp=datetime(2025, 1, 15, 23, 30, 0, tzinfo=timezone.utc),
        object_type="person",
        track_id="trk-abc123",
        bounding_box=BoundingBox(x=100, y=80, width=200, height=400),
        confidence=0.92,
        frame_crop="dGVzdA==",
    )
    defaults.update(overrides)
    return StructuredEvent(**defaults)


def _make_scene_understanding(event_id: str, **overrides) -> SceneUnderstanding:
    """Create a SceneUnderstanding with sensible defaults."""
    defaults = dict(
        event_id=event_id,
        scene_description="Person detected in restricted area.",
        threat_level="high",
        recommended_action="alert",
        confidence=0.88,
    )
    defaults.update(overrides)
    return SceneUnderstanding(**defaults)


def _generate_encryption_key() -> str:
    """Generate a valid 64-char hex AES-256 key."""
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def timeseries_db():
    """In-memory TimeSeriesDB."""
    db = TimeSeriesDB(":memory:")
    yield db
    db.close()


@pytest.fixture
def rule_store():
    """In-memory RuleStore."""
    store = RuleStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def alert_system():
    """AlertSystem with a push notification channel and short cooldown."""
    channel = PushNotificationChannel()
    cooldown = CooldownConfig(default_seconds=1)
    return AlertSystem(channels=[channel], cooldown_config=cooldown)


@pytest.fixture
def mcp_server():
    """Shared MCPContextServer instance."""
    return MCPContextServer()


@pytest.fixture
def a2a_hub():
    """Shared A2ACommHub instance."""
    return A2ACommHub()


# ---------------------------------------------------------------------------
# 1. Multi-camera coordination via shared MCPContextServer
# ---------------------------------------------------------------------------


class TestMultiCameraMCPCoordination:
    """Validates: Requirements 6.2, 16.1

    Two cameras process events through MCPContextTool with a shared
    MCPContextServer. Cross-camera context is visible to each other.
    """

    def test_cross_camera_context_sharing(self, mcp_server, timeseries_db) -> None:
        """Events from camera A are visible as cross-camera context to camera B."""
        tool_cam_a = MCPContextTool(server=mcp_server)
        tool_cam_b = MCPContextTool(server=mcp_server)

        # Camera A processes an event
        event_a = _make_structured_event(
            event_id="evt-cam-a-001",
            camera_id="cam-lobby-01",
        )
        scene_a = _make_scene_understanding(
            event_id="evt-cam-a-001",
            scene_description="Person in lobby after hours.",
        )
        result_a = tool_cam_a.invoke(scene_a, event_a)
        assert result_a.success is True

        # Camera B processes an event — should see camera A's context
        event_b = _make_structured_event(
            event_id="evt-cam-b-001",
            camera_id="cam-parking-01",
        )
        scene_b = _make_scene_understanding(
            event_id="evt-cam-b-001",
            scene_description="Vehicle in parking lot.",
        )
        result_b = tool_cam_b.invoke(scene_b, event_b)
        assert result_b.success is True
        assert result_b.data["cross_camera_count"] >= 1

        # Verify the cross-camera context contains camera A's data
        cross_entries = result_b.data["cross_camera_context"]
        cam_a_entries = [
            e for e in cross_entries if e["camera_id"] == "cam-lobby-01"
        ]
        assert len(cam_a_entries) == 1
        assert cam_a_entries[0]["event_id"] == "evt-cam-a-001"

    def test_context_isolation_per_camera(self, mcp_server) -> None:
        """A camera's own context is not returned as cross-camera context."""
        tool = MCPContextTool(server=mcp_server)

        event = _make_structured_event(
            event_id="evt-self-001",
            camera_id="cam-lobby-01",
        )
        scene = _make_scene_understanding(event_id="evt-self-001")
        result = tool.invoke(scene, event)

        # Should not see own context in cross-camera results
        assert result.success is True
        own_entries = [
            e
            for e in result.data.get("cross_camera_context", [])
            if e["camera_id"] == "cam-lobby-01"
        ]
        assert len(own_entries) == 0


# ---------------------------------------------------------------------------
# 2. Multi-camera coordination via shared A2ACommHub
# ---------------------------------------------------------------------------


class TestMultiCameraA2ACoordination:
    """Validates: Requirements 6.3, 16.2

    Multiple agents registered in a shared A2ACommHub can broadcast
    and receive messages for inter-agent coordination.
    """

    def test_broadcast_message_received_by_other_agents(self, a2a_hub) -> None:
        """Broadcast from agent A is received by agent B and C."""
        a2a_hub.register_agent("cam-lobby-01")
        a2a_hub.register_agent("cam-parking-01")
        a2a_hub.register_agent("cam-entrance-01")

        # Agent A broadcasts
        a2a_hub.broadcast_message(
            from_agent_id="cam-lobby-01",
            message_data={"threat_level": "high", "event_id": "evt-001"},
        )

        # Agent B should receive the broadcast
        msgs_b = a2a_hub.receive_messages("cam-parking-01")
        assert len(msgs_b) == 1
        assert msgs_b[0].from_agent_id == "cam-lobby-01"
        assert msgs_b[0].message_data["threat_level"] == "high"

        # Agent C should also receive the broadcast
        msgs_c = a2a_hub.receive_messages("cam-entrance-01")
        assert len(msgs_c) == 1

        # Agent A should NOT receive its own broadcast
        msgs_a = a2a_hub.receive_messages("cam-lobby-01")
        assert len(msgs_a) == 0

    def test_a2a_comm_tool_integration(self, a2a_hub) -> None:
        """A2ACommTool broadcasts scene summaries and receives messages."""
        a2a_hub.register_agent("cam-lobby-01")
        a2a_hub.register_agent("cam-parking-01")

        tool_a = A2ACommTool(hub=a2a_hub, agent_id="cam-lobby-01")
        tool_b = A2ACommTool(hub=a2a_hub, agent_id="cam-parking-01")

        # Tool A broadcasts via invoke
        event_a = _make_structured_event(
            event_id="evt-a2a-001", camera_id="cam-lobby-01"
        )
        scene_a = _make_scene_understanding(event_id="evt-a2a-001")
        result_a = tool_a.invoke(scene_a, event_a)
        assert result_a.success is True

        # Tool B should receive the broadcast when it invokes
        event_b = _make_structured_event(
            event_id="evt-a2a-002", camera_id="cam-parking-01"
        )
        scene_b = _make_scene_understanding(
            event_id="evt-a2a-002", threat_level="low"
        )
        result_b = tool_b.invoke(scene_b, event_b)
        assert result_b.success is True
        assert result_b.data["received_count"] >= 1

    def test_direct_message_between_agents(self, a2a_hub) -> None:
        """Direct message from agent A to agent B is only received by B."""
        a2a_hub.register_agent("cam-lobby-01")
        a2a_hub.register_agent("cam-parking-01")
        a2a_hub.register_agent("cam-entrance-01")

        a2a_hub.send_message(
            from_agent_id="cam-lobby-01",
            to_agent_id="cam-parking-01",
            message_data={"alert": "suspicious person heading your way"},
        )

        # Only agent B should receive the message
        msgs_b = a2a_hub.receive_messages("cam-parking-01")
        assert len(msgs_b) == 1

        msgs_c = a2a_hub.receive_messages("cam-entrance-01")
        assert len(msgs_c) == 0


# ---------------------------------------------------------------------------
# 3. Watchdog offline/restored alerts
# ---------------------------------------------------------------------------


class TestWatchdogOfflineRestoredAlerts:
    """Validates: Requirements 9.2, 9.4, 16.3

    Watchdog detects camera offline when heartbeat gap > 60s and sends
    restored alert when heartbeat resumes.
    """

    @pytest.mark.asyncio
    async def test_watchdog_detects_offline_camera(self, alert_system) -> None:
        """Camera with no heartbeat for > threshold is marked offline."""
        # Create a mock MQTT subscriber
        mock_subscriber = MagicMock()
        mock_subscriber.subscribe = AsyncMock()

        watchdog = Watchdog(
            mqtt_subscriber=mock_subscriber,
            alert_system=alert_system,
            offline_threshold=0.5,  # Very short for testing
            check_interval=0.1,
        )

        await watchdog.start()

        # Simulate a heartbeat arriving
        heartbeat_data = {
            "camera_id": "cam-test-01",
            "tenant_id": "tenant-acme",
            "site_id": "site-hq",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "temperature_celsius": 65.0,
            "inference_latency_ms": 30.0,
            "gpu_utilization_percent": 70.0,
        }
        payload = json.dumps(heartbeat_data).encode("utf-8")
        watchdog._on_heartbeat(
            "tenant-acme/site-hq/cam-test-01/health", payload, 1
        )

        # Camera should be online
        status = watchdog.get_device_status("cam-test-01")
        assert status.status == "online"

        # Wait for the offline threshold to pass
        await asyncio.sleep(0.8)

        # Camera should now be offline
        status = watchdog.get_device_status("cam-test-01")
        assert status.status == "offline"

        await watchdog.stop()

    @pytest.mark.asyncio
    async def test_watchdog_sends_restored_alert(self, alert_system) -> None:
        """Camera that goes offline and then sends heartbeat is restored."""
        mock_subscriber = MagicMock()
        mock_subscriber.subscribe = AsyncMock()

        watchdog = Watchdog(
            mqtt_subscriber=mock_subscriber,
            alert_system=alert_system,
            offline_threshold=0.3,
            check_interval=0.1,
        )

        await watchdog.start()

        # Send initial heartbeat
        heartbeat_data = {
            "camera_id": "cam-test-02",
            "tenant_id": "tenant-acme",
            "site_id": "site-hq",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "temperature_celsius": 65.0,
            "inference_latency_ms": 30.0,
            "gpu_utilization_percent": 70.0,
        }
        payload = json.dumps(heartbeat_data).encode("utf-8")
        watchdog._on_heartbeat(
            "tenant-acme/site-hq/cam-test-02/health", payload, 1
        )

        # Wait for offline
        await asyncio.sleep(0.5)
        status = watchdog.get_device_status("cam-test-02")
        assert status.status == "offline"

        # Send another heartbeat — should restore
        heartbeat_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(heartbeat_data).encode("utf-8")
        watchdog._on_heartbeat(
            "tenant-acme/site-hq/cam-test-02/health", payload, 1
        )

        status = watchdog.get_device_status("cam-test-02")
        assert status.status == "online"

        await watchdog.stop()

    def test_watchdog_unknown_camera_is_offline(self) -> None:
        """Querying a camera that has never sent a heartbeat returns offline."""
        mock_subscriber = MagicMock()
        alert_sys = AlertSystem(
            channels=[PushNotificationChannel()],
            cooldown_config=CooldownConfig(default_seconds=1),
        )
        watchdog = Watchdog(
            mqtt_subscriber=mock_subscriber,
            alert_system=alert_sys,
        )
        status = watchdog.get_device_status("cam-unknown")
        assert status.status == "offline"
        assert status.last_heartbeat is None


# ---------------------------------------------------------------------------
# 4. Frame Crop Store encryption round-trip + retention
# ---------------------------------------------------------------------------


class TestFrameCropStoreEncryptionAndRetention:
    """Validates: Requirements 10.3, 10.4, 10.5

    Crops are encrypted at rest, decryptable with the correct key,
    and expired crops are purged by retention.
    """

    def test_encryption_round_trip(self, tmp_path) -> None:
        """Store a crop, retrieve it, verify original bytes are recovered."""
        key_hex = _generate_encryption_key()
        store = FrameCropStore(
            crop_path=str(tmp_path / "crops"),
            encryption_key_hex=key_hex,
            retention_hours=72,
        )

        original_bytes = b"JPEG_FAKE_DATA_" + secrets.token_bytes(100)
        presigned = store.store_crop(
            event_id="evt-crop-001",
            tenant_id="tenant-acme",
            camera_id="cam-lobby-01",
            crop_bytes=original_bytes,
        )

        # Retrieve via pre-signed URL
        recovered = store.get_crop(presigned.url)
        assert recovered == original_bytes

        # Retrieve via event_id
        recovered_by_id = store.get_crop_by_event_id("evt-crop-001")
        assert recovered_by_id == original_bytes

        store.close()

    def test_encrypted_file_is_not_plaintext(self, tmp_path) -> None:
        """The stored .enc file should not contain the original plaintext."""
        key_hex = _generate_encryption_key()
        store = FrameCropStore(
            crop_path=str(tmp_path / "crops"),
            encryption_key_hex=key_hex,
        )

        original_bytes = b"THIS_IS_PLAINTEXT_DATA_THAT_SHOULD_BE_ENCRYPTED"
        store.store_crop(
            event_id="evt-enc-001",
            tenant_id="tenant-acme",
            camera_id="cam-lobby-01",
            crop_bytes=original_bytes,
        )

        # Read the raw encrypted file
        import os

        enc_path = os.path.join(store.crop_path, "evt-enc-001.enc")
        with open(enc_path, "rb") as f:
            raw_encrypted = f.read()

        # The encrypted file should NOT contain the plaintext
        assert original_bytes not in raw_encrypted
        # And should be different from the original
        assert raw_encrypted != original_bytes

        store.close()

    def test_retention_purge_removes_expired_crops(self, tmp_path) -> None:
        """Crops older than retention period are purged."""
        key_hex = _generate_encryption_key()
        store = FrameCropStore(
            crop_path=str(tmp_path / "crops"),
            encryption_key_hex=key_hex,
            retention_hours=0,  # Immediate expiry for testing
        )

        store.store_crop(
            event_id="evt-expire-001",
            tenant_id="tenant-acme",
            camera_id="cam-lobby-01",
            crop_bytes=b"old_crop_data",
        )

        assert store.count() == 1

        # Purge — with 0-hour retention, everything is expired
        purged = store.purge_expired()
        assert purged == 1
        assert store.count() == 0

        # Verify the crop is no longer retrievable
        result = store.get_crop_by_event_id("evt-expire-001")
        assert result is None

        store.close()


# ---------------------------------------------------------------------------
# 5. Tenant isolation across full pipeline
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    """Validates: Requirements 10.6

    Events for two tenants are isolated: queries scoped to one tenant
    never return the other's data across TimeSeriesDB, FrameCropStore,
    and RuleStore.
    """

    def test_timeseries_db_tenant_isolation(self, timeseries_db) -> None:
        """Events from tenant A are not visible to tenant B queries."""
        event_a = _make_structured_event(
            event_id="evt-tenant-a-001",
            tenant_id="tenant-alpha",
            camera_id="cam-01",
        )
        event_b = _make_structured_event(
            event_id="evt-tenant-b-001",
            tenant_id="tenant-beta",
            camera_id="cam-01",
        )

        timeseries_db.insert_event(event_a)
        timeseries_db.insert_event(event_b)

        # Query scoped to tenant-alpha
        rows_a = timeseries_db.get_events(tenant_id="tenant-alpha")
        assert len(rows_a) == 1
        assert rows_a[0]["tenant_id"] == "tenant-alpha"

        # Query scoped to tenant-beta
        rows_b = timeseries_db.get_events(tenant_id="tenant-beta")
        assert len(rows_b) == 1
        assert rows_b[0]["tenant_id"] == "tenant-beta"

    def test_frame_crop_store_tenant_isolation(self, tmp_path) -> None:
        """Crops from tenant A are not accessible to tenant B."""
        key_hex = _generate_encryption_key()
        store = FrameCropStore(
            crop_path=str(tmp_path / "crops"),
            encryption_key_hex=key_hex,
        )

        store.store_crop(
            event_id="evt-iso-a",
            tenant_id="tenant-alpha",
            camera_id="cam-01",
            crop_bytes=b"alpha_crop",
        )
        store.store_crop(
            event_id="evt-iso-b",
            tenant_id="tenant-beta",
            camera_id="cam-01",
            crop_bytes=b"beta_crop",
        )

        # Tenant-alpha can access their own crop
        result_a = store.get_crop_by_event_id(
            "evt-iso-a", tenant_id="tenant-alpha"
        )
        assert result_a == b"alpha_crop"

        # Tenant-alpha cannot access tenant-beta's crop
        result_cross = store.get_crop_by_event_id(
            "evt-iso-b", tenant_id="tenant-alpha"
        )
        assert result_cross is None

        # Tenant-beta can access their own crop
        result_b = store.get_crop_by_event_id(
            "evt-iso-b", tenant_id="tenant-beta"
        )
        assert result_b == b"beta_crop"

        # Verify get_crops_by_tenant scoping
        crops_a = store.get_crops_by_tenant("tenant-alpha")
        assert len(crops_a) == 1
        assert crops_a[0]["tenant_id"] == "tenant-alpha"

        crops_b = store.get_crops_by_tenant("tenant-beta")
        assert len(crops_b) == 1
        assert crops_b[0]["tenant_id"] == "tenant-beta"

        store.close()

    def test_rule_store_tenant_isolation(self, rule_store) -> None:
        """RuleSets from tenant A are not visible to tenant B."""
        ruleset_a = RuleSet(
            version_id=f"rs-{uuid.uuid4().hex[:12]}",
            camera_id="cam-01",
            rules=[Rule(rule_id="rule-a", object_type="person")],
            created_at=datetime.utcnow(),
        )
        ruleset_b = RuleSet(
            version_id=f"rs-{uuid.uuid4().hex[:12]}",
            camera_id="cam-01",
            rules=[Rule(rule_id="rule-b", object_type="vehicle")],
            created_at=datetime.utcnow(),
        )

        rule_store.save_ruleset(
            "cam-01", ruleset_a, tenant_id="tenant-alpha"
        )
        rule_store.save_ruleset(
            "cam-01", ruleset_b, tenant_id="tenant-beta"
        )

        # Tenant-alpha sees only their ruleset
        active_a = rule_store.get_active_ruleset(
            "cam-01", tenant_id="tenant-alpha"
        )
        assert active_a is not None
        assert active_a.rules[0].rule_id == "rule-a"

        # Tenant-beta sees only their ruleset
        active_b = rule_store.get_active_ruleset(
            "cam-01", tenant_id="tenant-beta"
        )
        assert active_b is not None
        assert active_b.rules[0].rule_id == "rule-b"

        # Version history is scoped by tenant
        history_a = rule_store.get_version_history(
            "cam-01", tenant_id="tenant-alpha"
        )
        history_b = rule_store.get_version_history(
            "cam-01", tenant_id="tenant-beta"
        )
        assert len(history_a) == 1
        assert len(history_b) == 1


# ---------------------------------------------------------------------------
# 6. RetentionScheduler purges old data
# ---------------------------------------------------------------------------


class TestRetentionScheduler:
    """Validates: Requirements 12.1, 12.2, 12.3

    RetentionScheduler properly aggregates and purges old data.
    """

    @pytest.mark.asyncio
    async def test_retention_scheduler_purges_old_events(
        self, timeseries_db
    ) -> None:
        """Old events are aggregated and purged by the retention scheduler."""
        # Insert an event with a very old timestamp
        old_event = _make_structured_event(
            event_id="evt-old-001",
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        timeseries_db.insert_event(old_event)

        # Insert a recent event
        recent_event = _make_structured_event(
            event_id="evt-recent-001",
            timestamp=datetime.now(timezone.utc),
        )
        timeseries_db.insert_event(recent_event)

        # Run retention with 1-day raw retention (so old event is purged)
        scheduler = RetentionScheduler(
            timeseries_db=timeseries_db,
            raw_events_days=1,
            aggregated_events_days=365,
            interval_seconds=86400,
        )

        results = await scheduler.run_retention()

        # Old event should have been aggregated and purged
        assert results["purged_raw_events"] >= 1

        # Recent event should still exist
        rows = timeseries_db.get_events()
        event_ids = [r["event_id"] for r in rows]
        assert "evt-recent-001" in event_ids
        assert "evt-old-001" not in event_ids

    @pytest.mark.asyncio
    async def test_retention_scheduler_start_stop(
        self, timeseries_db
    ) -> None:
        """RetentionScheduler can be started and stopped cleanly."""
        scheduler = RetentionScheduler(
            timeseries_db=timeseries_db,
            raw_events_days=90,
            aggregated_events_days=365,
            interval_seconds=86400,
        )

        assert scheduler.running is False
        await scheduler.start()
        assert scheduler.running is True
        await scheduler.stop()
        assert scheduler.running is False
