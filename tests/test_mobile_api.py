"""Tests for the mobile-compatible REST API.

Covers all mobile API endpoints: alert management with pagination, prompt
configuration, push notification device registration, CORS headers, and
the mobile-optimised dashboard summary.

Uses ``aiohttp.test_utils.TestClient`` directly (no pytest-aiohttp dependency).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentic_cctv.alert_system import MobilePushChannel
from agentic_cctv.dashboard import _get_alert_by_id, _migrate_alert_status
from agentic_cctv.mobile_api import create_mobile_app
from agentic_cctv.models import (
    AlertPayload,
    AlertResult,
    BoundingBox,
    CompiledRuleSet,
    DeviceHealth,
    HeartbeatMessage,
    PromptScope,
    Rule,
    RuleSet,
    StructuredEvent,
)
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_dashboard.py)
# ---------------------------------------------------------------------------


def _make_watchdog_mock():
    """Create a mock Watchdog with two cameras."""
    wd = MagicMock()
    cam1 = DeviceHealth(
        camera_id="cam-01",
        status="online",
        last_heartbeat=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        metrics=HeartbeatMessage(
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            cpu_percent=45.0,
            memory_percent=60.0,
            temperature_celsius=65.0,
            inference_latency_ms=30.0,
            gpu_utilization_percent=70.0,
        ),
    )
    cam2 = DeviceHealth(
        camera_id="cam-02",
        status="offline",
        last_heartbeat=None,
        metrics=None,
    )
    wd.get_all_device_status.return_value = [cam1, cam2]
    wd.get_device_status.side_effect = (
        lambda cid: cam1 if cid == "cam-01" else cam2
    )
    return wd


def _make_tsdb():
    """Create a real in-memory TimeSeriesDB with test data."""
    db = TimeSeriesDB(":memory:")
    _migrate_alert_status(db)

    for i in range(5):
        event = StructuredEvent(
            event_id=f"evt-{i:03d}",
            camera_id="cam-01" if i < 3 else "cam-02",
            tenant_id="t1" if i < 4 else "t2",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, 14, 30, i, tzinfo=timezone.utc),
            object_type="person",
            track_id=f"trk-{i:03d}",
            bounding_box=BoundingBox(x=10, y=20, width=100, height=200),
            confidence=0.9,
            frame_crop="base64data",
        )
        db.insert_event(event)

    for i in range(4):
        alert = AlertPayload(
            alert_id=f"alert-{i:03d}",
            event_id=f"evt-{i:03d}",
            camera_id="cam-01" if i < 2 else "cam-02",
            tenant_id="t1",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, 14, 31, i, tzinfo=timezone.utc),
            alert_type="intrusion",
            description=f"Test alert {i}",
            threat_level="high" if i < 2 else "critical",
            frame_crop_url=None,
        )
        db.insert_alert(alert, delivered_channels=["push"])

    return db


def _make_rule_store():
    """Create a real in-memory RuleStore with test data."""
    rs = RuleStore(":memory:")
    ruleset = RuleSet(
        version_id="rs-v1",
        camera_id="cam-01",
        rules=[
            Rule(rule_id="rule-001", object_type="person", min_confidence=0.8),
        ],
        created_at=datetime(2025, 1, 15, 10, 0, 0),
    )
    rs.save_ruleset("cam-01", ruleset)
    return rs


def _make_prompt_compiler_mock():
    """Create a mock PromptCompiler."""
    pc = MagicMock()
    compiled = CompiledRuleSet(
        ruleset=RuleSet(
            version_id="rs-compiled",
            camera_id="cam-01",
            rules=[
                Rule(
                    rule_id="rule-c1",
                    object_type="person",
                    min_confidence=0.7,
                ),
            ],
            created_at=datetime(2025, 1, 15, 12, 0, 0),
        ),
        original_prompt="Alert on persons at night",
        explanation="Detects persons with confidence >= 0.7",
        confidence=0.85,
    )
    pc.compile = AsyncMock(return_value=compiled)
    pc.confirm_and_activate = AsyncMock(return_value=["rs-activated-01"])
    return pc


def _make_alert_system_mock():
    """Create a mock AlertSystem."""
    alert_sys = MagicMock()
    alert_sys.send_alert = AsyncMock(
        return_value=AlertResult(delivered=True, channels=["push"]),
    )
    return alert_sys


def _make_context_filter_mock():
    """Create a mock ContextFilter."""
    cf = MagicMock()
    cf.reload_rules = MagicMock()
    return cf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def watchdog_mock():
    return _make_watchdog_mock()


@pytest.fixture
def tsdb():
    return _make_tsdb()


@pytest.fixture
def rule_store():
    return _make_rule_store()


@pytest.fixture
def alert_system_mock():
    return _make_alert_system_mock()


@pytest.fixture
def prompt_compiler_mock():
    return _make_prompt_compiler_mock()


@pytest.fixture
def context_filter_mock():
    return _make_context_filter_mock()


@pytest.fixture
async def client(
    watchdog_mock,
    tsdb,
    rule_store,
    alert_system_mock,
    prompt_compiler_mock,
    context_filter_mock,
):
    """Create a TestClient with all dependencies wired up."""
    app = create_mobile_app(
        watchdog=watchdog_mock,
        timeseries_db=tsdb,
        alert_system=alert_system_mock,
        rule_store=rule_store,
        prompt_compiler=prompt_compiler_mock,
        context_filter=context_filter_mock,
    )
    async with TestClient(TestServer(app)) as tc:
        yield tc


@pytest.fixture
async def minimal_client(watchdog_mock, tsdb):
    """Create a TestClient with only required dependencies."""
    app = create_mobile_app(
        watchdog=watchdog_mock,
        timeseries_db=tsdb,
    )
    async with TestClient(TestServer(app)) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Tests — CORS Headers
# ---------------------------------------------------------------------------


class TestCORS:
    async def test_cors_headers_on_get(self, client):
        resp = await client.get("/api/mobile/summary")
        assert resp.status == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert "GET" in resp.headers["Access-Control-Allow-Methods"]
        assert "POST" in resp.headers["Access-Control-Allow-Methods"]

    async def test_cors_preflight_options(self, client):
        resp = await client.options("/api/mobile/alerts")
        assert resp.status == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert "Content-Type" in resp.headers["Access-Control-Allow-Headers"]

    async def test_cors_headers_on_post(self, client):
        resp = await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-123",
                "platform": "ios",
                "tenant_id": "t1",
            },
        )
        assert resp.status == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"


# ---------------------------------------------------------------------------
# Tests — Alert Management
# ---------------------------------------------------------------------------


class TestAlerts:
    async def test_get_alerts_default(self, client):
        resp = await client.get("/api/mobile/alerts")
        assert resp.status == 200
        data = await resp.json()
        assert "alerts" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert len(data["alerts"]) == 4

    async def test_get_alerts_with_camera_filter(self, client):
        resp = await client.get("/api/mobile/alerts?camera_id=cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 2

    async def test_get_alerts_with_tenant_filter(self, client):
        resp = await client.get("/api/mobile/alerts?tenant_id=t1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 4

    async def test_get_alerts_with_pagination(self, client):
        resp = await client.get("/api/mobile/alerts?limit=2&offset=0")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

    async def test_get_alerts_with_offset(self, client):
        resp = await client.get("/api/mobile/alerts?limit=2&offset=2")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 2
        assert data["offset"] == 2

    async def test_get_alerts_invalid_limit_uses_default(self, client):
        resp = await client.get("/api/mobile/alerts?limit=abc")
        assert resp.status == 200
        data = await resp.json()
        assert data["limit"] == 20

    async def test_get_alerts_invalid_offset_uses_default(self, client):
        resp = await client.get("/api/mobile/alerts?offset=abc")
        assert resp.status == 200
        data = await resp.json()
        assert data["offset"] == 0

    async def test_get_single_alert(self, client):
        resp = await client.get("/api/mobile/alerts/alert-000")
        assert resp.status == 200
        data = await resp.json()
        assert data["alert_id"] == "alert-000"

    async def test_get_single_alert_not_found(self, client):
        resp = await client.get("/api/mobile/alerts/nonexistent")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data

    async def test_acknowledge_alert(self, client, tsdb):
        resp = await client.post(
            "/api/mobile/alerts/alert-000/acknowledge",
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["alert_id"] == "alert-000"
        assert data["status"] == "acknowledged"
        alert = _get_alert_by_id(tsdb, "alert-000")
        assert alert["status"] == "acknowledged"

    async def test_dismiss_alert(self, client, tsdb):
        resp = await client.post("/api/mobile/alerts/alert-001/dismiss")
        assert resp.status == 200
        data = await resp.json()
        assert data["alert_id"] == "alert-001"
        assert data["status"] == "dismissed"
        alert = _get_alert_by_id(tsdb, "alert-001")
        assert alert["status"] == "dismissed"

    async def test_escalate_alert(self, client, tsdb, alert_system_mock):
        resp = await client.post("/api/mobile/alerts/alert-000/escalate")
        assert resp.status == 200
        data = await resp.json()
        assert data["alert_id"] == "alert-000"
        assert data["status"] == "escalated"
        alert = _get_alert_by_id(tsdb, "alert-000")
        assert alert["status"] == "escalated"
        alert_system_mock.send_alert.assert_called_once()

    async def test_acknowledge_nonexistent_alert(self, client):
        resp = await client.post(
            "/api/mobile/alerts/nonexistent/acknowledge",
        )
        assert resp.status == 404

    async def test_dismiss_nonexistent_alert(self, client):
        resp = await client.post("/api/mobile/alerts/nonexistent/dismiss")
        assert resp.status == 404

    async def test_escalate_nonexistent_alert(self, client):
        resp = await client.post("/api/mobile/alerts/nonexistent/escalate")
        assert resp.status == 404

    async def test_get_alerts_with_status_filter(self, client, tsdb):
        # Acknowledge one alert first
        from agentic_cctv.dashboard import _update_alert_status

        _update_alert_status(tsdb, "alert-000", "acknowledged")

        resp = await client.get("/api/mobile/alerts?status=acknowledged")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["alert_id"] == "alert-000"


# ---------------------------------------------------------------------------
# Tests — Prompt Configuration
# ---------------------------------------------------------------------------


class TestPromptConfiguration:
    async def test_compile_prompt(self, client, prompt_compiler_mock):
        resp = await client.post(
            "/api/mobile/prompt/compile",
            json={
                "prompt": "Alert on persons at night",
                "scope_type": "camera",
                "target_ids": ["cam-01"],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "compiled"
        assert data["confidence"] == 0.85
        assert data["rules_count"] == 1
        assert "ruleset" in data
        prompt_compiler_mock.compile.assert_called_once()

    async def test_compile_prompt_missing_prompt(self, client):
        resp = await client.post(
            "/api/mobile/prompt/compile",
            json={"scope_type": "camera", "target_ids": ["cam-01"]},
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    async def test_compile_prompt_missing_target_ids(self, client):
        resp = await client.post(
            "/api/mobile/prompt/compile",
            json={"prompt": "Alert on persons", "scope_type": "camera"},
        )
        assert resp.status == 400

    async def test_compile_prompt_invalid_json(self, client):
        resp = await client.post(
            "/api/mobile/prompt/compile",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_activate_prompt(self, client, prompt_compiler_mock):
        resp = await client.post(
            "/api/mobile/prompt/activate",
            json={
                "prompt": "Alert on persons at night",
                "scope_type": "camera",
                "target_ids": ["cam-01"],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "activated"
        assert "version_ids" in data
        prompt_compiler_mock.confirm_and_activate.assert_called_once()

    async def test_activate_prompt_missing_target_ids(self, client):
        resp = await client.post(
            "/api/mobile/prompt/activate",
            json={"prompt": "Alert on persons"},
        )
        assert resp.status == 400

    async def test_activate_prompt_missing_prompt(self, client):
        resp = await client.post(
            "/api/mobile/prompt/activate",
            json={"scope_type": "camera", "target_ids": ["cam-01"]},
        )
        assert resp.status == 400

    async def test_compile_prompt_no_compiler(self, minimal_client):
        resp = await minimal_client.post(
            "/api/mobile/prompt/compile",
            json={"prompt": "test", "target_ids": ["cam-01"]},
        )
        assert resp.status == 501

    async def test_activate_prompt_no_compiler(self, minimal_client):
        resp = await minimal_client.post(
            "/api/mobile/prompt/activate",
            json={"prompt": "test", "target_ids": ["cam-01"]},
        )
        assert resp.status == 501

    async def test_compile_prompt_compiler_failure(
        self, client, prompt_compiler_mock,
    ):
        prompt_compiler_mock.compile = AsyncMock(
            side_effect=RuntimeError("LLM failed"),
        )
        resp = await client.post(
            "/api/mobile/prompt/compile",
            json={
                "prompt": "Alert on persons",
                "scope_type": "camera",
                "target_ids": ["cam-01"],
            },
        )
        assert resp.status == 500
        data = await resp.json()
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests — Rules
# ---------------------------------------------------------------------------


class TestRules:
    async def test_get_active_rules(self, client):
        resp = await client.get("/api/mobile/rules/cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-01"
        assert "rules" in data
        assert len(data["rules"]) == 1
        assert data["rules"][0]["rule_id"] == "rule-001"

    async def test_get_rules_no_active_ruleset(self, client):
        resp = await client.get("/api/mobile/rules/cam-nonexistent")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data

    async def test_get_rules_no_rule_store(self, minimal_client):
        resp = await minimal_client.get("/api/mobile/rules/cam-01")
        assert resp.status == 501


# ---------------------------------------------------------------------------
# Tests — Push Notification Registration
# ---------------------------------------------------------------------------


class TestPushRegistration:
    async def test_register_device(self, client):
        resp = await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-abc123",
                "platform": "ios",
                "tenant_id": "t1",
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "registered"
        assert data["device_token"] == "tok-abc123"
        assert data["platform"] == "ios"
        assert data["tenant_id"] == "t1"

    async def test_register_device_android(self, client):
        resp = await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-android-1",
                "platform": "android",
                "tenant_id": "t1",
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["platform"] == "android"

    async def test_register_device_missing_token(self, client):
        resp = await client.post(
            "/api/mobile/push/register",
            json={"platform": "ios", "tenant_id": "t1"},
        )
        assert resp.status == 400

    async def test_register_device_invalid_platform(self, client):
        resp = await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-123",
                "platform": "windows",
                "tenant_id": "t1",
            },
        )
        assert resp.status == 400

    async def test_register_device_missing_tenant(self, client):
        resp = await client.post(
            "/api/mobile/push/register",
            json={"device_token": "tok-123", "platform": "ios"},
        )
        assert resp.status == 400

    async def test_register_device_invalid_json(self, client):
        resp = await client.post(
            "/api/mobile/push/register",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_unregister_device(self, client):
        # Register first
        await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-to-remove",
                "platform": "ios",
                "tenant_id": "t1",
            },
        )
        # Unregister
        resp = await client.delete(
            "/api/mobile/push/unregister",
            json={"device_token": "tok-to-remove"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "unregistered"
        assert data["device_token"] == "tok-to-remove"

    async def test_unregister_device_not_found(self, client):
        resp = await client.delete(
            "/api/mobile/push/unregister",
            json={"device_token": "nonexistent"},
        )
        assert resp.status == 404

    async def test_unregister_device_missing_token(self, client):
        resp = await client.delete(
            "/api/mobile/push/unregister",
            json={},
        )
        assert resp.status == 400

    async def test_unregister_device_invalid_json(self, client):
        resp = await client.delete(
            "/api/mobile/push/unregister",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_list_devices_empty(self, client):
        resp = await client.get("/api/mobile/push/devices")
        assert resp.status == 200
        data = await resp.json()
        assert data["devices"] == []

    async def test_list_devices_after_registration(self, client):
        await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-1",
                "platform": "ios",
                "tenant_id": "t1",
            },
        )
        await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-2",
                "platform": "android",
                "tenant_id": "t2",
            },
        )

        resp = await client.get("/api/mobile/push/devices")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["devices"]) == 2

    async def test_list_devices_with_tenant_filter(self, client):
        await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-a",
                "platform": "ios",
                "tenant_id": "t1",
            },
        )
        await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-b",
                "platform": "android",
                "tenant_id": "t2",
            },
        )

        resp = await client.get("/api/mobile/push/devices?tenant_id=t1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["devices"]) == 1
        assert data["devices"][0]["tenant_id"] == "t1"

    async def test_register_device_overwrites_existing(self, client):
        """Re-registering the same token updates the entry."""
        await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-dup",
                "platform": "ios",
                "tenant_id": "t1",
            },
        )
        await client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "tok-dup",
                "platform": "android",
                "tenant_id": "t2",
            },
        )

        resp = await client.get("/api/mobile/push/devices")
        data = await resp.json()
        assert len(data["devices"]) == 1
        assert data["devices"][0]["platform"] == "android"
        assert data["devices"][0]["tenant_id"] == "t2"


# ---------------------------------------------------------------------------
# Tests — Mobile Summary
# ---------------------------------------------------------------------------


class TestSummary:
    async def test_get_summary(self, client):
        resp = await client.get("/api/mobile/summary")
        assert resp.status == 200
        data = await resp.json()
        assert data["total_cameras"] == 2
        assert data["online_cameras"] == 1
        assert data["offline_cameras"] == 1
        assert data["total_alerts"] == 4
        assert "active_alerts" in data
        assert "recent_critical_alerts" in data

    async def test_summary_includes_critical_alerts(self, client):
        resp = await client.get("/api/mobile/summary")
        data = await resp.json()
        # We have 2 critical alerts in test data
        assert len(data["recent_critical_alerts"]) >= 2


# ---------------------------------------------------------------------------
# Tests — MobilePushChannel
# ---------------------------------------------------------------------------


class TestMobilePushChannel:
    async def test_mobile_push_channel_deliver(self):
        channel = MobilePushChannel(
            device_token="tok-test-123", platform="ios",
        )
        payload = AlertPayload(
            alert_id="alert-test",
            event_id="evt-test",
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            alert_type="intrusion",
            description="Test alert",
            threat_level="high",
            frame_crop_url=None,
        )
        result = await channel.deliver(payload)
        assert result is True

    async def test_mobile_push_channel_android(self):
        channel = MobilePushChannel(
            device_token="tok-android-456", platform="android",
        )
        payload = AlertPayload(
            alert_id="alert-test-2",
            event_id="evt-test-2",
            camera_id="cam-02",
            tenant_id="t1",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            alert_type="fire",
            description="Fire detected",
            threat_level="critical",
            frame_crop_url=None,
        )
        result = await channel.deliver(payload)
        assert result is True


# ---------------------------------------------------------------------------
# Tests — App Factory with minimal dependencies
# ---------------------------------------------------------------------------


class TestAppFactory:
    async def test_create_app_minimal(self, minimal_client):
        resp = await minimal_client.get("/api/mobile/summary")
        assert resp.status == 200

        resp = await minimal_client.get("/api/mobile/alerts")
        assert resp.status == 200

        resp = await minimal_client.get("/api/mobile/push/devices")
        assert resp.status == 200

    async def test_create_app_with_all_deps(self, client):
        resp = await client.get("/api/mobile/rules/cam-01")
        assert resp.status == 200

        resp = await client.get("/api/mobile/summary")
        assert resp.status == 200
