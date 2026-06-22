"""Tests for the operator dashboard (web UI and REST API).

Covers all API endpoints, alert management, rule set operations,
prompt compilation, overview stats, HTML dashboard serving, and error handling.

Uses ``aiohttp.test_utils.TestClient`` directly (no pytest-aiohttp dependency).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentic_cctv.dashboard import (
    _get_alert_by_id,
    _get_event_by_id,
    _migrate_alert_status,
    _update_alert_status,
    create_dashboard_app,
)
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
# Helpers
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
    wd.get_device_status.side_effect = lambda cid: cam1 if cid == "cam-01" else cam2
    return wd


def _make_tsdb():
    """Create a real in-memory TimeSeriesDB with test data."""
    db = TimeSeriesDB(":memory:")
    _migrate_alert_status(db)

    for i in range(3):
        event = StructuredEvent(
            event_id=f"evt-{i:03d}",
            camera_id="cam-01" if i < 2 else "cam-02",
            tenant_id="t1",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, 14, 30, i, tzinfo=timezone.utc),
            object_type="person",
            track_id=f"trk-{i:03d}",
            bounding_box=BoundingBox(x=10, y=20, width=100, height=200),
            confidence=0.9,
            frame_crop="base64data",
        )
        db.insert_event(event)

    for i in range(2):
        alert = AlertPayload(
            alert_id=f"alert-{i:03d}",
            event_id=f"evt-{i:03d}",
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=datetime(2025, 1, 15, 14, 31, i, tzinfo=timezone.utc),
            alert_type="intrusion",
            description=f"Test alert {i}",
            threat_level="high",
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
            rules=[Rule(rule_id="rule-c1", object_type="person", min_confidence=0.7)],
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
        return_value=AlertResult(delivered=True, channels=["push"])
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
    watchdog_mock, tsdb, rule_store, alert_system_mock,
    prompt_compiler_mock, context_filter_mock,
):
    """Create a TestClient with all dependencies wired up."""
    app = create_dashboard_app(
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
    app = create_dashboard_app(
        watchdog=watchdog_mock,
        timeseries_db=tsdb,
    )
    async with TestClient(TestServer(app)) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Tests — Dashboard HTML
# ---------------------------------------------------------------------------


class TestDashboardHTML:
    async def test_serve_dashboard_returns_html(self, client):
        resp = await client.get("/dashboard")
        assert resp.status == 200
        assert "text/html" in resp.content_type
        text = await resp.text()
        assert "CCTV Monitoring Dashboard" in text
        assert "<script>" in text

    async def test_dashboard_contains_key_sections(self, client):
        resp = await client.get("/dashboard")
        text = await resp.text()
        assert "Camera Health" in text
        assert "Live Event Feed" in text
        assert "Alerts" in text
        assert "Rule Set Viewer" in text
        assert "Prompt Configuration" in text


# ---------------------------------------------------------------------------
# Tests — Camera Health
# ---------------------------------------------------------------------------


class TestCameraHealth:
    async def test_get_all_cameras(self, client, watchdog_mock):
        resp = await client.get("/api/dashboard/cameras")
        assert resp.status == 200
        data = await resp.json()
        assert "cameras" in data
        assert len(data["cameras"]) == 2
        assert data["cameras"][0]["camera_id"] == "cam-01"
        assert data["cameras"][0]["status"] == "online"
        assert data["cameras"][1]["camera_id"] == "cam-02"
        assert data["cameras"][1]["status"] == "offline"
        watchdog_mock.get_all_device_status.assert_called_once()

    async def test_get_single_camera_online(self, client):
        resp = await client.get("/api/dashboard/cameras/cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-01"
        assert data["status"] == "online"
        assert data["last_heartbeat"] is not None
        assert data["metrics"] is not None
        assert data["metrics"]["cpu_percent"] == 45.0

    async def test_get_single_camera_offline(self, client):
        resp = await client.get("/api/dashboard/cameras/cam-02")
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-02"
        assert data["status"] == "offline"
        assert data["last_heartbeat"] is None
        assert data["metrics"] is None


# ---------------------------------------------------------------------------
# Tests — Events
# ---------------------------------------------------------------------------


class TestEvents:
    async def test_get_events_default(self, client):
        resp = await client.get("/api/dashboard/events")
        assert resp.status == 200
        data = await resp.json()
        assert "events" in data
        assert len(data["events"]) == 3

    async def test_get_events_with_camera_filter(self, client):
        resp = await client.get("/api/dashboard/events?camera_id=cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["events"]) == 2

    async def test_get_events_with_tenant_filter(self, client):
        resp = await client.get("/api/dashboard/events?tenant_id=t1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["events"]) == 3

    async def test_get_events_with_limit(self, client):
        resp = await client.get("/api/dashboard/events?limit=1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["events"]) == 1

    async def test_get_events_invalid_limit_uses_default(self, client):
        resp = await client.get("/api/dashboard/events?limit=abc")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["events"]) == 3

    async def test_get_single_event(self, client):
        resp = await client.get("/api/dashboard/events/evt-000")
        assert resp.status == 200
        data = await resp.json()
        assert data["event_id"] == "evt-000"
        assert data["camera_id"] == "cam-01"

    async def test_get_single_event_not_found(self, client):
        resp = await client.get("/api/dashboard/events/nonexistent")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data


# ---------------------------------------------------------------------------
# Tests — Alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    async def test_get_alerts_default(self, client):
        resp = await client.get("/api/dashboard/alerts")
        assert resp.status == 200
        data = await resp.json()
        assert "alerts" in data
        assert len(data["alerts"]) == 2

    async def test_get_alerts_with_camera_filter(self, client):
        resp = await client.get("/api/dashboard/alerts?camera_id=cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 2

    async def test_get_alerts_with_tenant_filter(self, client):
        resp = await client.get("/api/dashboard/alerts?tenant_id=t1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 2

    async def test_get_alerts_with_limit(self, client):
        resp = await client.get("/api/dashboard/alerts?limit=1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 1

    async def test_acknowledge_alert(self, client, tsdb):
        resp = await client.post("/api/dashboard/alerts/alert-000/acknowledge")
        assert resp.status == 200
        data = await resp.json()
        assert data["alert_id"] == "alert-000"
        assert data["status"] == "acknowledged"
        alert = _get_alert_by_id(tsdb, "alert-000")
        assert alert["status"] == "acknowledged"

    async def test_dismiss_alert(self, client, tsdb):
        resp = await client.post("/api/dashboard/alerts/alert-001/dismiss")
        assert resp.status == 200
        data = await resp.json()
        assert data["alert_id"] == "alert-001"
        assert data["status"] == "dismissed"
        alert = _get_alert_by_id(tsdb, "alert-001")
        assert alert["status"] == "dismissed"

    async def test_escalate_alert(self, client, tsdb, alert_system_mock):
        resp = await client.post("/api/dashboard/alerts/alert-000/escalate")
        assert resp.status == 200
        data = await resp.json()
        assert data["alert_id"] == "alert-000"
        assert data["status"] == "escalated"
        alert = _get_alert_by_id(tsdb, "alert-000")
        assert alert["status"] == "escalated"
        alert_system_mock.send_alert.assert_called_once()

    async def test_acknowledge_nonexistent_alert(self, client):
        resp = await client.post("/api/dashboard/alerts/nonexistent/acknowledge")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data

    async def test_dismiss_nonexistent_alert(self, client):
        resp = await client.post("/api/dashboard/alerts/nonexistent/dismiss")
        assert resp.status == 404

    async def test_escalate_nonexistent_alert(self, client):
        resp = await client.post("/api/dashboard/alerts/nonexistent/escalate")
        assert resp.status == 404


# ---------------------------------------------------------------------------
# Tests — Rule Sets
# ---------------------------------------------------------------------------


class TestRuleSets:
    async def test_get_active_rules(self, client):
        resp = await client.get("/api/dashboard/rules/cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-01"
        assert "rules" in data
        assert len(data["rules"]) == 1
        assert data["rules"][0]["rule_id"] == "rule-001"

    async def test_get_rules_no_active_ruleset(self, client):
        resp = await client.get("/api/dashboard/rules/cam-nonexistent")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data

    async def test_get_rules_history(self, client):
        resp = await client.get("/api/dashboard/rules/cam-01/history")
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-01"
        assert "versions" in data
        assert len(data["versions"]) >= 1

    async def test_get_rules_history_empty(self, client):
        resp = await client.get("/api/dashboard/rules/cam-nonexistent/history")
        assert resp.status == 200
        data = await resp.json()
        assert data["versions"] == []

    async def test_rollback_rules(self, client, rule_store, context_filter_mock):
        rs2 = RuleSet(
            version_id="rs-v2",
            camera_id="cam-01",
            rules=[Rule(rule_id="rule-002", object_type="vehicle", min_confidence=0.6)],
            created_at=datetime(2025, 1, 15, 11, 0, 0),
        )
        rule_store.save_ruleset("cam-01", rs2)

        resp = await client.post(
            "/api/dashboard/rules/cam-01/rollback",
            json={"version_id": "rs-v1"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-01"
        assert data["status"] == "ok"
        assert "new_version_id" in data
        context_filter_mock.reload_rules.assert_called_with("cam-01")

    async def test_rollback_missing_version_id(self, client):
        resp = await client.post(
            "/api/dashboard/rules/cam-01/rollback",
            json={},
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    async def test_rollback_nonexistent_version(self, client):
        resp = await client.post(
            "/api/dashboard/rules/cam-01/rollback",
            json={"version_id": "nonexistent"},
        )
        assert resp.status == 404

    async def test_rollback_invalid_json(self, client):
        resp = await client.post(
            "/api/dashboard/rules/cam-01/rollback",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400


# ---------------------------------------------------------------------------
# Tests — Prompt Configuration
# ---------------------------------------------------------------------------


class TestPromptConfiguration:
    async def test_compile_prompt(self, client, prompt_compiler_mock):
        resp = await client.post(
            "/api/dashboard/prompt/compile",
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
            "/api/dashboard/prompt/compile",
            json={"scope_type": "camera", "target_ids": ["cam-01"]},
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    async def test_compile_prompt_missing_target_ids(self, client):
        resp = await client.post(
            "/api/dashboard/prompt/compile",
            json={"prompt": "Alert on persons", "scope_type": "camera"},
        )
        assert resp.status == 400

    async def test_compile_prompt_invalid_json(self, client):
        resp = await client.post(
            "/api/dashboard/prompt/compile",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_activate_prompt(self, client, prompt_compiler_mock):
        resp = await client.post(
            "/api/dashboard/prompt/activate",
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
            "/api/dashboard/prompt/activate",
            json={"prompt": "Alert on persons"},
        )
        assert resp.status == 400

    async def test_activate_prompt_missing_prompt(self, client):
        resp = await client.post(
            "/api/dashboard/prompt/activate",
            json={"scope_type": "camera", "target_ids": ["cam-01"]},
        )
        assert resp.status == 400

    async def test_compile_prompt_no_compiler(self, minimal_client):
        resp = await minimal_client.post(
            "/api/dashboard/prompt/compile",
            json={"prompt": "test", "target_ids": ["cam-01"]},
        )
        assert resp.status == 501

    async def test_activate_prompt_no_compiler(self, minimal_client):
        resp = await minimal_client.post(
            "/api/dashboard/prompt/activate",
            json={"prompt": "test", "target_ids": ["cam-01"]},
        )
        assert resp.status == 501

    async def test_compile_prompt_compiler_failure(self, client, prompt_compiler_mock):
        prompt_compiler_mock.compile = AsyncMock(side_effect=RuntimeError("LLM failed"))
        resp = await client.post(
            "/api/dashboard/prompt/compile",
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
# Tests — Overview
# ---------------------------------------------------------------------------


class TestOverview:
    async def test_get_overview(self, client):
        resp = await client.get("/api/dashboard/overview")
        assert resp.status == 200
        data = await resp.json()
        assert data["total_cameras"] == 2
        assert data["online_cameras"] == 1
        assert data["offline_cameras"] == 1
        assert data["recent_alert_count"] == 2
        assert data["recent_event_count"] == 3


# ---------------------------------------------------------------------------
# Tests — DB helpers
# ---------------------------------------------------------------------------


class TestDBHelpers:
    def test_migrate_alert_status_idempotent(self, tsdb):
        _migrate_alert_status(tsdb)
        _migrate_alert_status(tsdb)

    def test_get_alert_by_id(self, tsdb):
        alert = _get_alert_by_id(tsdb, "alert-000")
        assert alert is not None
        assert alert["alert_id"] == "alert-000"

    def test_get_alert_by_id_not_found(self, tsdb):
        assert _get_alert_by_id(tsdb, "nonexistent") is None

    def test_get_event_by_id(self, tsdb):
        event = _get_event_by_id(tsdb, "evt-000")
        assert event is not None
        assert event["event_id"] == "evt-000"

    def test_get_event_by_id_not_found(self, tsdb):
        assert _get_event_by_id(tsdb, "nonexistent") is None

    def test_update_alert_status(self, tsdb):
        assert _update_alert_status(tsdb, "alert-000", "acknowledged") is True
        alert = _get_alert_by_id(tsdb, "alert-000")
        assert alert["status"] == "acknowledged"

    def test_update_alert_status_nonexistent(self, tsdb):
        assert _update_alert_status(tsdb, "nonexistent", "acknowledged") is False


# ---------------------------------------------------------------------------
# Tests — App factory
# ---------------------------------------------------------------------------


class TestAppFactory:
    async def test_create_app_minimal(self, minimal_client):
        resp = await minimal_client.get("/dashboard")
        assert resp.status == 200

        resp = await minimal_client.get("/api/dashboard/cameras")
        assert resp.status == 200

        resp = await minimal_client.get("/api/dashboard/events")
        assert resp.status == 200

        resp = await minimal_client.get("/api/dashboard/alerts")
        assert resp.status == 200

        resp = await minimal_client.get("/api/dashboard/overview")
        assert resp.status == 200

    async def test_create_app_with_all_deps(self, client):
        resp = await client.get("/api/dashboard/rules/cam-01")
        assert resp.status == 200
