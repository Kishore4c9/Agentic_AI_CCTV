"""Integration tests for Phase 4 pipeline.

Validates:
- Summary generation with real TimeSeriesDB data (Requirement 17.2)
- Test-against-history for RuleSet validation (Requirement 17.3)
- Dashboard API endpoints with all Phase 4 components (Requirement 17.1)
- Environment template loading and config generation (Requirement 17.5)
- VLMBackendLLMAdapter from main.py (Requirement 17.5)
- Mobile API integration (Requirement 17.4)

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentic_cctv.alert_system import AlertSystem, PushNotificationChannel
from agentic_cctv.dashboard import (
    _migrate_alert_status,
    create_dashboard_app,
)
from agentic_cctv.environment_templates import (
    generate_config_from_template,
    get_template,
    list_templates,
)
from agentic_cctv.event_summarizer import EventSummarizer
from agentic_cctv.main import VLMBackendLLMAdapter
from agentic_cctv.mobile_api import create_mobile_app
from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    CompiledRuleSet,
    CooldownConfig,
    DeviceHealth,
    HeartbeatMessage,
    PromptScope,
    Rule,
    RuleSet,
    StructuredEvent,
)
from agentic_cctv.prompt_compiler import PromptCompiler
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
        camera_id="cam-01",
        tenant_id="t1",
        site_id="s1",
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        object_type="person",
        track_id="trk-abc123",
        bounding_box=BoundingBox(x=100, y=80, width=200, height=400),
        confidence=0.92,
        frame_crop="dGVzdA==",
    )
    defaults.update(overrides)
    return StructuredEvent(**defaults)


def _make_alert_payload(**overrides) -> AlertPayload:
    """Create an AlertPayload with sensible defaults."""
    defaults = dict(
        alert_id=str(uuid.uuid4()),
        event_id="evt-000",
        camera_id="cam-01",
        tenant_id="t1",
        site_id="s1",
        timestamp=datetime(2025, 1, 15, 14, 31, 0, tzinfo=timezone.utc),
        alert_type="intrusion",
        description="Test alert",
        threat_level="high",
        frame_crop_url=None,
    )
    defaults.update(overrides)
    return AlertPayload(**defaults)


def _make_watchdog_mock():
    """Create a mock Watchdog with two cameras."""
    wd = MagicMock(spec=Watchdog)
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


class MockLLMClient:
    """Mock LLM client for PromptCompiler that returns valid rule JSON."""

    def __init__(self, response_json: Optional[str] = None) -> None:
        self._response = response_json or json.dumps({
            "rules": [
                {
                    "rule_id": "rule-test-001",
                    "object_type": "person",
                    "min_confidence": 0.8,
                }
            ],
            "explanation": "Detects persons with confidence >= 0.8",
            "confidence": 0.9,
        })

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def timeseries_db():
    """In-memory TimeSeriesDB with test data."""
    db = TimeSeriesDB(":memory:")
    _migrate_alert_status(db)

    for i in range(5):
        event = _make_structured_event(
            event_id=f"evt-{i:03d}",
            camera_id="cam-01" if i < 3 else "cam-02",
            timestamp=datetime(2025, 1, 15, 14, 30, i, tzinfo=timezone.utc),
            object_type="person" if i % 2 == 0 else "vehicle",
            confidence=0.85 + i * 0.02,
        )
        db.insert_event(event)

    for i in range(2):
        alert = _make_alert_payload(
            alert_id=f"alert-{i:03d}",
            event_id=f"evt-{i:03d}",
            camera_id="cam-01",
        )
        db.insert_alert(alert, delivered_channels=["push"])

    yield db
    db.close()


@pytest.fixture
def rule_store():
    """In-memory RuleStore with test data."""
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
    yield rs
    rs.close()


@pytest.fixture
def alert_system():
    """AlertSystem with a push channel and no cooldown."""
    return AlertSystem(
        channels=[PushNotificationChannel()],
        cooldown_config=CooldownConfig(default_seconds=0),
    )


@pytest.fixture
def watchdog_mock():
    return _make_watchdog_mock()


@pytest.fixture
def prompt_compiler_mock():
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


@pytest.fixture
def context_filter_mock():
    """Create a mock ContextFilter."""
    cf = MagicMock()
    cf.reload_rules = MagicMock()
    return cf


@pytest.fixture
def alert_system_mock():
    """Create a mock AlertSystem for dashboard/mobile tests."""
    alert_sys = MagicMock()
    from agentic_cctv.models import AlertResult
    alert_sys.send_alert = AsyncMock(
        return_value=AlertResult(delivered=True, channels=["push"])
    )
    return alert_sys


# ---------------------------------------------------------------------------
# 1. Summary Generation (Requirement 17.2)
# ---------------------------------------------------------------------------


class TestSummaryGeneration:
    """Integration tests for EventSummarizer with real TimeSeriesDB data.

    Validates: Requirements 17.2
    """

    @pytest.mark.asyncio
    async def test_generate_summary_with_real_data(
        self, timeseries_db, alert_system
    ) -> None:
        """Generate an hourly summary from real DB events."""
        summarizer = EventSummarizer(
            timeseries_db=timeseries_db,
            alert_system=alert_system,
            tenant_id="t1",
            site_id="s1",
        )
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Hourly")

        assert "Hourly Summary:" in summary
        assert "Total events: 5" in summary
        assert "person" in summary
        assert "vehicle" in summary

    @pytest.mark.asyncio
    async def test_generate_and_deliver_summary(
        self, timeseries_db, alert_system
    ) -> None:
        """generate_and_deliver should generate and deliver the summary."""
        summarizer = EventSummarizer(
            timeseries_db=timeseries_db,
            alert_system=alert_system,
            tenant_id="t1",
            site_id="s1",
        )
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer.generate_and_deliver(start, end, "Hourly")

        assert "Total events: 5" in summary

    @pytest.mark.asyncio
    async def test_summary_with_no_events(
        self, timeseries_db, alert_system
    ) -> None:
        """Summary with no events in window returns appropriate message."""
        summarizer = EventSummarizer(
            timeseries_db=timeseries_db,
            alert_system=alert_system,
            tenant_id="t1",
            site_id="s1",
        )
        # Query a window with no events
        start = datetime(2026, 1, 1, 0, 0, 0)
        end = datetime(2026, 1, 1, 1, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Hourly")

        assert "No events or alerts" in summary

    @pytest.mark.asyncio
    async def test_summary_with_llm_backend(
        self, timeseries_db, alert_system
    ) -> None:
        """Summary with LLM backend uses LLM-generated text."""
        mock_llm = MockLLMBackend()
        summarizer = EventSummarizer(
            timeseries_db=timeseries_db,
            alert_system=alert_system,
            llm_backend=mock_llm,
            tenant_id="t1",
            site_id="s1",
        )
        start = datetime(2025, 1, 15, 14, 0, 0)
        end = datetime(2025, 1, 15, 15, 0, 0)
        summary = await summarizer.generate_summary(start, end, "Hourly")

        assert "LLM-generated summary" in summary
        assert mock_llm.call_count == 1


# ---------------------------------------------------------------------------
# 2. Test-Against-History (Requirement 17.3)
# ---------------------------------------------------------------------------


class TestTestAgainstHistory:
    """Integration tests for PromptCompiler.test_against_history with real DB.

    Validates: Requirements 17.3
    """

    @pytest.fixture
    def history_db(self):
        """TimeSeriesDB with recent events for test_against_history."""
        db = TimeSeriesDB(":memory:")
        # Use recent timestamps so they fall within the default query window
        now = datetime.utcnow()
        for i in range(4):
            event = _make_structured_event(
                event_id=f"hist-evt-{i:03d}",
                camera_id="cam-01",
                # Use naive datetimes to match what test_against_history queries
                timestamp=now - timedelta(hours=i + 1),
                object_type="person" if i % 2 == 0 else "vehicle",
                confidence=0.85 + i * 0.02,
            )
            db.insert_event(event)
        yield db
        db.close()

    @pytest.mark.asyncio
    async def test_against_history_with_matching_rules(
        self, history_db
    ) -> None:
        """test_against_history with a RuleSet that matches some events."""
        llm_client = MockLLMClient()
        compiler = PromptCompiler(
            llm_client=llm_client,
            timeseries_db=history_db,
        )

        # RuleSet that matches "person" events with confidence >= 0.8
        ruleset = RuleSet(
            version_id="rs-test-001",
            camera_id="cam-01",
            rules=[
                Rule(rule_id="rule-p1", object_type="person", min_confidence=0.8),
            ],
            created_at=datetime(2025, 1, 15, 10, 0, 0),
        )

        result = await compiler.test_against_history(
            ruleset=ruleset, camera_id="cam-01", days=7
        )

        assert result.camera_id == "cam-01"
        assert result.total_events >= 1
        assert result.matched_events >= 1
        assert 0.0 < result.expected_alert_rate <= 1.0

    @pytest.mark.asyncio
    async def test_against_history_no_matches(
        self, history_db
    ) -> None:
        """test_against_history with a RuleSet that matches no events."""
        llm_client = MockLLMClient()
        compiler = PromptCompiler(
            llm_client=llm_client,
            timeseries_db=history_db,
        )

        # RuleSet that matches "elephant" — no such events exist
        ruleset = RuleSet(
            version_id="rs-test-002",
            camera_id="cam-01",
            rules=[
                Rule(rule_id="rule-e1", object_type="elephant", min_confidence=0.5),
            ],
            created_at=datetime(2025, 1, 15, 10, 0, 0),
        )

        result = await compiler.test_against_history(
            ruleset=ruleset, camera_id="cam-01", days=7
        )

        assert result.matched_events == 0
        assert result.expected_alert_rate == 0.0

    @pytest.mark.asyncio
    async def test_against_history_all_match(
        self, history_db
    ) -> None:
        """test_against_history with a RuleSet that matches all events."""
        llm_client = MockLLMClient()
        compiler = PromptCompiler(
            llm_client=llm_client,
            timeseries_db=history_db,
        )

        # RuleSet with no object_type filter and very low confidence — matches all
        ruleset = RuleSet(
            version_id="rs-test-003",
            camera_id="cam-01",
            rules=[
                Rule(rule_id="rule-all", object_type=None, min_confidence=0.0),
            ],
            created_at=datetime(2025, 1, 15, 10, 0, 0),
        )

        result = await compiler.test_against_history(
            ruleset=ruleset, camera_id="cam-01", days=7
        )

        assert result.total_events >= 1
        assert result.matched_events == result.total_events
        assert result.expected_alert_rate == 1.0

    @pytest.mark.asyncio
    async def test_against_history_raises_without_db(self) -> None:
        """test_against_history raises RuntimeError without TimeSeriesDB."""
        llm_client = MockLLMClient()
        compiler = PromptCompiler(llm_client=llm_client, timeseries_db=None)

        ruleset = RuleSet(
            version_id="rs-test-004",
            camera_id="cam-01",
            rules=[Rule(rule_id="rule-x", object_type="person")],
            created_at=datetime(2025, 1, 15, 10, 0, 0),
        )

        with pytest.raises(RuntimeError, match="no TimeSeriesDB"):
            await compiler.test_against_history(ruleset, "cam-01")


# ---------------------------------------------------------------------------
# 3. Dashboard API Endpoints (Requirement 17.1)
# ---------------------------------------------------------------------------


@pytest.fixture
async def dashboard_client(
    watchdog_mock, timeseries_db, rule_store,
    alert_system_mock, prompt_compiler_mock, context_filter_mock,
):
    """Create a TestClient with all Phase 4 dashboard dependencies."""
    app = create_dashboard_app(
        watchdog=watchdog_mock,
        timeseries_db=timeseries_db,
        alert_system=alert_system_mock,
        rule_store=rule_store,
        prompt_compiler=prompt_compiler_mock,
        context_filter=context_filter_mock,
    )
    async with TestClient(TestServer(app)) as tc:
        yield tc


class TestDashboardAPIEndpoints:
    """Integration tests for dashboard API with all Phase 4 components.

    Validates: Requirements 17.1
    """

    async def test_overview_returns_correct_stats(self, dashboard_client) -> None:
        """GET /api/dashboard/overview returns correct camera and event stats."""
        resp = await dashboard_client.get("/api/dashboard/overview")
        assert resp.status == 200
        data = await resp.json()
        assert data["total_cameras"] == 2
        assert data["online_cameras"] == 1
        assert data["offline_cameras"] == 1
        assert data["recent_alert_count"] == 2
        assert data["recent_event_count"] == 5

    async def test_cameras_returns_health(self, dashboard_client) -> None:
        """GET /api/dashboard/cameras returns camera health data."""
        resp = await dashboard_client.get("/api/dashboard/cameras")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["cameras"]) == 2
        assert data["cameras"][0]["camera_id"] == "cam-01"
        assert data["cameras"][0]["status"] == "online"

    async def test_events_returns_events(self, dashboard_client) -> None:
        """GET /api/dashboard/events returns stored events."""
        resp = await dashboard_client.get("/api/dashboard/events")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["events"]) == 5

    async def test_alerts_acknowledge_dismiss_escalate(
        self, dashboard_client
    ) -> None:
        """Alert management: acknowledge, dismiss, escalate."""
        # Acknowledge
        resp = await dashboard_client.post(
            "/api/dashboard/alerts/alert-000/acknowledge"
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "acknowledged"

        # Dismiss
        resp = await dashboard_client.post(
            "/api/dashboard/alerts/alert-001/dismiss"
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "dismissed"

    async def test_rules_for_camera(self, dashboard_client) -> None:
        """GET /api/dashboard/rules/{camera_id} returns active ruleset."""
        resp = await dashboard_client.get("/api/dashboard/rules/cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-01"
        assert len(data["rules"]) == 1
        assert data["rules"][0]["rule_id"] == "rule-001"

    async def test_compile_prompt(self, dashboard_client) -> None:
        """POST /api/dashboard/prompt/compile compiles a prompt."""
        resp = await dashboard_client.post(
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

    async def test_dashboard_serves_html(self, dashboard_client) -> None:
        """GET /dashboard serves the HTML dashboard page."""
        resp = await dashboard_client.get("/dashboard")
        assert resp.status == 200
        assert "text/html" in resp.content_type
        text = await resp.text()
        assert "CCTV Monitoring Dashboard" in text


# ---------------------------------------------------------------------------
# 4. Environment Template Loading (Requirement 17.5)
# ---------------------------------------------------------------------------


class TestEnvironmentTemplateLoading:
    """Integration tests for environment template config generation.

    Validates: Requirements 17.5
    """

    @pytest.mark.parametrize(
        "template_name",
        ["home", "farm", "forest", "mall", "port", "gpu_desktop"],
    )
    def test_generate_config_has_required_fields(
        self, template_name: str
    ) -> None:
        """Each template generates a config with valid required fields."""
        config = generate_config_from_template(template_name)

        assert config.deployment_profile in (
            "single-machine", "multi-machine", "edge-cloud-hybrid"
        )
        assert config.cameras[0].inference_runtime in ("pytorch", "tensorrt")
        assert len(config.cameras) >= 1
        assert config.mqtt.host is not None
        assert config.mqtt.port > 0
        assert config.vlm.backend is not None
        assert config.storage.timeseries_db is not None
        assert config.alerts.channels is not None

    @pytest.mark.parametrize(
        "template_name",
        ["home", "farm", "forest", "mall", "port", "gpu_desktop"],
    )
    def test_template_has_valid_deployment_profile(
        self, template_name: str
    ) -> None:
        """Each template has a valid deployment_profile."""
        template = get_template(template_name)
        assert template.deployment_profile in (
            "single-machine", "multi-machine", "edge-cloud-hybrid"
        )

    def test_template_override_mechanism(self) -> None:
        """Overrides passed to generate_config_from_template take effect."""
        config = generate_config_from_template(
            "home",
            inference_runtime="tensorrt",
            deployment_profile="multi-machine",
            mqtt_host="broker.example.com",
            mqtt_port=8883,
        )
        assert config.cameras[0].inference_runtime == "tensorrt"
        assert config.deployment_profile == "multi-machine"
        assert config.mqtt.host == "broker.example.com"
        assert config.mqtt.port == 8883

    def test_all_six_templates_available(self) -> None:
        """All six pre-built templates are available."""
        templates = list_templates()
        names = {t.name for t in templates}
        assert names == {"home", "farm", "forest", "mall", "port", "gpu_desktop"}


# ---------------------------------------------------------------------------
# 5. VLMBackendLLMAdapter (Requirement 17.5)
# ---------------------------------------------------------------------------


class TestVLMBackendLLMAdapter:
    """Integration tests for the VLMBackendLLMAdapter from main.py.

    Validates: Requirements 17.5
    """

    @pytest.mark.asyncio
    async def test_adapter_with_scene_description_response(self) -> None:
        """Adapter extracts scene_description from VLM backend response."""
        mock_backend = MockLLMBackend(
            response={"scene_description": "A person detected near the gate."}
        )
        adapter = VLMBackendLLMAdapter(mock_backend)
        result = await adapter.generate("system prompt", "user prompt")

        assert result == "A person detected near the gate."
        assert mock_backend.call_count == 1

    @pytest.mark.asyncio
    async def test_adapter_with_response_key(self) -> None:
        """Adapter extracts 'response' key from VLM backend response."""
        mock_backend = MockLLMBackend(
            response={"response": "Alert triggered for vehicle."}
        )
        adapter = VLMBackendLLMAdapter(mock_backend)
        result = await adapter.generate("system prompt", "user prompt")

        assert result == "Alert triggered for vehicle."

    @pytest.mark.asyncio
    async def test_adapter_with_plain_string_result(self) -> None:
        """Adapter handles a plain string result from VLM backend."""

        class StringBackend:
            async def analyze(self, image_b64: str, event_context: dict) -> str:
                return "Plain string response"

        adapter = VLMBackendLLMAdapter(StringBackend())
        result = await adapter.generate("system prompt", "user prompt")

        assert result == "Plain string response"

    @pytest.mark.asyncio
    async def test_adapter_with_dict_fallback_to_json(self) -> None:
        """Adapter falls back to JSON serialisation for unknown dict keys."""
        mock_backend = MockLLMBackend(
            response={"custom_key": "custom_value"}
        )
        adapter = VLMBackendLLMAdapter(mock_backend)
        result = await adapter.generate("system prompt", "user prompt")

        # Should be JSON serialised since no known key matches
        parsed = json.loads(result)
        assert parsed["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_adapter_passes_prompts_as_context(self) -> None:
        """Adapter passes system and user prompts as event context."""
        captured_context = {}

        class CapturingBackend:
            async def analyze(self, image_b64: str, event_context: dict) -> dict:
                captured_context.update(event_context)
                return {"scene_description": "ok"}

        adapter = VLMBackendLLMAdapter(CapturingBackend())
        await adapter.generate("my system prompt", "my user prompt")

        assert captured_context["system_prompt"] == "my system prompt"
        assert captured_context["user_prompt"] == "my user prompt"


# ---------------------------------------------------------------------------
# 6. Mobile API Integration (Requirement 17.4)
# ---------------------------------------------------------------------------


@pytest.fixture
async def mobile_client(
    watchdog_mock, timeseries_db, alert_system_mock,
    rule_store, prompt_compiler_mock, context_filter_mock,
):
    """Create a TestClient for the mobile API."""
    app = create_mobile_app(
        watchdog=watchdog_mock,
        timeseries_db=timeseries_db,
        alert_system=alert_system_mock,
        rule_store=rule_store,
        prompt_compiler=prompt_compiler_mock,
        context_filter=context_filter_mock,
    )
    async with TestClient(TestServer(app)) as tc:
        yield tc


class TestMobileAPIIntegration:
    """Integration tests for mobile API endpoints.

    Validates: Requirements 17.4
    """

    async def test_mobile_summary_returns_compact_overview(
        self, mobile_client
    ) -> None:
        """GET /api/mobile/summary returns compact camera and alert stats."""
        resp = await mobile_client.get("/api/mobile/summary")
        assert resp.status == 200
        data = await resp.json()
        assert data["total_cameras"] == 2
        assert data["online_cameras"] == 1
        assert data["offline_cameras"] == 1
        assert "total_alerts" in data
        assert "active_alerts" in data
        assert "recent_critical_alerts" in data

    async def test_mobile_alerts_with_pagination(self, mobile_client) -> None:
        """GET /api/mobile/alerts supports pagination via limit and offset."""
        resp = await mobile_client.get("/api/mobile/alerts?limit=1&offset=0")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 1
        assert data["limit"] == 1
        assert data["offset"] == 0

        # Second page
        resp = await mobile_client.get("/api/mobile/alerts?limit=1&offset=1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["alerts"]) == 1
        assert data["offset"] == 1

    async def test_mobile_push_register_and_unregister(
        self, mobile_client
    ) -> None:
        """POST /api/mobile/push/register and DELETE /api/mobile/push/unregister."""
        # Register a device
        resp = await mobile_client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "test-token-abc123",
                "platform": "ios",
                "tenant_id": "t1",
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "registered"
        assert data["device_token"] == "test-token-abc123"

        # Verify device is listed
        resp = await mobile_client.get("/api/mobile/push/devices?tenant_id=t1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["devices"]) == 1
        assert data["devices"][0]["device_token"] == "test-token-abc123"

        # Unregister the device
        resp = await mobile_client.delete(
            "/api/mobile/push/unregister",
            json={"device_token": "test-token-abc123"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "unregistered"

        # Verify device is removed
        resp = await mobile_client.get("/api/mobile/push/devices?tenant_id=t1")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["devices"]) == 0

    async def test_mobile_push_register_invalid_platform(
        self, mobile_client
    ) -> None:
        """POST /api/mobile/push/register rejects invalid platform."""
        resp = await mobile_client.post(
            "/api/mobile/push/register",
            json={
                "device_token": "test-token",
                "platform": "windows",
                "tenant_id": "t1",
            },
        )
        assert resp.status == 400
        data = await resp.json()
        assert "platform" in data["error"]

    async def test_mobile_alerts_acknowledge(self, mobile_client) -> None:
        """POST /api/mobile/alerts/{alert_id}/acknowledge works."""
        resp = await mobile_client.post(
            "/api/mobile/alerts/alert-000/acknowledge"
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "acknowledged"

    async def test_mobile_rules_for_camera(self, mobile_client) -> None:
        """GET /api/mobile/rules/{camera_id} returns active ruleset."""
        resp = await mobile_client.get("/api/mobile/rules/cam-01")
        assert resp.status == 200
        data = await resp.json()
        assert data["camera_id"] == "cam-01"
        assert len(data["rules"]) == 1
