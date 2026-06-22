"""Integration tests for Phase 2 pipeline.

Validates end-to-end flows: event → Context Filter (pass/suppress) →
VLM Reasoner (with mocked backend) → Orchestration Agent → Alert System.
Also tests VLM fallback to rule-based classification.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_cctv.alert_system import AlertSystem, PushNotificationChannel
from agentic_cctv.context_filter import ContextFilter
from agentic_cctv.event_encoder import _structured_event_to_dict
from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    CooldownConfig,
    FilterResult,
    Rule,
    RuleSet,
    StructuredEvent,
    TimeWindow,
    Zone,
)
from agentic_cctv.orchestration_agent import (
    AlertTool,
    LogTool,
    MCPContextTool,
    A2ACommTool,
    OrchestrationAgent,
)
from agentic_cctv.phase2_pipeline import ContextFilterSubscriber
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB
from agentic_cctv.vlm_reasoner import VLMReasoner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structured_event(**overrides: object) -> StructuredEvent:
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
    return StructuredEvent(**defaults)  # type: ignore[arg-type]


def _make_ruleset(
    camera_id: str = "cam-lobby-01",
    rules: Optional[list[Rule]] = None,
) -> RuleSet:
    """Create a RuleSet with sensible defaults."""
    if rules is None:
        rules = [
            Rule(
                rule_id="rule-001",
                object_type="person",
                min_confidence=0.7,
                time_window=TimeWindow(start="22:00", end="06:00"),
                zone=Zone(polygon=[[0, 0], [640, 0], [640, 480], [0, 480]]),
            ),
        ]
    return RuleSet(
        version_id=f"rs-{uuid.uuid4().hex[:12]}",
        camera_id=camera_id,
        rules=rules,
        created_at=datetime.utcnow(),
    )


class MockVLMBackend:
    """A mock VLM backend that returns configurable responses."""

    def __init__(
        self,
        response: Optional[dict] = None,
        fail_count: int = 0,
    ) -> None:
        self._response = response or {
            "scene_description": "Person detected in restricted area after hours.",
            "threat_level": "high",
            "objects_identified": [
                {"type": "person", "action": "walking", "location": "lobby"},
            ],
            "recommended_action": "alert",
            "confidence": 0.88,
        }
        self._fail_count = fail_count
        self._call_count = 0

    async def analyze(self, image_b64: str, event_context: dict, **kwargs) -> dict:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Mock VLM failure #{self._call_count}")
        return self._response


class AlwaysFailVLMBackend:
    """A VLM backend that always fails, triggering rule-based fallback."""

    async def analyze(self, image_b64: str, event_context: dict, **kwargs) -> dict:
        raise RuntimeError("VLM backend unavailable")


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
    """AlertSystem with a push notification channel and 60s cooldown."""
    channel = PushNotificationChannel()
    cooldown = CooldownConfig(default_seconds=60)
    return AlertSystem(channels=[channel], cooldown_config=cooldown)


# ---------------------------------------------------------------------------
# 1. Event passes ContextFilter → VLMReasoner → OrchestrationAgent → AlertSystem
# ---------------------------------------------------------------------------


class TestPhase2EventPassesThroughPipeline:
    """Validates: Requirements 4.1, 5.1, 6.1, 15.1, 15.2, 15.3

    End-to-end: event with matching rule → ContextFilter passes →
    mocked VLMReasoner → OrchestrationAgent → AlertSystem.
    """

    @pytest.mark.asyncio
    async def test_event_passes_full_pipeline(
        self, timeseries_db, rule_store, alert_system
    ) -> None:
        """Event matching a rule flows through the entire Phase 2 pipeline."""
        # Set up a ruleset that matches our test event
        ruleset = _make_ruleset(camera_id="cam-lobby-01")
        rule_store.save_ruleset("cam-lobby-01", ruleset)

        # Create components
        context_filter = ContextFilter(
            rule_store=rule_store,
            timeseries_db=timeseries_db,
        )
        mock_backend = MockVLMBackend(
            response={
                "scene_description": "Person in restricted area after hours.",
                "threat_level": "high",
                "objects_identified": [
                    {"type": "person", "action": "walking", "location": "lobby"},
                ],
                "recommended_action": "alert",
                "confidence": 0.90,
            }
        )
        vlm_reasoner = VLMReasoner(
            backend=mock_backend,
            timeseries_db=timeseries_db,
        )
        orchestration_agent = OrchestrationAgent(
            tools=[
                AlertTool(alert_system),
                LogTool(timeseries_db),
                MCPContextTool(),
                A2ACommTool(),
            ]
        )

        # Create event (person, high confidence, nighttime → matches rule)
        event = _make_structured_event(
            event_id="evt-phase2-001",
            object_type="person",
            confidence=0.92,
            timestamp=datetime(2025, 1, 15, 23, 30, 0, tzinfo=timezone.utc),
        )

        # Step 1: ContextFilter should pass the event
        filter_result = context_filter.evaluate(event)
        assert filter_result.passed is True
        assert "rule-001" in filter_result.matched_rules

        # Step 2: VLMReasoner should produce a SceneUnderstanding
        scene = await vlm_reasoner.reason(event, filter_result.matched_rules)
        assert scene.event_id == "evt-phase2-001"
        assert scene.threat_level == "high"
        assert scene.recommended_action == "alert"

        # Step 3: OrchestrationAgent should decide "alert"
        action_result = orchestration_agent.process(scene, event)
        assert action_result.action == "alert"
        assert action_result.alert_payload is not None

        # Step 4: AlertSystem should deliver the alert
        alert_result = await alert_system.send_alert(action_result.alert_payload)
        assert alert_result.delivered is True
        assert "PushNotificationChannel" in alert_result.channels


# ---------------------------------------------------------------------------
# 2. Event suppressed by ContextFilter — no VLM invocation
# ---------------------------------------------------------------------------


class TestPhase2EventSuppressedByContextFilter:
    """Validates: Requirements 4.1, 4.3, 15.1

    Event that does not match any rule is suppressed by ContextFilter.
    VLMReasoner should NOT be invoked.
    """

    def test_event_suppressed_no_vlm_invocation(
        self, timeseries_db, rule_store
    ) -> None:
        """Event not matching any rule is suppressed; VLM is not called."""
        # Set up a ruleset that only matches "vehicle"
        ruleset = _make_ruleset(
            camera_id="cam-lobby-01",
            rules=[
                Rule(
                    rule_id="rule-vehicle",
                    object_type="vehicle",
                    min_confidence=0.5,
                ),
            ],
        )
        rule_store.save_ruleset("cam-lobby-01", ruleset)

        context_filter = ContextFilter(
            rule_store=rule_store,
            timeseries_db=timeseries_db,
        )

        # Create a "person" event — should NOT match the "vehicle" rule
        event = _make_structured_event(
            event_id="evt-suppressed-001",
            object_type="person",
            confidence=0.85,
        )

        filter_result = context_filter.evaluate(event)
        assert filter_result.passed is False
        assert filter_result.suppressed_reason == "no_matching_rules"

        # Verify the suppressed event was logged to TimeSeriesDB
        rows = timeseries_db.get_events(camera_id="cam-lobby-01")
        assert len(rows) == 1
        assert rows[0]["event_id"] == "evt-suppressed-001"
        assert rows[0]["context_gate_passed"] == 0  # False

    def test_event_suppressed_no_active_ruleset(
        self, timeseries_db, rule_store
    ) -> None:
        """Event with no active ruleset is suppressed."""
        context_filter = ContextFilter(
            rule_store=rule_store,
            timeseries_db=timeseries_db,
        )

        event = _make_structured_event(
            event_id="evt-no-rules-001",
            object_type="person",
        )

        filter_result = context_filter.evaluate(event)
        assert filter_result.passed is False
        assert filter_result.suppressed_reason == "no_active_ruleset"


# ---------------------------------------------------------------------------
# 3. VLM fallback to rule-based classification
# ---------------------------------------------------------------------------


class TestPhase2VLMFallback:
    """Validates: Requirements 5.1, 5.6, 15.2

    When VLM backend fails twice, the VLMReasoner falls back to
    rule-based classification.
    """

    @pytest.mark.asyncio
    async def test_vlm_fallback_on_double_failure(
        self, timeseries_db, rule_store, alert_system
    ) -> None:
        """VLM fails twice → rule-based fallback → OrchestrationAgent processes."""
        # Set up ruleset
        ruleset = _make_ruleset(camera_id="cam-lobby-01")
        rule_store.save_ruleset("cam-lobby-01", ruleset)

        context_filter = ContextFilter(
            rule_store=rule_store,
            timeseries_db=timeseries_db,
        )

        # VLM backend that always fails
        vlm_reasoner = VLMReasoner(
            backend=AlwaysFailVLMBackend(),
            timeseries_db=timeseries_db,
        )
        orchestration_agent = OrchestrationAgent(
            tools=[
                AlertTool(alert_system),
                LogTool(timeseries_db),
                MCPContextTool(),
                A2ACommTool(),
            ]
        )

        event = _make_structured_event(
            event_id="evt-fallback-001",
            object_type="person",
            confidence=0.92,
            timestamp=datetime(2025, 1, 15, 23, 30, 0, tzinfo=timezone.utc),
        )

        # ContextFilter passes
        filter_result = context_filter.evaluate(event)
        assert filter_result.passed is True

        # VLMReasoner should fall back to rule-based classification
        scene = await vlm_reasoner.reason(event, filter_result.matched_rules)
        assert scene.event_id == "evt-fallback-001"
        assert "Rule-based fallback" in scene.scene_description
        # High confidence (0.92) → threat_level "high" in fallback
        assert scene.threat_level == "high"
        assert scene.recommended_action == "log"  # fallback always recommends "log"

        # OrchestrationAgent processes the fallback result
        action_result = orchestration_agent.process(scene, event)
        # threat_level "high" → action "alert" per OrchestrationAgent logic
        assert action_result.action == "alert"

    @pytest.mark.asyncio
    async def test_vlm_retry_succeeds_on_second_attempt(
        self, timeseries_db
    ) -> None:
        """VLM fails once, succeeds on retry — no fallback needed."""
        mock_backend = MockVLMBackend(
            response={
                "scene_description": "Person detected near entrance.",
                "threat_level": "medium",
                "objects_identified": [
                    {"type": "person", "action": "standing", "location": "entrance"},
                ],
                "recommended_action": "log",
                "confidence": 0.75,
            },
            fail_count=1,  # Fail on first call, succeed on second
        )
        vlm_reasoner = VLMReasoner(
            backend=mock_backend,
            timeseries_db=timeseries_db,
        )

        event = _make_structured_event(
            event_id="evt-retry-001",
            confidence=0.80,
        )

        scene = await vlm_reasoner.reason(event)
        assert scene.event_id == "evt-retry-001"
        assert scene.threat_level == "medium"
        assert "Rule-based fallback" not in scene.scene_description
        assert mock_backend._call_count == 2  # First failed, second succeeded


# ---------------------------------------------------------------------------
# 4. Different threat levels produce different actions
# ---------------------------------------------------------------------------


class TestPhase2ThreatLevelActions:
    """Validates: Requirements 6.1, 15.3

    Different threat levels from VLM produce different orchestration actions.
    """

    @pytest.mark.asyncio
    async def test_critical_threat_produces_alert(
        self, timeseries_db, alert_system
    ) -> None:
        """Critical threat level → alert action."""
        mock_backend = MockVLMBackend(
            response={
                "scene_description": "Armed intruder detected.",
                "threat_level": "critical",
                "objects_identified": [
                    {"type": "person", "action": "running", "location": "entrance"},
                ],
                "recommended_action": "escalate",
                "confidence": 0.95,
            }
        )
        vlm_reasoner = VLMReasoner(backend=mock_backend, timeseries_db=timeseries_db)
        orchestration_agent = OrchestrationAgent(
            tools=[AlertTool(alert_system), LogTool(timeseries_db)]
        )

        event = _make_structured_event(event_id="evt-critical-001", confidence=0.95)
        scene = await vlm_reasoner.reason(event)
        action_result = orchestration_agent.process(scene, event)

        assert action_result.action == "alert"
        assert action_result.alert_payload is not None

    @pytest.mark.asyncio
    async def test_low_threat_produces_log(
        self, timeseries_db, alert_system
    ) -> None:
        """Low threat level → log action."""
        mock_backend = MockVLMBackend(
            response={
                "scene_description": "Cat walking across parking lot.",
                "threat_level": "low",
                "objects_identified": [
                    {"type": "animal", "action": "walking", "location": "parking lot"},
                ],
                "recommended_action": "log",
                "confidence": 0.60,
            }
        )
        vlm_reasoner = VLMReasoner(backend=mock_backend, timeseries_db=timeseries_db)
        orchestration_agent = OrchestrationAgent(
            tools=[AlertTool(alert_system), LogTool(timeseries_db)]
        )

        event = _make_structured_event(event_id="evt-low-001", confidence=0.60)
        scene = await vlm_reasoner.reason(event)
        action_result = orchestration_agent.process(scene, event)

        assert action_result.action == "log"
        assert action_result.alert_payload is None

    @pytest.mark.asyncio
    async def test_medium_threat_uses_recommended_action(
        self, timeseries_db, alert_system
    ) -> None:
        """Medium threat level → uses VLM's recommended_action."""
        mock_backend = MockVLMBackend(
            response={
                "scene_description": "Unattended package near entrance.",
                "threat_level": "medium",
                "objects_identified": [
                    {"type": "package", "action": "stationary", "location": "entrance"},
                ],
                "recommended_action": "summarise",
                "confidence": 0.72,
            }
        )
        vlm_reasoner = VLMReasoner(backend=mock_backend, timeseries_db=timeseries_db)
        orchestration_agent = OrchestrationAgent(
            tools=[AlertTool(alert_system), LogTool(timeseries_db)]
        )

        event = _make_structured_event(event_id="evt-medium-001", confidence=0.72)
        scene = await vlm_reasoner.reason(event)
        action_result = orchestration_agent.process(scene, event)

        assert action_result.action == "summarise"


# ---------------------------------------------------------------------------
# 5. ContextFilterSubscriber end-to-end via MQTT callback
# ---------------------------------------------------------------------------


class TestContextFilterSubscriberCallback:
    """Validates: Requirements 4.1, 5.1, 6.1, 15.1, 15.2

    Test the ContextFilterSubscriber MQTT callback class end-to-end.
    """

    @pytest.mark.asyncio
    async def test_subscriber_processes_passing_event(
        self, timeseries_db, rule_store, alert_system
    ) -> None:
        """ContextFilterSubscriber forwards passing events through the pipeline."""
        # Set up ruleset
        ruleset = _make_ruleset(camera_id="cam-lobby-01")
        rule_store.save_ruleset("cam-lobby-01", ruleset)

        context_filter = ContextFilter(
            rule_store=rule_store,
            timeseries_db=timeseries_db,
        )
        mock_backend = MockVLMBackend(
            response={
                "scene_description": "Person in restricted area.",
                "threat_level": "high",
                "objects_identified": [
                    {"type": "person", "action": "walking", "location": "lobby"},
                ],
                "recommended_action": "alert",
                "confidence": 0.88,
            }
        )
        vlm_reasoner = VLMReasoner(
            backend=mock_backend,
            timeseries_db=timeseries_db,
        )
        orchestration_agent = OrchestrationAgent(
            tools=[
                AlertTool(alert_system),
                LogTool(timeseries_db),
                MCPContextTool(),
                A2ACommTool(),
            ]
        )

        subscriber = ContextFilterSubscriber(
            context_filter=context_filter,
            vlm_reasoner=vlm_reasoner,
            orchestration_agent=orchestration_agent,
            alert_system=alert_system,
            loop=None,
        )

        # Create event and serialize to JSON
        event = _make_structured_event(
            event_id="evt-callback-001",
            object_type="person",
            confidence=0.92,
            timestamp=datetime(2025, 1, 15, 23, 30, 0, tzinfo=timezone.utc),
        )
        payload = json.dumps(_structured_event_to_dict(event)).encode("utf-8")

        # Deserialize and evaluate directly (simulates what __call__ does)
        deserialized = subscriber._deserialize_event(payload)
        filter_result = context_filter.evaluate(deserialized)
        assert filter_result.passed is True

        # Call the async pipeline directly (avoids sync-to-async bridge issues in tests)
        await subscriber._run_pipeline(deserialized, filter_result.matched_rules)

        # Verify the VLM backend was called
        assert mock_backend._call_count == 1

    @pytest.mark.asyncio
    async def test_subscriber_suppresses_non_matching_event(
        self, timeseries_db, rule_store, alert_system
    ) -> None:
        """ContextFilterSubscriber suppresses events that don't match rules."""
        # Set up ruleset that only matches "vehicle"
        ruleset = _make_ruleset(
            camera_id="cam-lobby-01",
            rules=[
                Rule(rule_id="rule-vehicle", object_type="vehicle", min_confidence=0.5),
            ],
        )
        rule_store.save_ruleset("cam-lobby-01", ruleset)

        context_filter = ContextFilter(
            rule_store=rule_store,
            timeseries_db=timeseries_db,
        )
        mock_backend = MockVLMBackend()
        vlm_reasoner = VLMReasoner(
            backend=mock_backend,
            timeseries_db=timeseries_db,
        )
        orchestration_agent = OrchestrationAgent()

        subscriber = ContextFilterSubscriber(
            context_filter=context_filter,
            vlm_reasoner=vlm_reasoner,
            orchestration_agent=orchestration_agent,
            alert_system=alert_system,
            loop=None,
        )

        # Create a "person" event — should NOT match the "vehicle" rule
        event = _make_structured_event(
            event_id="evt-suppress-001",
            object_type="person",
            confidence=0.85,
        )
        payload = json.dumps(_structured_event_to_dict(event)).encode("utf-8")
        topic = "tenant-acme/site-hq/cam-lobby-01/events"

        subscriber(topic, payload, qos=1)

        # VLM backend should NOT have been called
        assert mock_backend._call_count == 0

        # Suppressed event should be logged to TimeSeriesDB
        rows = timeseries_db.get_events(camera_id="cam-lobby-01")
        assert len(rows) == 1
        assert rows[0]["context_gate_passed"] == 0

    def test_subscriber_handles_invalid_payload(
        self, timeseries_db, rule_store, alert_system
    ) -> None:
        """ContextFilterSubscriber gracefully handles invalid JSON payloads."""
        context_filter = ContextFilter(
            rule_store=rule_store,
            timeseries_db=timeseries_db,
        )
        mock_backend = MockVLMBackend()
        vlm_reasoner = VLMReasoner(backend=mock_backend)
        orchestration_agent = OrchestrationAgent()

        subscriber = ContextFilterSubscriber(
            context_filter=context_filter,
            vlm_reasoner=vlm_reasoner,
            orchestration_agent=orchestration_agent,
            alert_system=alert_system,
            loop=None,
        )

        # Invalid JSON payload — should not raise
        subscriber("some/topic", b"not-valid-json", qos=1)
        assert mock_backend._call_count == 0
