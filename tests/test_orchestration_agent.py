"""Unit tests for the OrchestrationAgent.

Tests:
- Action decisions for each threat level (none, low, medium, high, critical)
- Tool fallback on failure (AlertTool / LogTool failure → default action)
- MCP/A2A stub behaviour (no-op, return empty results without errors)
- Agent always returns a valid ActionResult

Requirements: 6.1, 6.5
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agentic_cctv.models import (
    ActionResult,
    BoundingBox,
    IdentifiedObject,
    SceneUnderstanding,
    StructuredEvent,
)
from agentic_cctv.orchestration_agent import (
    A2ACommTool,
    AlertTool,
    LogTool,
    MCPContextTool,
    OrchestrationAgent,
    ToolResult,
    VALID_ACTIONS,
    VectorSearchTool,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_event(event_id: str = "evt-001") -> StructuredEvent:
    """Create a minimal StructuredEvent for testing."""
    return StructuredEvent(
        event_id=event_id,
        camera_id="cam-01",
        tenant_id="tenant-a",
        site_id="site-hq",
        timestamp=datetime(2025, 1, 15, 14, 30, 0),
        object_type="person",
        track_id="trk-001",
        bounding_box=BoundingBox(x=10, y=20, width=100, height=200),
        confidence=0.9,
        frame_crop="base64data",
    )


def _make_scene(
    threat_level: str = "medium",
    recommended_action: str = "alert",
    event_id: str = "evt-001",
) -> SceneUnderstanding:
    """Create a minimal SceneUnderstanding for testing."""
    return SceneUnderstanding(
        event_id=event_id,
        scene_description="Test scene",
        threat_level=threat_level,
        objects_identified=[
            IdentifiedObject(type="person", action="walking", location="lobby")
        ],
        recommended_action=recommended_action,
        confidence=0.85,
        raw_response={},
        embedding=None,
    )


# ---------------------------------------------------------------------------
# Failing tool stubs for fallback testing
# ---------------------------------------------------------------------------


class FailingAlertTool:
    """An AlertTool that always fails."""

    @property
    def name(self) -> str:
        return "AlertTool"

    def invoke(self, scene: SceneUnderstanding, event: StructuredEvent) -> ToolResult:
        return ToolResult(success=False, error="Alert delivery failed")


class FailingLogTool:
    """A LogTool that always fails."""

    @property
    def name(self) -> str:
        return "LogTool"

    def invoke(self, scene: SceneUnderstanding, event: StructuredEvent) -> ToolResult:
        return ToolResult(success=False, error="DB write failed")


class ExceptionAlertTool:
    """An AlertTool that raises an exception."""

    @property
    def name(self) -> str:
        return "AlertTool"

    def invoke(self, scene: SceneUnderstanding, event: StructuredEvent) -> ToolResult:
        raise RuntimeError("Unexpected alert error")


# ---------------------------------------------------------------------------
# Tests: Action decisions for each threat level
# ---------------------------------------------------------------------------


class TestDecideAction:
    """Test action decision logic for each threat level."""

    def test_critical_threat_returns_alert(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="critical", recommended_action="log")
        assert agent.decide_action(scene) == "alert"

    def test_high_threat_returns_alert(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="high", recommended_action="log")
        assert agent.decide_action(scene) == "alert"

    def test_medium_threat_uses_recommended_action_alert(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="medium", recommended_action="alert")
        assert agent.decide_action(scene) == "alert"

    def test_medium_threat_uses_recommended_action_summarise(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="medium", recommended_action="summarise")
        assert agent.decide_action(scene) == "summarise"

    def test_medium_threat_uses_recommended_action_escalate(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="medium", recommended_action="escalate")
        assert agent.decide_action(scene) == "escalate"

    def test_medium_threat_uses_recommended_action_log(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="medium", recommended_action="log")
        assert agent.decide_action(scene) == "log"

    def test_low_threat_returns_log(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="low", recommended_action="alert")
        assert agent.decide_action(scene) == "log"

    def test_none_threat_returns_log(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="none", recommended_action="alert")
        assert agent.decide_action(scene) == "log"

    def test_medium_threat_invalid_recommended_action_defaults_to_log(self) -> None:
        """When recommended_action is not in the valid set, default to 'log'."""
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="medium", recommended_action="invalid_action")
        assert agent.decide_action(scene) == "log"

    def test_unknown_threat_level_defaults_to_log(self) -> None:
        """Unknown threat levels should default to 'log'."""
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="unknown", recommended_action="alert")
        assert agent.decide_action(scene) == "log"


# ---------------------------------------------------------------------------
# Tests: Tool fallback on failure
# ---------------------------------------------------------------------------


class TestToolFallback:
    """Test that tool failures trigger fallback to default action."""

    def test_alert_tool_failure_falls_back_for_high_threat(self) -> None:
        """When AlertTool fails for high threat, fallback should be 'alert'."""
        agent = OrchestrationAgent(tools=[FailingAlertTool()])
        scene = _make_scene(threat_level="high")
        event = _make_event()
        result = agent.process(scene, event)
        assert result.action in VALID_ACTIONS
        # Fallback for high threat is still "alert"
        assert result.action == "alert"

    def test_alert_tool_failure_falls_back_for_medium_threat(self) -> None:
        """When AlertTool fails for medium threat with recommended_action='alert',
        fallback should be 'log'."""
        agent = OrchestrationAgent(tools=[FailingAlertTool()])
        scene = _make_scene(threat_level="medium", recommended_action="alert")
        event = _make_event()
        result = agent.process(scene, event)
        assert result.action in VALID_ACTIONS
        # Fallback for medium threat is "log"
        assert result.action == "log"

    def test_log_tool_failure_falls_back_for_low_threat(self) -> None:
        """When LogTool fails for low threat, fallback should be 'log'."""
        agent = OrchestrationAgent(tools=[FailingLogTool()])
        scene = _make_scene(threat_level="low")
        event = _make_event()
        result = agent.process(scene, event)
        assert result.action in VALID_ACTIONS
        assert result.action == "log"

    def test_alert_tool_exception_falls_back(self) -> None:
        """When AlertTool raises an exception, agent should still return valid result."""
        agent = OrchestrationAgent(tools=[ExceptionAlertTool()])
        scene = _make_scene(threat_level="critical")
        event = _make_event()
        result = agent.process(scene, event)
        assert result.action in VALID_ACTIONS
        assert isinstance(result, ActionResult)


# ---------------------------------------------------------------------------
# Tests: MCP/A2A stub behaviour
# ---------------------------------------------------------------------------


class TestMCPAndA2AStubs:
    """Test that MCP and A2A stubs return empty/no-op results without errors."""

    def test_mcp_context_tool_returns_success(self) -> None:
        tool = MCPContextTool()
        scene = _make_scene()
        event = _make_event()
        result = tool.invoke(scene, event)
        assert result.success is True
        # Functional MCPContextTool now returns cross-camera context data
        assert "cross_camera_count" in result.data
        assert result.data["cross_camera_count"] == 0
        assert result.error is None

    def test_a2a_comm_tool_returns_success(self) -> None:
        tool = A2ACommTool()
        scene = _make_scene()
        event = _make_event()
        result = tool.invoke(scene, event)
        assert result.success is True
        # Functional A2ACommTool now returns received message data
        assert "received_count" in result.data
        assert result.data["received_count"] == 0
        assert result.error is None

    def test_mcp_tool_name(self) -> None:
        assert MCPContextTool().name == "MCPContextTool"

    def test_a2a_tool_name(self) -> None:
        assert A2ACommTool().name == "A2ACommTool"

    def test_stubs_integrated_in_agent(self) -> None:
        """MCP and A2A stubs should work when registered in the agent."""
        agent = OrchestrationAgent(tools=[MCPContextTool(), A2ACommTool()])
        scene = _make_scene(threat_level="low")
        event = _make_event()
        result = agent.process(scene, event)
        assert result.action in VALID_ACTIONS
        assert isinstance(result, ActionResult)


# ---------------------------------------------------------------------------
# Tests: Agent always returns valid ActionResult
# ---------------------------------------------------------------------------


class TestActionResultValidity:
    """Test that the agent always returns a valid ActionResult."""

    def test_process_returns_action_result(self) -> None:
        agent = OrchestrationAgent()
        scene = _make_scene()
        event = _make_event()
        result = agent.process(scene, event)
        assert isinstance(result, ActionResult)
        assert result.action in VALID_ACTIONS

    def test_process_with_all_tools(self) -> None:
        """Agent with all tools registered should return valid result."""
        agent = OrchestrationAgent(
            tools=[
                MCPContextTool(),
                A2ACommTool(),
            ]
        )
        scene = _make_scene(threat_level="high")
        event = _make_event()
        result = agent.process(scene, event)
        assert isinstance(result, ActionResult)
        assert result.action in VALID_ACTIONS

    def test_process_no_tools_still_returns_valid_result(self) -> None:
        """Agent with no tools should still decide an action and return."""
        agent = OrchestrationAgent()
        scene = _make_scene(threat_level="critical")
        event = _make_event()
        result = agent.process(scene, event)
        assert isinstance(result, ActionResult)
        assert result.action == "alert"

    def test_register_tool(self) -> None:
        """Test that tools can be registered after construction."""
        agent = OrchestrationAgent()
        agent.register_tool(MCPContextTool())
        agent.register_tool(A2ACommTool())
        scene = _make_scene(threat_level="none")
        event = _make_event()
        result = agent.process(scene, event)
        assert result.action == "log"

    def test_cross_camera_refs_is_list(self) -> None:
        """cross_camera_refs should always be a list."""
        agent = OrchestrationAgent()
        scene = _make_scene()
        event = _make_event()
        result = agent.process(scene, event)
        assert isinstance(result.cross_camera_refs, list)
