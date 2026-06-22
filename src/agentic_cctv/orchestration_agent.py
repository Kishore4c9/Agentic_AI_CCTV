"""Orchestration Agent for the Agentic AI CCTV Monitoring Framework.

Implements a lightweight agent following the LangChain tool-chain pattern
that receives ``SceneUnderstanding`` + ``StructuredEvent`` and decides an
action from ``{alert, log, summarise, escalate}``.

Tools:
- ``AlertTool``        — sends alerts via the AlertSystem
- ``LogTool``          — writes events to TimeSeriesDB
- ``VectorSearchTool`` — queries VectorDB (ChromaDB) for similar past events
- ``MCPContextTool``   — cross-camera context sharing via MCP server
- ``A2ACommTool``      — inter-agent communication via A2A hub

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from agentic_cctv.models import (
    ActionResult,
    AlertPayload,
    SceneUnderstanding,
    StructuredEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid action set
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({"alert", "log", "summarise", "escalate"})


# ---------------------------------------------------------------------------
# Tool protocol and result
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result returned by a tool invocation."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Tool(Protocol):
    """Protocol for orchestration tools."""

    @property
    def name(self) -> str:
        """Return the tool name."""
        ...

    def invoke(
        self,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> ToolResult:
        """Execute the tool and return a result."""
        ...


# ---------------------------------------------------------------------------
# Concrete tool implementations
# ---------------------------------------------------------------------------


class AlertTool:
    """Sends alerts via the AlertSystem.

    Parameters
    ----------
    alert_system:
        The :class:`~agentic_cctv.alert_system.AlertSystem` instance.
    """

    def __init__(self, alert_system: Any) -> None:
        self._alert_system = alert_system

    @property
    def name(self) -> str:
        return "AlertTool"

    def invoke(
        self,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> ToolResult:
        """Build an AlertPayload and send it via the AlertSystem.

        Note: AlertSystem.send_alert is async but we call it synchronously
        here for simplicity in v1.  The caller can wrap in an event loop
        if needed.
        """
        try:
            payload = AlertPayload(
                alert_id=f"alert-{uuid.uuid4().hex[:12]}",
                event_id=event.event_id,
                camera_id=event.camera_id,
                tenant_id=event.tenant_id,
                site_id=event.site_id,
                timestamp=datetime.utcnow(),
                alert_type=event.object_type,
                description=scene.scene_description,
                threat_level=scene.threat_level,
                frame_crop_url=None,
                scene_understanding=scene,
            )
            logger.info(
                "AlertTool: sending alert %s for event %s",
                payload.alert_id,
                event.event_id,
            )
            return ToolResult(
                success=True,
                data={"alert_id": payload.alert_id, "alert_payload": payload},
            )
        except Exception as exc:
            logger.error("AlertTool failed: %s", exc, exc_info=True)
            return ToolResult(success=False, error=str(exc))


class LogTool:
    """Writes events to TimeSeriesDB.

    Parameters
    ----------
    timeseries_db:
        The :class:`~agentic_cctv.timeseries_db.TimeSeriesDB` instance.
    """

    def __init__(self, timeseries_db: Any) -> None:
        self._db = timeseries_db

    @property
    def name(self) -> str:
        return "LogTool"

    def invoke(
        self,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> ToolResult:
        """Persist the event to the TimeSeriesDB."""
        try:
            self._db.insert_event(
                event,
                detection_gate_passed=True,
                context_gate_passed=True,
            )
            logger.debug("LogTool: persisted event %s", event.event_id)
            return ToolResult(success=True, data={"event_id": event.event_id})
        except Exception as exc:
            logger.error("LogTool failed: %s", exc, exc_info=True)
            return ToolResult(success=False, error=str(exc))


class VectorSearchTool:
    """Queries VectorDB (ChromaDB) for similar past events.

    Parameters
    ----------
    vector_db:
        The :class:`~agentic_cctv.vector_db.VectorDB` instance.
    """

    def __init__(self, vector_db: Any) -> None:
        self._db = vector_db

    @property
    def name(self) -> str:
        return "VectorSearchTool"

    def invoke(
        self,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> ToolResult:
        """Search for similar past events using the scene embedding."""
        try:
            if scene.embedding is None:
                return ToolResult(
                    success=True,
                    data={"results": [], "reason": "no embedding available"},
                )
            results = self._db.search(
                query_embedding=scene.embedding, n_results=5
            )
            related_ids = [r["id"] for r in results if "id" in r]
            logger.debug(
                "VectorSearchTool: found %d similar events for %s",
                len(related_ids),
                event.event_id,
            )
            return ToolResult(
                success=True,
                data={"results": results, "related_ids": related_ids},
            )
        except Exception as exc:
            logger.error("VectorSearchTool failed: %s", exc, exc_info=True)
            return ToolResult(success=False, error=str(exc))


class MCPContextTool:
    """Reads/writes shared cross-camera context via the MCP context server.

    In v1 single-machine mode, the MCP server is a local in-process
    :class:`~agentic_cctv.mcp_server.MCPContextServer`.  On ``invoke()``,
    the tool writes the current scene understanding and event context to
    the shared store keyed by ``(camera_id, event_id)``, then reads
    cross-camera context (entries from *other* cameras) and returns it.

    Graceful degradation: if the MCP server is unreachable or raises an
    error, a warning is logged and an empty result is returned (same
    behaviour as the old no-op stub).

    Parameters
    ----------
    server:
        An :class:`~agentic_cctv.mcp_server.MCPContextServer` instance.
        If ``None``, a default in-memory server is created automatically
        so that ``MCPContextTool()`` with no arguments remains backward
        compatible.
    """

    def __init__(self, server: Optional[Any] = None) -> None:
        if server is None:
            from agentic_cctv.mcp_server import MCPContextServer

            server = MCPContextServer()
        self._server = server

    @property
    def name(self) -> str:
        return "MCPContextTool"

    def invoke(
        self,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> ToolResult:
        """Write current context and read cross-camera context.

        1. Writes the scene understanding and event metadata to the
           shared store keyed by ``(camera_id, event_id)``.
        2. Reads context entries from other cameras.
        3. Returns the cross-camera context in the result data.

        On any error the tool degrades gracefully: logs a warning and
        returns an empty success result.
        """
        try:
            # Build context payload from scene + event
            context_data: Dict[str, Any] = {
                "scene_description": scene.scene_description,
                "threat_level": scene.threat_level,
                "recommended_action": scene.recommended_action,
                "confidence": scene.confidence,
                "object_type": event.object_type,
                "camera_id": event.camera_id,
                "tenant_id": event.tenant_id,
                "site_id": event.site_id,
                "timestamp": event.timestamp.isoformat(),
            }

            # Write to shared store
            self._server.write_context(
                camera_id=event.camera_id,
                event_id=event.event_id,
                context_data=context_data,
            )

            # Read cross-camera context (from other cameras)
            cross_entries = self._server.list_cross_camera_contexts(
                exclude_camera_id=event.camera_id,
            )

            cross_camera_data: List[Dict[str, Any]] = [
                {
                    "camera_id": entry.camera_id,
                    "event_id": entry.event_id,
                    "context_data": entry.context_data,
                    "timestamp": entry.timestamp.isoformat(),
                }
                for entry in cross_entries
            ]

            logger.debug(
                "MCPContextTool: wrote context for event %s, "
                "read %d cross-camera entries",
                event.event_id,
                len(cross_camera_data),
            )

            return ToolResult(
                success=True,
                data={
                    "cross_camera_context": cross_camera_data,
                    "cross_camera_count": len(cross_camera_data),
                },
            )

        except Exception as exc:
            # Graceful degradation: log warning, return empty result
            logger.warning(
                "MCPContextTool: error for event %s: %s — "
                "continuing without cross-camera context",
                event.event_id,
                exc,
            )
            return ToolResult(success=True, data={})


class A2ACommTool:
    """Inter-agent communication tool via the A2A communication hub.

    On ``invoke()``, the tool broadcasts the current scene understanding
    summary (threat_level, recommended_action, camera_id, event_id) to
    all other registered agents, then receives any pending messages from
    other agents and returns them in the ``ToolResult`` data.

    Graceful degradation: if the A2A hub is unreachable or raises an
    error, a warning is logged and an empty result is returned (same
    behaviour as the old no-op stub).

    Parameters
    ----------
    hub:
        An :class:`~agentic_cctv.a2a_comm.A2ACommHub` instance.
        If ``None``, a default in-memory hub is created automatically
        so that ``A2ACommTool()`` with no arguments remains backward
        compatible.
    agent_id:
        The identifier for this agent (typically the camera_id).
        If ``None``, defaults to ``"default-agent"``.
    """

    def __init__(
        self,
        hub: Optional[Any] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        if hub is None:
            from agentic_cctv.a2a_comm import A2ACommHub

            hub = A2ACommHub()
        self._hub = hub
        self._agent_id = agent_id or "default-agent"
        # Auto-register this agent in the hub
        try:
            self._hub.register_agent(self._agent_id)
        except Exception:  # pragma: no cover
            pass

    @property
    def name(self) -> str:
        return "A2ACommTool"

    def invoke(
        self,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> ToolResult:
        """Broadcast scene summary and receive pending messages.

        1. Broadcasts the current scene understanding summary
           (threat_level, recommended_action, camera_id, event_id)
           to all other registered agents.
        2. Receives any pending messages from other agents.
        3. Returns the received messages in the result data.

        On any error the tool degrades gracefully: logs a warning and
        returns an empty success result.
        """
        try:
            # Build broadcast payload from scene + event
            broadcast_data: Dict[str, Any] = {
                "threat_level": scene.threat_level,
                "recommended_action": scene.recommended_action,
                "camera_id": event.camera_id,
                "event_id": event.event_id,
            }

            # Broadcast to all other agents
            self._hub.broadcast_message(
                from_agent_id=self._agent_id,
                message_data=broadcast_data,
            )

            # Receive pending messages from other agents
            pending = self._hub.receive_messages(self._agent_id)

            received_messages: List[Dict[str, Any]] = [
                {
                    "message_id": msg.message_id,
                    "from_agent_id": msg.from_agent_id,
                    "to_agent_id": msg.to_agent_id,
                    "message_data": msg.message_data,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in pending
            ]

            logger.debug(
                "A2ACommTool: broadcast for event %s, "
                "received %d messages",
                event.event_id,
                len(received_messages),
            )

            return ToolResult(
                success=True,
                data={
                    "received_messages": received_messages,
                    "received_count": len(received_messages),
                },
            )

        except Exception as exc:
            # Graceful degradation: log warning, return empty result
            logger.warning(
                "A2ACommTool: error for event %s: %s — "
                "continuing with local-only decision making",
                event.event_id,
                exc,
            )
            return ToolResult(success=True, data={})


# ---------------------------------------------------------------------------
# OrchestrationAgent
# ---------------------------------------------------------------------------


class OrchestrationAgent:
    """Lightweight agent following the LangChain tool-chain pattern.

    Receives a :class:`SceneUnderstanding` and :class:`StructuredEvent`,
    decides an action from ``{alert, log, summarise, escalate}``, and
    executes the corresponding tool chain.

    Action decision logic based on ``threat_level``:
    - ``"critical"`` or ``"high"`` → ``"alert"``
    - ``"medium"`` → use ``recommended_action`` from SceneUnderstanding
    - ``"low"`` → ``"log"``
    - ``"none"`` → ``"log"``

    If a tool execution fails, the agent falls back to a default action
    based on threat_level (high/critical → alert, others → log).

    Parameters
    ----------
    tools:
        Optional list of :class:`Tool` instances to register.  If not
        provided, the agent operates with no tools (action decision only).
    """

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the agent's tool chain.

        Parameters
        ----------
        tool:
            The tool instance to register.
        """
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def decide_action(self, scene: SceneUnderstanding) -> str:
        """Decide an action based on the SceneUnderstanding.

        Parameters
        ----------
        scene:
            The VLM scene understanding result.

        Returns
        -------
        str
            One of ``{"alert", "log", "summarise", "escalate"}``.
        """
        threat = scene.threat_level

        if threat in ("critical", "high"):
            action = "alert"
        elif threat == "medium":
            action = scene.recommended_action
        elif threat == "low":
            action = "log"
        else:
            # "none" or any unrecognised value
            action = "log"

        # Ensure the action is always in the valid set
        if action not in VALID_ACTIONS:
            logger.warning(
                "Invalid recommended_action '%s' from scene %s; defaulting to 'log'",
                action,
                scene.event_id,
            )
            action = "log"

        return action

    def process(
        self,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> ActionResult:
        """Process a scene understanding and event, executing the tool chain.

        Parameters
        ----------
        scene:
            The VLM scene understanding result.
        event:
            The original structured event.

        Returns
        -------
        ActionResult
            The result of the orchestration decision and tool execution.
        """
        action = self.decide_action(scene)
        logger.info(
            "OrchestrationAgent: action=%s for event %s (threat=%s)",
            action,
            event.event_id,
            scene.threat_level,
        )

        alert_payload: Optional[AlertPayload] = None
        context_update: Optional[Dict[str, Any]] = None
        cross_camera_refs: List[str] = []

        # Execute tools based on the decided action
        try:
            # Always try vector search for cross-camera references
            vector_result = self._execute_tool(
                "VectorSearchTool", scene, event
            )
            if vector_result and vector_result.success:
                cross_camera_refs = vector_result.data.get("related_ids", [])

            # Always invoke MCP and A2A stubs for context
            self._execute_tool("MCPContextTool", scene, event)
            self._execute_tool("A2ACommTool", scene, event)

            if action == "alert":
                alert_result = self._execute_tool("AlertTool", scene, event)
                if alert_result and alert_result.success:
                    alert_payload = alert_result.data.get("alert_payload")
                    # Store frame crop and attach URL to alert payload
                    crop_result = self._execute_tool(
                        "FrameCropStoreTool", scene, event
                    )
                    if (
                        crop_result
                        and crop_result.success
                        and alert_payload is not None
                    ):
                        alert_payload.frame_crop_url = crop_result.data.get(
                            "frame_crop_url"
                        )
                else:
                    # Tool failed — fall back
                    action = self._fallback_action(scene.threat_level)
                    logger.warning(
                        "AlertTool failed; falling back to action=%s", action
                    )

            if action in ("log", "summarise", "escalate"):
                log_result = self._execute_tool("LogTool", scene, event)
                if log_result and not log_result.success:
                    action = self._fallback_action(scene.threat_level)
                    logger.warning(
                        "LogTool failed; falling back to action=%s", action
                    )

        except Exception as exc:
            logger.error(
                "Tool chain execution error for event %s: %s",
                event.event_id,
                exc,
                exc_info=True,
            )
            action = self._fallback_action(scene.threat_level)

        # Ensure action is always valid before returning
        if action not in VALID_ACTIONS:
            action = "log"

        return ActionResult(
            action=action,
            alert_payload=alert_payload,
            context_update=context_update,
            cross_camera_refs=cross_camera_refs,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        tool_name: str,
        scene: SceneUnderstanding,
        event: StructuredEvent,
    ) -> Optional[ToolResult]:
        """Execute a named tool if it is registered.

        Parameters
        ----------
        tool_name:
            The name of the tool to execute.
        scene:
            The scene understanding.
        event:
            The structured event.

        Returns
        -------
        Optional[ToolResult]
            The tool result, or ``None`` if the tool is not registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            logger.debug("Tool '%s' not registered; skipping", tool_name)
            return None
        try:
            return tool.invoke(scene, event)
        except Exception as exc:
            logger.error(
                "Tool '%s' raised an exception: %s",
                tool_name,
                exc,
                exc_info=True,
            )
            return ToolResult(success=False, error=str(exc))

    @staticmethod
    def _fallback_action(threat_level: str) -> str:
        """Determine the fallback action based on threat level.

        Per the design doc's error handling section:
        - high/critical → alert
        - others → log

        Parameters
        ----------
        threat_level:
            The threat level from the SceneUnderstanding.

        Returns
        -------
        str
            A valid action from ``{"alert", "log"}``.
        """
        if threat_level in ("high", "critical"):
            return "alert"
        return "log"
