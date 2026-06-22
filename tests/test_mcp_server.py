"""Unit tests for the MCP Context Server and functional MCPContextTool.

Tests:
- MCPContextServer read/write/list/clear operations
- Thread-safety of the shared context store
- MCPContextTool functional behaviour (write + cross-camera read)
- MCPContextTool graceful degradation on server errors
- Backward compatibility: MCPContextTool() with no args

Requirements: 6.2, 16.1
"""

from __future__ import annotations

import threading
from datetime import datetime

import pytest

from agentic_cctv.mcp_server import ContextEntry, MCPContextServer
from agentic_cctv.models import (
    BoundingBox,
    IdentifiedObject,
    SceneUnderstanding,
    StructuredEvent,
)
from agentic_cctv.orchestration_agent import (
    MCPContextTool,
    OrchestrationAgent,
    ToolResult,
    VALID_ACTIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    camera_id: str = "cam-01",
    event_id: str = "evt-001",
) -> StructuredEvent:
    return StructuredEvent(
        event_id=event_id,
        camera_id=camera_id,
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
# Tests: MCPContextServer
# ---------------------------------------------------------------------------


class TestMCPContextServer:
    """Test the in-process MCP context server."""

    def test_write_and_read_context(self) -> None:
        server = MCPContextServer()
        server.write_context("cam-01", "evt-001", {"threat": "high"})
        result = server.read_context("cam-01", "evt-001")
        assert result == {"threat": "high"}

    def test_read_nonexistent_returns_none(self) -> None:
        server = MCPContextServer()
        assert server.read_context("cam-01", "evt-999") is None

    def test_write_overwrites_existing(self) -> None:
        server = MCPContextServer()
        server.write_context("cam-01", "evt-001", {"v": 1})
        server.write_context("cam-01", "evt-001", {"v": 2})
        assert server.read_context("cam-01", "evt-001") == {"v": 2}

    def test_list_contexts_for_camera(self) -> None:
        server = MCPContextServer()
        server.write_context("cam-01", "evt-001", {"a": 1})
        server.write_context("cam-01", "evt-002", {"a": 2})
        server.write_context("cam-02", "evt-003", {"b": 1})

        entries = server.list_contexts("cam-01")
        assert len(entries) == 2
        assert all(e.camera_id == "cam-01" for e in entries)

    def test_list_contexts_empty_camera(self) -> None:
        server = MCPContextServer()
        assert server.list_contexts("cam-99") == []

    def test_list_cross_camera_contexts(self) -> None:
        server = MCPContextServer()
        server.write_context("cam-01", "evt-001", {"a": 1})
        server.write_context("cam-02", "evt-002", {"b": 1})
        server.write_context("cam-03", "evt-003", {"c": 1})

        cross = server.list_cross_camera_contexts("cam-01")
        assert len(cross) == 2
        camera_ids = {e.camera_id for e in cross}
        assert "cam-01" not in camera_ids
        assert "cam-02" in camera_ids
        assert "cam-03" in camera_ids

    def test_list_cross_camera_contexts_empty_when_only_one_camera(self) -> None:
        server = MCPContextServer()
        server.write_context("cam-01", "evt-001", {"a": 1})
        cross = server.list_cross_camera_contexts("cam-01")
        assert cross == []

    def test_clear_context(self) -> None:
        server = MCPContextServer()
        server.write_context("cam-01", "evt-001", {"a": 1})
        server.write_context("cam-01", "evt-002", {"a": 2})
        server.write_context("cam-02", "evt-003", {"b": 1})

        removed = server.clear_context("cam-01")
        assert removed == 2
        assert server.read_context("cam-01", "evt-001") is None
        assert server.read_context("cam-01", "evt-002") is None
        # cam-02 unaffected
        assert server.read_context("cam-02", "evt-003") == {"b": 1}

    def test_clear_nonexistent_camera(self) -> None:
        server = MCPContextServer()
        assert server.clear_context("cam-99") == 0

    def test_size(self) -> None:
        server = MCPContextServer()
        assert server.size() == 0
        server.write_context("cam-01", "evt-001", {"a": 1})
        assert server.size() == 1
        server.write_context("cam-02", "evt-002", {"b": 1})
        assert server.size() == 2
        server.clear_context("cam-01")
        assert server.size() == 1

    def test_thread_safety_concurrent_writes(self) -> None:
        """Multiple threads writing concurrently should not corrupt state."""
        server = MCPContextServer()
        errors: list[Exception] = []

        def writer(cam_id: str, count: int) -> None:
            try:
                for i in range(count):
                    server.write_context(cam_id, f"evt-{i}", {"i": i})
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(f"cam-{t}", 50))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert server.size() == 200  # 4 cameras × 50 events


# ---------------------------------------------------------------------------
# Tests: MCPContextTool (functional)
# ---------------------------------------------------------------------------


class TestMCPContextTool:
    """Test the functional MCPContextTool."""

    def test_name(self) -> None:
        assert MCPContextTool().name == "MCPContextTool"

    def test_backward_compatible_no_args(self) -> None:
        """MCPContextTool() with no args should still work."""
        tool = MCPContextTool()
        scene = _make_scene()
        event = _make_event()
        result = tool.invoke(scene, event)
        assert result.success is True

    def test_writes_context_to_server(self) -> None:
        server = MCPContextServer()
        tool = MCPContextTool(server=server)
        scene = _make_scene()
        event = _make_event(camera_id="cam-01", event_id="evt-001")

        tool.invoke(scene, event)

        stored = server.read_context("cam-01", "evt-001")
        assert stored is not None
        assert stored["scene_description"] == "Test scene"
        assert stored["threat_level"] == "medium"
        assert stored["camera_id"] == "cam-01"

    def test_returns_cross_camera_context(self) -> None:
        server = MCPContextServer()
        # Pre-populate context from another camera
        server.write_context("cam-02", "evt-002", {"threat_level": "high"})

        tool = MCPContextTool(server=server)
        scene = _make_scene()
        event = _make_event(camera_id="cam-01", event_id="evt-001")

        result = tool.invoke(scene, event)

        assert result.success is True
        assert result.data["cross_camera_count"] == 1
        assert len(result.data["cross_camera_context"]) == 1
        assert result.data["cross_camera_context"][0]["camera_id"] == "cam-02"

    def test_excludes_own_camera_from_cross_context(self) -> None:
        server = MCPContextServer()
        tool = MCPContextTool(server=server)

        # Write from cam-01
        scene1 = _make_scene(event_id="evt-001")
        event1 = _make_event(camera_id="cam-01", event_id="evt-001")
        tool.invoke(scene1, event1)

        # Write from cam-01 again
        scene2 = _make_scene(event_id="evt-002")
        event2 = _make_event(camera_id="cam-01", event_id="evt-002")
        result = tool.invoke(scene2, event2)

        # Cross-camera context should be empty (only cam-01 entries)
        assert result.data["cross_camera_count"] == 0

    def test_graceful_degradation_on_error(self) -> None:
        """If the server raises, the tool should return empty success."""

        class BrokenServer:
            def write_context(self, *a: object, **kw: object) -> None:
                raise RuntimeError("Server down")

            def list_cross_camera_contexts(self, *a: object, **kw: object) -> list:
                raise RuntimeError("Server down")

        tool = MCPContextTool(server=BrokenServer())
        scene = _make_scene()
        event = _make_event()
        result = tool.invoke(scene, event)

        assert result.success is True
        assert result.data == {}

    def test_integrated_in_agent(self) -> None:
        """MCPContextTool should work when registered in the OrchestrationAgent."""
        server = MCPContextServer()
        tool = MCPContextTool(server=server)
        agent = OrchestrationAgent(tools=[tool])

        scene = _make_scene(threat_level="low")
        event = _make_event()
        result = agent.process(scene, event)

        assert result.action in VALID_ACTIONS
        # Context should have been written
        assert server.size() == 1

    def test_multi_camera_cross_context(self) -> None:
        """Context from multiple cameras should be available."""
        server = MCPContextServer()
        tool = MCPContextTool(server=server)

        # Simulate events from 3 cameras
        for cam_idx in range(1, 4):
            cam_id = f"cam-{cam_idx:02d}"
            evt_id = f"evt-{cam_idx:03d}"
            scene = _make_scene(event_id=evt_id)
            event = _make_event(camera_id=cam_id, event_id=evt_id)
            tool.invoke(scene, event)

        # Now invoke from cam-04 — should see 3 cross-camera entries
        scene4 = _make_scene(event_id="evt-004")
        event4 = _make_event(camera_id="cam-04", event_id="evt-004")
        result = tool.invoke(scene4, event4)

        assert result.data["cross_camera_count"] == 3
