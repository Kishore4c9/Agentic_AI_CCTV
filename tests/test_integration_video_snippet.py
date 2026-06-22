"""Integration tests for the full video snippet pipeline.

Tests end-to-end flow: VideoFeeder ring buffer → SnippetAssembler → EventEncoder
→ VLMReasoner dispatch, backward compatibility, fallback chain, and thread safety.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agentic_cctv.config_manager import ConfigManager
from agentic_cctv.event_encoder import EventEncoder
from agentic_cctv.models import (
    BoundingBox,
    CameraConfig,
    Frame,
    StructuredEvent,
    Track,
)
from agentic_cctv.snippet_assembler import SnippetAssembler, SnippetAssemblyError
from agentic_cctv.video_feeder import FrameRingBuffer, VideoFeeder
from agentic_cctv.vlm_reasoner import VLMReasoner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_base_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_camera_config(
    vlm_input_mode: str = "image",
    vlm_video_duration_seconds: int = 10,
    frame_skip: int = 3,
) -> CameraConfig:
    return CameraConfig(
        camera_id="cam-test-01",
        uri="rtsp://192.168.1.100:554/stream1",
        tenant_id="tenant-acme",
        site_id="site-hq",
        confidence_threshold=0.7,
        monitored_classes=["person", "vehicle"],
        vlm_input_mode=vlm_input_mode,
        vlm_video_duration_seconds=vlm_video_duration_seconds,
        frame_skip=frame_skip,
    )


def _make_frame(index: int, width: int = 64, height: int = 48) -> Frame:
    return Frame(
        camera_id="cam-test-01",
        timestamp=_base_time + timedelta(seconds=index),
        image=np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        frame_number=index,
    )


def _make_track() -> Track:
    return Track(
        track_id=str(uuid.uuid4()),
        object_type="person",
        bounding_box=BoundingBox(x=5, y=5, width=30, height=30),
        confidence=0.85,
        age=1,
        is_new=True,
    )


def _valid_vlm_response() -> dict:
    return {
        "scene_description": "A person detected in the scene.",
        "threat_level": "low",
        "objects_identified": [
            {"type": "person", "action": "walking", "location": "entrance"},
        ],
        "recommended_action": "log",
        "confidence": 0.85,
    }


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


class TestEndToEndVideoSnippetPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_video_mode(self) -> None:
        """VideoFeeder pushes frames → SnippetAssembler encodes → EventEncoder
        populates StructuredEvent → VLMReasoner dispatches video to mock backend."""
        config = _make_camera_config(vlm_input_mode="video", frame_skip=1)

        # 1. Simulate ring buffer with frames
        ring_buffer = FrameRingBuffer(20)
        for i in range(10):
            ring_buffer.push(_make_frame(i))

        # 2. Create assembler
        assembler = SnippetAssembler(fps=10.0)

        # 3. Create encoder with video support
        encoder = EventEncoder(
            config,
            snippet_assembler=assembler,
            ring_buffer=ring_buffer,
        )

        # 4. Encode a track
        track = _make_track()
        frame = _make_frame(5)
        event = encoder.encode(track, frame)

        assert event.media_type == "video"
        assert event.video_snippet is not None
        assert len(event.video_snippet) > 0
        assert len(event.frame_crop) > 0

        # 5. VLMReasoner dispatches video
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response()

        camera_configs = {config.camera_id: config}
        reasoner = VLMReasoner(backend=backend, camera_configs=camera_configs)

        result = await reasoner.reason(event)

        assert result.scene_description == "A person detected in the scene."
        call_args = backend.analyze.call_args
        assert call_args[0][0] == event.video_snippet
        assert call_args[1]["media_type"] == "video"


# ---------------------------------------------------------------------------
# Backward compatibility test
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_existing_config_validates_cleanly(self) -> None:
        """Loading existing config.example.yaml produces zero validation errors
        for the new fields (only the expected API key placeholder error)."""
        cm = ConfigManager(config_path="config.example.yaml")
        config = cm.load()
        errors = cm.validate()

        # Filter out the expected API key placeholder error
        non_api_key_errors = [
            e for e in errors if "api_key" not in e.field_path
        ]

        # No errors related to vlm_input_mode or vlm_video_duration_seconds
        vlm_field_errors = [
            e for e in non_api_key_errors
            if "vlm_input_mode" in e.field_path
            or "vlm_video_duration_seconds" in e.field_path
        ]
        assert vlm_field_errors == []

    def test_image_mode_cameras_have_no_ring_buffer(self) -> None:
        """Image-mode cameras do not allocate ring buffers."""
        config = _make_camera_config(vlm_input_mode="image")
        vf = VideoFeeder(config)
        assert vf.ring_buffer is None

    def test_video_mode_cameras_have_ring_buffer(self) -> None:
        """Video-mode cameras allocate ring buffers."""
        config = _make_camera_config(vlm_input_mode="video")
        vf = VideoFeeder(config)
        assert vf.ring_buffer is not None


# ---------------------------------------------------------------------------
# Fallback chain test
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_snippet_assembly_error_degrades_to_image(self) -> None:
        """When SnippetAssemblyError occurs, EventEncoder falls back to image mode."""
        config = _make_camera_config(vlm_input_mode="video")

        # Empty ring buffer will cause SnippetAssemblyError
        ring_buffer = FrameRingBuffer(10)
        assembler = SnippetAssembler(fps=10.0)

        encoder = EventEncoder(
            config,
            snippet_assembler=assembler,
            ring_buffer=ring_buffer,
        )

        track = _make_track()
        frame = _make_frame(0)
        event = encoder.encode(track, frame)

        assert event.media_type == "image"
        assert event.video_snippet is None
        assert len(event.frame_crop) > 0

    @pytest.mark.asyncio
    async def test_vlm_reasoner_falls_back_on_missing_snippet(self) -> None:
        """VLMReasoner falls back to frame_crop when video_snippet is None."""
        config = _make_camera_config(vlm_input_mode="video")
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response()

        camera_configs = {config.camera_id: config}
        reasoner = VLMReasoner(backend=backend, camera_configs=camera_configs)

        event = StructuredEvent(
            event_id="test-evt",
            camera_id=config.camera_id,
            tenant_id=config.tenant_id,
            site_id=config.site_id,
            timestamp=_base_time,
            object_type="person",
            track_id="trk-001",
            bounding_box=BoundingBox(x=10, y=10, width=50, height=50),
            confidence=0.85,
            frame_crop="dGVzdA==",
            video_snippet=None,
            media_type="image",
        )

        result = await reasoner.reason(event)
        call_args = backend.analyze.call_args
        assert call_args[0][0] == "dGVzdA=="
        assert call_args[1]["media_type"] == "image"


# ---------------------------------------------------------------------------
# Thread safety test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_push_and_read(self) -> None:
        """Concurrent push/read on FrameRingBuffer under load."""
        buf = FrameRingBuffer(100)
        errors = []
        n_writes = 500
        n_reads = 500

        def writer():
            try:
                for i in range(n_writes):
                    buf.push(_make_frame(i))
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(n_reads):
                    buf.get_all_frames()
                    buf.get_frames_in_range(
                        _base_time,
                        _base_time + timedelta(seconds=1000),
                    )
                    _ = buf.size
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
