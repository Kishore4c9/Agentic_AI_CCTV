"""Property-based tests for EventEncoder video snippet support.

Feature: vlm-video-snippet, Property 8: EventEncoder Populates Fields Correctly Based on Input Mode
Feature: vlm-video-snippet, Property 9: Frame Crop Always Present Invariant
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
from hypothesis import given, settings, strategies as st

from agentic_cctv.event_encoder import EventEncoder
from agentic_cctv.models import BoundingBox, CameraConfig, Frame, Track
from agentic_cctv.snippet_assembler import SnippetAssembler
from agentic_cctv.video_feeder import FrameRingBuffer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_base_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_camera_config(vlm_input_mode: str = "image") -> CameraConfig:
    return CameraConfig(
        camera_id="cam-test",
        uri="rtsp://test:554/stream",
        tenant_id="tenant-test",
        site_id="site-test",
        confidence_threshold=0.7,
        monitored_classes=["person"],
        vlm_input_mode=vlm_input_mode,
        vlm_video_duration_seconds=10,
    )


def _make_track(confidence: float = 0.85) -> Track:
    return Track(
        track_id=str(uuid.uuid4()),
        object_type="person",
        bounding_box=BoundingBox(x=10, y=10, width=50, height=50),
        confidence=confidence,
        age=1,
        is_new=True,
    )


def _make_frame(width: int = 64, height: int = 48) -> Frame:
    return Frame(
        camera_id="cam-test",
        timestamp=_base_time,
        image=np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        frame_number=1,
    )


def _make_ring_buffer_with_frames(n: int = 5) -> FrameRingBuffer:
    buf = FrameRingBuffer(n + 10)
    for i in range(n):
        buf.push(Frame(
            camera_id="cam-test",
            timestamp=_base_time + timedelta(seconds=i),
            image=np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8),
            frame_number=i,
        ))
    return buf


# ---------------------------------------------------------------------------
# Property 8: EventEncoder Populates Fields Correctly Based on Input Mode
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(
    vlm_mode=st.sampled_from(["image", "video"]),
    confidence=st.floats(min_value=0.1, max_value=1.0),
)
def test_event_encoder_populates_fields_by_mode(
    vlm_mode: str, confidence: float
) -> None:
    """**Validates: Requirements 6.3, 6.4**

    For any track and frame: when vlm_input_mode="video" and assembly succeeds,
    media_type == "video" and video_snippet is non-empty; when vlm_input_mode="image",
    media_type == "image" and video_snippet is None.
    """
    config = _make_camera_config(vlm_input_mode=vlm_mode)
    track = _make_track(confidence=confidence)
    frame = _make_frame()

    if vlm_mode == "video":
        ring_buffer = _make_ring_buffer_with_frames(5)
        assembler = SnippetAssembler(fps=10.0)
        encoder = EventEncoder(
            config,
            snippet_assembler=assembler,
            ring_buffer=ring_buffer,
        )
    else:
        encoder = EventEncoder(config)

    event = encoder.encode(track, frame)

    if vlm_mode == "video":
        assert event.media_type == "video"
        assert event.video_snippet is not None
        assert len(event.video_snippet) > 0
    else:
        assert event.media_type == "image"
        assert event.video_snippet is None


# ---------------------------------------------------------------------------
# Property 9: Frame Crop Always Present Invariant
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(
    vlm_mode=st.sampled_from(["image", "video"]),
    confidence=st.floats(min_value=0.1, max_value=1.0),
)
def test_frame_crop_always_present(vlm_mode: str, confidence: float) -> None:
    """**Validates: Requirements 6.5**

    For any track, frame, and vlm_input_mode value, the EventEncoder always
    produces a StructuredEvent with a non-empty frame_crop string.
    """
    config = _make_camera_config(vlm_input_mode=vlm_mode)
    track = _make_track(confidence=confidence)
    frame = _make_frame()

    if vlm_mode == "video":
        ring_buffer = _make_ring_buffer_with_frames(5)
        assembler = SnippetAssembler(fps=10.0)
        encoder = EventEncoder(
            config,
            snippet_assembler=assembler,
            ring_buffer=ring_buffer,
        )
    else:
        encoder = EventEncoder(config)

    event = encoder.encode(track, frame)

    assert event.frame_crop is not None
    assert len(event.frame_crop) > 0
