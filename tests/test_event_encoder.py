"""Unit tests for the EventEncoder module."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest

from agentic_cctv.event_encoder import (
    EventEncoder,
    MQTTPublisherProtocol,
    _clip_bounding_box,
    _crop_and_encode,
    _structured_event_to_dict,
)
from agentic_cctv.models import (
    BoundingBox,
    CameraConfig,
    Frame,
    StructuredEvent,
    Track,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_camera_config(
    camera_id: str = "cam-lobby-01",
    tenant_id: str = "tenant-acme",
    site_id: str = "site-hq",
) -> CameraConfig:
    return CameraConfig(
        camera_id=camera_id,
        uri="rtsp://192.168.1.100:554/stream1",
        tenant_id=tenant_id,
        site_id=site_id,
        confidence_threshold=0.7,
        monitored_classes=["person", "vehicle"],
    )


def _make_frame(
    width: int = 640,
    height: int = 480,
    camera_id: str = "cam-lobby-01",
    frame_number: int = 0,
) -> Frame:
    """Create a Frame with a coloured numpy image."""
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Frame(
        camera_id=camera_id,
        timestamp=datetime(2025, 1, 15, 14, 30, 0, 123000, tzinfo=timezone.utc),
        image=image,
        frame_number=frame_number,
    )


def _make_track(
    x: int = 120,
    y: int = 80,
    w: int = 200,
    h: int = 400,
    confidence: float = 0.92,
    object_type: str = "person",
    track_id: str | None = None,
) -> Track:
    return Track(
        track_id=track_id or str(uuid.uuid4()),
        object_type=object_type,
        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
        confidence=confidence,
        age=5,
        is_new=False,
    )


class FakeMQTTPublisher:
    """A simple fake publisher that records calls."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []

    async def publish(
        self, topic: str, payload: bytes, qos: int = 1, retain: bool = False
    ) -> None:
        self.published.append((topic, payload, qos, retain))


# ---------------------------------------------------------------------------
# _clip_bounding_box tests
# ---------------------------------------------------------------------------

class TestClipBoundingBox:
    def test_box_within_frame(self) -> None:
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        x1, y1, x2, y2 = _clip_bounding_box(bbox, frame_h=480, frame_w=640)
        assert (x1, y1, x2, y2) == (10, 20, 110, 70)

    def test_box_extends_beyond_right_bottom(self) -> None:
        bbox = BoundingBox(x=600, y=450, width=100, height=100)
        x1, y1, x2, y2 = _clip_bounding_box(bbox, frame_h=480, frame_w=640)
        assert x2 == 640
        assert y2 == 480

    def test_box_extends_beyond_left_top(self) -> None:
        bbox = BoundingBox(x=-10, y=-20, width=100, height=100)
        x1, y1, x2, y2 = _clip_bounding_box(bbox, frame_h=480, frame_w=640)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 90
        assert y2 == 80

    def test_box_completely_outside_frame(self) -> None:
        bbox = BoundingBox(x=700, y=500, width=50, height=50)
        x1, y1, x2, y2 = _clip_bounding_box(bbox, frame_h=480, frame_w=640)
        # x2 = min(640, 750) = 640, x1 = max(0, 700) = 700 → x2 <= x1
        assert x2 <= x1 or y2 <= y1


# ---------------------------------------------------------------------------
# _crop_and_encode tests
# ---------------------------------------------------------------------------

class TestCropAndEncode:
    def test_valid_crop_returns_base64_jpeg(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[80:200, 120:320] = 128  # grey region
        bbox = BoundingBox(x=120, y=80, width=200, height=120)

        result = _crop_and_encode(image, bbox)

        # Should be valid base64
        decoded = base64.b64decode(result)
        # JPEG magic bytes
        assert decoded[:2] == b"\xff\xd8"

    def test_crop_dimensions_match_bbox(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x=100, y=50, width=200, height=150)

        result = _crop_and_encode(image, bbox)
        decoded = base64.b64decode(result)

        # Decode the JPEG back and check dimensions
        arr = np.frombuffer(decoded, dtype=np.uint8)
        crop_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert crop_img.shape[0] == 150  # height
        assert crop_img.shape[1] == 200  # width

    def test_out_of_bounds_bbox_clips_to_frame(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=80, y=80, width=50, height=50)

        result = _crop_and_encode(image, bbox)
        decoded = base64.b64decode(result)

        arr = np.frombuffer(decoded, dtype=np.uint8)
        crop_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # Clipped: x 80..100 (20px), y 80..100 (20px)
        assert crop_img.shape[0] == 20
        assert crop_img.shape[1] == 20

    def test_completely_outside_bbox_returns_fallback(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=200, y=200, width=50, height=50)

        result = _crop_and_encode(image, bbox)
        # Should still return valid base64 JPEG (1x1 fallback)
        decoded = base64.b64decode(result)
        assert decoded[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# _structured_event_to_dict tests
# ---------------------------------------------------------------------------

class TestStructuredEventToDict:
    def test_all_fields_present(self) -> None:
        event = StructuredEvent(
            event_id="test-uuid",
            camera_id="cam-01",
            tenant_id="tenant-a",
            site_id="site-1",
            timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            object_type="person",
            track_id="trk-123",
            bounding_box=BoundingBox(x=10, y=20, width=30, height=40),
            confidence=0.95,
            frame_crop="base64data",
        )
        d = _structured_event_to_dict(event)

        assert d["event_id"] == "test-uuid"
        assert d["camera_id"] == "cam-01"
        assert d["tenant_id"] == "tenant-a"
        assert d["site_id"] == "site-1"
        assert d["timestamp"] == "2025-01-15T14:30:00+00:00"
        assert d["object_type"] == "person"
        assert d["track_id"] == "trk-123"
        assert d["bounding_box"] == {"x": 10, "y": 20, "width": 30, "height": 40}
        assert d["confidence"] == 0.95
        assert d["frame_crop"] == "base64data"

    def test_serialisable_to_json(self) -> None:
        event = StructuredEvent(
            event_id="id",
            camera_id="cam",
            tenant_id="t",
            site_id="s",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            object_type="vehicle",
            track_id="trk",
            bounding_box=BoundingBox(x=0, y=0, width=1, height=1),
            confidence=0.5,
            frame_crop="abc",
        )
        # Should not raise
        result = json.dumps(_structured_event_to_dict(event))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# EventEncoder.encode tests
# ---------------------------------------------------------------------------

class TestEventEncoderEncode:
    def test_encode_returns_structured_event(self) -> None:
        config = _make_camera_config()
        encoder = EventEncoder(config)
        track = _make_track()
        frame = _make_frame()

        event = encoder.encode(track, frame)

        assert isinstance(event, StructuredEvent)

    def test_event_id_is_valid_uuid(self) -> None:
        encoder = EventEncoder(_make_camera_config())
        event = encoder.encode(_make_track(), _make_frame())
        # Should not raise
        uuid.UUID(event.event_id)

    def test_camera_fields_from_config(self) -> None:
        config = _make_camera_config(
            camera_id="cam-x", tenant_id="tenant-y", site_id="site-z"
        )
        encoder = EventEncoder(config)
        event = encoder.encode(_make_track(), _make_frame())

        assert event.camera_id == "cam-x"
        assert event.tenant_id == "tenant-y"
        assert event.site_id == "site-z"

    def test_track_fields_propagated(self) -> None:
        track = _make_track(
            object_type="vehicle",
            confidence=0.88,
            track_id="trk-abc",
            x=50, y=60, w=100, h=200,
        )
        encoder = EventEncoder(_make_camera_config())
        event = encoder.encode(track, _make_frame())

        assert event.object_type == "vehicle"
        assert event.confidence == 0.88
        assert event.track_id == "trk-abc"
        assert event.bounding_box == track.bounding_box

    def test_timestamp_from_frame(self) -> None:
        frame = _make_frame()
        encoder = EventEncoder(_make_camera_config())
        event = encoder.encode(_make_track(), frame)

        assert event.timestamp == frame.timestamp

    def test_frame_crop_is_nonempty_base64_jpeg(self) -> None:
        encoder = EventEncoder(_make_camera_config())
        event = encoder.encode(_make_track(), _make_frame())

        assert len(event.frame_crop) > 0
        decoded = base64.b64decode(event.frame_crop)
        assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_encode_with_bbox_exceeding_frame(self) -> None:
        """Bounding box larger than frame should be clipped, not error."""
        encoder = EventEncoder(_make_camera_config())
        track = _make_track(x=600, y=440, w=200, h=200)
        frame = _make_frame(width=640, height=480)

        event = encoder.encode(track, frame)
        assert len(event.frame_crop) > 0

    def test_each_encode_generates_unique_event_id(self) -> None:
        encoder = EventEncoder(_make_camera_config())
        track = _make_track()
        frame = _make_frame()

        e1 = encoder.encode(track, frame)
        e2 = encoder.encode(track, frame)
        assert e1.event_id != e2.event_id


# ---------------------------------------------------------------------------
# EventEncoder.encode_and_publish tests
# ---------------------------------------------------------------------------

class TestEventEncoderEncodeAndPublish:
    @pytest.mark.asyncio
    async def test_returns_structured_event(self) -> None:
        encoder = EventEncoder(_make_camera_config())
        event = await encoder.encode_and_publish(_make_track(), _make_frame())
        assert isinstance(event, StructuredEvent)

    @pytest.mark.asyncio
    async def test_no_publisher_does_not_raise(self) -> None:
        encoder = EventEncoder(_make_camera_config(), mqtt_publisher=None)
        event = await encoder.encode_and_publish(_make_track(), _make_frame())
        assert event.event_id  # event still created

    @pytest.mark.asyncio
    async def test_publishes_to_correct_topic(self) -> None:
        config = _make_camera_config(
            camera_id="cam-01", tenant_id="tenant-a", site_id="site-1"
        )
        publisher = FakeMQTTPublisher()
        encoder = EventEncoder(config, mqtt_publisher=publisher)

        await encoder.encode_and_publish(_make_track(), _make_frame())

        assert len(publisher.published) == 1
        topic, payload, qos, retain = publisher.published[0]
        assert topic == "tenant-a/site-1/cam-01/events"
        assert qos == 1
        assert retain is False

    @pytest.mark.asyncio
    async def test_published_payload_is_valid_json(self) -> None:
        publisher = FakeMQTTPublisher()
        encoder = EventEncoder(_make_camera_config(), mqtt_publisher=publisher)

        await encoder.encode_and_publish(_make_track(), _make_frame())

        _, payload_bytes, _, _ = publisher.published[0]
        data = json.loads(payload_bytes.decode("utf-8"))

        # All required fields present
        required_fields = {
            "event_id", "camera_id", "tenant_id", "site_id",
            "timestamp", "object_type", "track_id", "bounding_box",
            "confidence", "frame_crop",
        }
        assert required_fields.issubset(data.keys())

    @pytest.mark.asyncio
    async def test_published_event_matches_returned_event(self) -> None:
        publisher = FakeMQTTPublisher()
        encoder = EventEncoder(_make_camera_config(), mqtt_publisher=publisher)

        event = await encoder.encode_and_publish(_make_track(), _make_frame())

        _, payload_bytes, _, _ = publisher.published[0]
        data = json.loads(payload_bytes.decode("utf-8"))

        assert data["event_id"] == event.event_id
        assert data["camera_id"] == event.camera_id
        assert data["confidence"] == event.confidence

    @pytest.mark.asyncio
    async def test_publisher_protocol_with_async_mock(self) -> None:
        """Verify EventEncoder works with any object matching MQTTPublisherProtocol."""
        mock_pub = AsyncMock()
        encoder = EventEncoder(_make_camera_config(), mqtt_publisher=mock_pub)

        await encoder.encode_and_publish(_make_track(), _make_frame())

        mock_pub.publish.assert_called_once()
        call_args = mock_pub.publish.call_args
        assert call_args[0][0].endswith("/events")  # topic


# ---------------------------------------------------------------------------
# EventEncoder video mode tests
# ---------------------------------------------------------------------------

from datetime import timedelta
from unittest.mock import MagicMock, patch as mock_patch

from agentic_cctv.snippet_assembler import SnippetAssembler, SnippetAssemblyError
from agentic_cctv.video_feeder import FrameRingBuffer


def _make_ring_buffer_with_frames(n: int = 5) -> FrameRingBuffer:
    """Create a ring buffer pre-loaded with synthetic frames."""
    base_time = datetime(2025, 1, 15, 14, 30, 0, 123000, tzinfo=timezone.utc)
    buf = FrameRingBuffer(n + 10)
    for i in range(n):
        buf.push(Frame(
            camera_id="cam-lobby-01",
            timestamp=base_time + timedelta(seconds=i),
            image=np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8),
            frame_number=i,
        ))
    return buf


class TestEventEncoderVideoMode:
    def test_video_mode_populates_video_snippet_and_media_type(self) -> None:
        """Video mode with successful assembly populates video_snippet and media_type='video'."""
        config = _make_camera_config()
        config = CameraConfig(
            camera_id=config.camera_id,
            uri=config.uri,
            tenant_id=config.tenant_id,
            site_id=config.site_id,
            confidence_threshold=config.confidence_threshold,
            monitored_classes=config.monitored_classes,
            vlm_input_mode="video",
            vlm_video_duration_seconds=10,
        )
        ring_buffer = _make_ring_buffer_with_frames(5)
        assembler = SnippetAssembler(fps=10.0)
        encoder = EventEncoder(config, snippet_assembler=assembler, ring_buffer=ring_buffer)

        track = _make_track()
        frame = _make_frame()
        event = encoder.encode(track, frame)

        assert event.media_type == "video"
        assert event.video_snippet is not None
        assert len(event.video_snippet) > 0

    def test_video_mode_assembly_failure_falls_back_to_image(self) -> None:
        """Video mode with assembly failure falls back to media_type='image'."""
        config = CameraConfig(
            camera_id="cam-lobby-01",
            uri="rtsp://test:554/stream",
            tenant_id="tenant-acme",
            site_id="site-hq",
            confidence_threshold=0.7,
            vlm_input_mode="video",
            vlm_video_duration_seconds=10,
        )
        # Empty ring buffer will cause SnippetAssemblyError
        ring_buffer = FrameRingBuffer(10)
        assembler = SnippetAssembler(fps=10.0)
        encoder = EventEncoder(config, snippet_assembler=assembler, ring_buffer=ring_buffer)

        track = _make_track()
        frame = _make_frame()
        event = encoder.encode(track, frame)

        assert event.media_type == "image"
        assert event.video_snippet is None

    def test_image_mode_leaves_video_snippet_none(self) -> None:
        """Image mode leaves video_snippet=None and media_type='image'."""
        config = _make_camera_config()
        encoder = EventEncoder(config)

        track = _make_track()
        frame = _make_frame()
        event = encoder.encode(track, frame)

        assert event.media_type == "image"
        assert event.video_snippet is None

    def test_frame_crop_always_nonempty_in_both_modes(self) -> None:
        """frame_crop is always non-empty in both image and video modes."""
        # Image mode
        config_img = _make_camera_config()
        encoder_img = EventEncoder(config_img)
        event_img = encoder_img.encode(_make_track(), _make_frame())
        assert len(event_img.frame_crop) > 0

        # Video mode
        config_vid = CameraConfig(
            camera_id="cam-lobby-01",
            uri="rtsp://test:554/stream",
            tenant_id="tenant-acme",
            site_id="site-hq",
            confidence_threshold=0.7,
            vlm_input_mode="video",
            vlm_video_duration_seconds=10,
        )
        ring_buffer = _make_ring_buffer_with_frames(5)
        assembler = SnippetAssembler(fps=10.0)
        encoder_vid = EventEncoder(config_vid, snippet_assembler=assembler, ring_buffer=ring_buffer)
        event_vid = encoder_vid.encode(_make_track(), _make_frame())
        assert len(event_vid.frame_crop) > 0
