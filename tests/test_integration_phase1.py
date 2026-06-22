"""Integration tests for Phase 1 pipeline.

Validates end-to-end flows WITHOUT requiring a real MQTT broker or real cameras.
Each test exercises multiple components wired together to verify correct
data flow across component boundaries.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock

import numpy as np
import pytest

from agentic_cctv.alert_system import AlertSystem, PushNotificationChannel
from agentic_cctv.detection_engine import DetectionEngine
from agentic_cctv.event_encoder import EventEncoder, _structured_event_to_dict
from agentic_cctv.main import CameraPipeline
from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    CameraConfig,
    CooldownConfig,
    Detection,
    Frame,
    RawDetection,
    StructuredEvent,
    Track,
)
from agentic_cctv.runtimes import InferenceRuntime
from agentic_cctv.store_and_forward import StoreAndForwardQueue
from agentic_cctv.timeseries_db import TimeSeriesDB, TimeSeriesDBSubscriber
from agentic_cctv.tracker import Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera_config(**overrides: object) -> CameraConfig:
    """Create a CameraConfig with sensible defaults, overridable via kwargs."""
    defaults = dict(
        camera_id="cam-lobby-01",
        uri="rtsp://192.168.1.100:554/stream1",
        tenant_id="tenant-acme",
        site_id="site-hq",
        confidence_threshold=0.5,
        monitored_classes=["person", "vehicle"],
        inference_runtime="pytorch",
        model_path="./models/yolov8n.pt",
        tracker_algorithm="deepsort",
        frame_skip=1,
    )
    defaults.update(overrides)
    return CameraConfig(**defaults)  # type: ignore[arg-type]


def _make_frame(camera_id: str = "cam-lobby-01") -> Frame:
    """Create a synthetic Frame with a 480×640 BGR numpy image."""
    return Frame(
        camera_id=camera_id,
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_number=1,
    )


def _make_structured_event(**overrides: object) -> StructuredEvent:
    """Create a StructuredEvent with sensible defaults."""
    defaults = dict(
        event_id=str(uuid.uuid4()),
        camera_id="cam-lobby-01",
        tenant_id="tenant-acme",
        site_id="site-hq",
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        object_type="person",
        track_id="trk-abc123",
        bounding_box=BoundingBox(x=100, y=80, width=200, height=400),
        confidence=0.92,
        frame_crop="dGVzdA==",
    )
    defaults.update(overrides)
    return StructuredEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stub InferenceRuntime
# ---------------------------------------------------------------------------


class StubInferenceRuntime:
    """A stub InferenceRuntime that returns pre-configured RawDetections."""

    def __init__(self, detections: List[RawDetection]) -> None:
        self._detections = detections

    def load_model(self, model_path: str) -> None:
        pass  # no-op

    def infer(self, image: np.ndarray) -> List[RawDetection]:
        return self._detections


# ---------------------------------------------------------------------------
# 1. Event encoding → TimeSeriesDB persistence test
# ---------------------------------------------------------------------------


class TestEventEncodingToTimeSeriesDB:
    """Validates: Requirements 3.1, 3.9, 14.3

    Create a StructuredEvent, serialize to JSON, pass through
    TimeSeriesDBSubscriber callback, and verify persistence.
    """

    def test_event_persisted_via_subscriber_callback(self) -> None:
        """End-to-end: StructuredEvent → JSON → TimeSeriesDBSubscriber → SQLite."""
        # 1. Create a StructuredEvent
        event = _make_structured_event(
            event_id="evt-integration-001",
            camera_id="cam-lobby-01",
            tenant_id="tenant-acme",
            site_id="site-hq",
            object_type="person",
            track_id="trk-int-001",
            confidence=0.88,
            bounding_box=BoundingBox(x=120, y=80, width=200, height=400),
        )

        # 2. Serialize to JSON bytes (same as EventEncoder would produce)
        payload = json.dumps(_structured_event_to_dict(event)).encode("utf-8")

        # 3. Create in-memory TimeSeriesDB and subscriber
        db = TimeSeriesDB(":memory:")
        subscriber = TimeSeriesDBSubscriber(db)

        # 4. Invoke the subscriber callback (simulates MQTT message arrival)
        topic = "tenant-acme/site-hq/cam-lobby-01/events"
        subscriber(topic, payload, qos=1)

        # 5. Verify the event is persisted
        rows = db.get_events(camera_id="cam-lobby-01")
        assert len(rows) == 1

        row = rows[0]
        assert row["event_id"] == "evt-integration-001"
        assert row["camera_id"] == "cam-lobby-01"
        assert row["tenant_id"] == "tenant-acme"
        assert row["site_id"] == "site-hq"
        assert row["object_type"] == "person"
        assert row["track_id"] == "trk-int-001"
        assert row["confidence"] == pytest.approx(0.88)

        # Verify bounding box stored as JSON
        bbox = json.loads(row["bounding_box"])
        assert bbox["x"] == 120
        assert bbox["y"] == 80
        assert bbox["width"] == 200
        assert bbox["height"] == 400

        db.close()


# ---------------------------------------------------------------------------
# 2. Detection → Tracking → Encoding pipeline test
# ---------------------------------------------------------------------------


class TestDetectionTrackingEncodingPipeline:
    """Validates: Requirements 3.1, 14.2

    Synthetic Frame → stub InferenceRuntime → DetectionEngine → Tracker →
    EventEncoder → verify StructuredEvent fields.
    """

    def test_full_detection_to_event_pipeline(self) -> None:
        """Detection → Tracking → Encoding produces a valid StructuredEvent."""
        camera_config = _make_camera_config(
            confidence_threshold=0.5,
            monitored_classes=["person"],
        )

        # Stub runtime returns a known detection
        raw_detections = [
            RawDetection(
                object_type="person",
                bounding_box=BoundingBox(x=100, y=50, width=150, height=300),
                confidence=0.85,
            ),
        ]
        stub_runtime = StubInferenceRuntime(raw_detections)

        # Wire up the pipeline
        detection_engine = DetectionEngine(camera_config, stub_runtime)
        tracker = Tracker(algorithm="deepsort", max_age=30)
        event_encoder = EventEncoder(camera_config=camera_config, mqtt_publisher=None)

        # Create a synthetic frame
        frame = _make_frame(camera_id="cam-lobby-01")

        # Run detection
        detections = detection_engine.detect(frame)
        assert len(detections) == 1
        assert detections[0].passed_gate is True

        # Run tracking
        tracks = tracker.update(detections, frame)
        assert len(tracks) == 1
        assert tracks[0].is_new is True
        assert tracks[0].object_type == "person"

        # Encode event
        event = event_encoder.encode(tracks[0], frame)

        # Verify StructuredEvent fields
        assert event.camera_id == "cam-lobby-01"
        assert event.tenant_id == "tenant-acme"
        assert event.site_id == "site-hq"
        assert event.object_type == "person"
        assert event.confidence == pytest.approx(0.85)
        assert event.track_id == tracks[0].track_id
        assert event.event_id  # non-empty UUID
        assert event.frame_crop  # non-empty base64 string
        assert event.timestamp == frame.timestamp

    def test_below_threshold_detection_not_tracked(self) -> None:
        """Detections below threshold should not produce tracks or events."""
        camera_config = _make_camera_config(
            confidence_threshold=0.9,
            monitored_classes=["person"],
        )

        raw_detections = [
            RawDetection(
                object_type="person",
                bounding_box=BoundingBox(x=100, y=50, width=150, height=300),
                confidence=0.5,  # below 0.9 threshold
            ),
        ]
        stub_runtime = StubInferenceRuntime(raw_detections)

        detection_engine = DetectionEngine(camera_config, stub_runtime)
        tracker = Tracker(algorithm="deepsort", max_age=30)

        frame = _make_frame()
        detections = detection_engine.detect(frame)
        assert len(detections) == 1
        assert detections[0].passed_gate is False

        tracks = tracker.update(detections, frame)
        assert len(tracks) == 0


# ---------------------------------------------------------------------------
# 3. Store-and-forward queue integration test
# ---------------------------------------------------------------------------


class TestStoreAndForwardQueueIntegration:
    """Validates: Requirements 3.7

    Enqueue messages → drain with mock publisher → verify order and completeness.
    """

    def test_enqueue_drain_preserves_order(self) -> None:
        """All enqueued messages are drained in FIFO order."""
        queue = StoreAndForwardQueue(db_path=":memory:", max_age_seconds=300)

        # Enqueue several messages
        messages = [
            ("tenant/site/cam/events", f"payload-{i}".encode(), 1)
            for i in range(5)
        ]
        for topic, payload, qos in messages:
            queue.enqueue(topic, payload, qos)

        assert queue.size() == 5

        # Create a mock publisher that records published messages
        published: List[tuple] = []

        mock_publisher = AsyncMock()

        async def _capture_publish(
            topic: str, payload: bytes, qos: int = 1, retain: bool = False
        ) -> None:
            published.append((topic, payload, qos, retain))

        mock_publisher.publish = AsyncMock(side_effect=_capture_publish)

        # Drain the queue
        drained = queue.drain(mock_publisher)

        assert drained == 5
        assert queue.size() == 0

        # Verify order
        for i, (topic, payload, qos, retain) in enumerate(published):
            assert topic == "tenant/site/cam/events"
            assert payload == f"payload-{i}".encode()
            assert qos == 1

        queue.close()

    def test_drain_empty_queue_returns_zero(self) -> None:
        """Draining an empty queue returns 0."""
        queue = StoreAndForwardQueue(db_path=":memory:")
        mock_publisher = AsyncMock()
        assert queue.drain(mock_publisher) == 0
        queue.close()


# ---------------------------------------------------------------------------
# 4. Alert system cooldown integration test
# ---------------------------------------------------------------------------


class TestAlertSystemCooldownIntegration:
    """Validates: Requirements 14.4

    Send multiple alerts for the same (camera_id, event_type) and verify
    cooldown suppression behaviour.
    """

    @pytest.mark.asyncio
    async def test_first_alert_delivered_rest_suppressed(self) -> None:
        """First alert is delivered; subsequent alerts within cooldown are suppressed."""
        channel = PushNotificationChannel()
        cooldown = CooldownConfig(default_seconds=60)
        alert_system = AlertSystem(channels=[channel], cooldown_config=cooldown)

        base_payload = AlertPayload(
            alert_id="alert-001",
            event_id="evt-001",
            camera_id="cam-lobby-01",
            tenant_id="tenant-acme",
            site_id="site-hq",
            timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            alert_type="intrusion",
            description="Person detected in restricted area",
            threat_level="high",
            frame_crop_url=None,
        )

        # First alert should be delivered
        result1 = await alert_system.send_alert(base_payload)
        assert result1.delivered is True
        assert result1.suppressed is False
        assert "PushNotificationChannel" in result1.channels

        # Send 4 more alerts for the same (camera_id, alert_type)
        suppressed_results = []
        for i in range(4):
            payload = AlertPayload(
                alert_id=f"alert-{i + 2:03d}",
                event_id=f"evt-{i + 2:03d}",
                camera_id="cam-lobby-01",
                tenant_id="tenant-acme",
                site_id="site-hq",
                timestamp=datetime(2025, 1, 15, 14, 30, i + 1, tzinfo=timezone.utc),
                alert_type="intrusion",
                description="Person detected in restricted area",
                threat_level="high",
                frame_crop_url=None,
            )
            result = await alert_system.send_alert(payload)
            suppressed_results.append(result)

        # All subsequent alerts should be suppressed
        for i, result in enumerate(suppressed_results):
            assert result.delivered is False
            assert result.suppressed is True
            assert result.suppressed_count == i + 1

    @pytest.mark.asyncio
    async def test_different_event_types_not_suppressed(self) -> None:
        """Alerts for different event types are not suppressed by each other."""
        channel = PushNotificationChannel()
        cooldown = CooldownConfig(default_seconds=60)
        alert_system = AlertSystem(channels=[channel], cooldown_config=cooldown)

        payload_intrusion = AlertPayload(
            alert_id="alert-001",
            event_id="evt-001",
            camera_id="cam-lobby-01",
            tenant_id="tenant-acme",
            site_id="site-hq",
            timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            alert_type="intrusion",
            description="Person detected",
            threat_level="high",
            frame_crop_url=None,
        )

        payload_fire = AlertPayload(
            alert_id="alert-002",
            event_id="evt-002",
            camera_id="cam-lobby-01",
            tenant_id="tenant-acme",
            site_id="site-hq",
            timestamp=datetime(2025, 1, 15, 14, 30, 1, tzinfo=timezone.utc),
            alert_type="fire",
            description="Fire detected",
            threat_level="critical",
            frame_crop_url=None,
        )

        result1 = await alert_system.send_alert(payload_intrusion)
        result2 = await alert_system.send_alert(payload_fire)

        assert result1.delivered is True
        assert result2.delivered is True


# ---------------------------------------------------------------------------
# 5. CameraPipeline construction test
# ---------------------------------------------------------------------------


class TestCameraPipelineConstruction:
    """Validates: Requirements 14.2, 14.3

    Verify that CameraPipeline correctly instantiates all sub-components
    from a CameraConfig.
    """

    def test_pipeline_components_instantiated(self) -> None:
        """CameraPipeline creates VideoFeeder, DetectionEngine, Tracker,
        EventEncoder, and HeartbeatPublisher."""
        camera_config = _make_camera_config()

        pipeline = CameraPipeline(
            camera_config=camera_config,
            mqtt_publisher=None,
        )

        # Verify all sub-components exist and are correctly typed
        assert pipeline.camera_config is camera_config
        assert pipeline.video_feeder is not None
        assert pipeline.detection_engine is not None
        assert pipeline.tracker is not None
        assert pipeline.event_encoder is not None
        assert pipeline.heartbeat_publisher is not None
        assert pipeline.is_running is False

    def test_pipeline_uses_camera_config(self) -> None:
        """CameraPipeline passes camera_config to sub-components."""
        camera_config = _make_camera_config(
            camera_id="cam-test-42",
            tracker_algorithm="bytetrack",
        )

        pipeline = CameraPipeline(
            camera_config=camera_config,
            mqtt_publisher=None,
        )

        assert pipeline.camera_config.camera_id == "cam-test-42"
        # Tracker should be created with the configured algorithm
        assert pipeline.tracker is not None
