"""Property-based test for Structured Event Schema Completeness.

# Feature: agentic-ai-cctv-monitoring, Property 2: Structured Event Schema Completeness

**Validates: Requirements 1.1, 2.3**

For random Track and Frame inputs, the encoded event contains all required fields
with correct types and non-empty ``frame_crop``.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.event_encoder import EventEncoder
from agentic_cctv.models import BoundingBox, CameraConfig, Frame, Track

# ---------------------------------------------------------------------------
# Common object types used for generation
# ---------------------------------------------------------------------------

COMMON_OBJECT_TYPES = [
    "person",
    "vehicle",
    "animal",
    "bicycle",
    "fire",
    "truck",
    "dog",
    "cat",
    "bird",
    "motorcycle",
]

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_object_type_strategy = st.sampled_from(COMMON_OBJECT_TYPES)

_confidence_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Image dimensions — small sizes to keep tests fast
_image_width_strategy = st.integers(min_value=100, max_value=640)
_image_height_strategy = st.integers(min_value=100, max_value=480)


@st.composite
def _bounding_box_strategy(draw: st.DrawFn, max_w: int = 640, max_h: int = 480) -> BoundingBox:
    """Generate a BoundingBox that fits within the given frame dimensions."""
    x = draw(st.integers(min_value=0, max_value=max(0, max_w - 1)))
    y = draw(st.integers(min_value=0, max_value=max(0, max_h - 1)))
    width = draw(st.integers(min_value=1, max_value=max(1, max_w - x)))
    height = draw(st.integers(min_value=1, max_value=max(1, max_h - y)))
    return BoundingBox(x=x, y=y, width=width, height=height)


@st.composite
def _track_and_frame_strategy(draw: st.DrawFn) -> tuple:
    """Generate a random (Track, Frame, CameraConfig) triple.

    The bounding box is constrained to lie within the generated image
    dimensions so that the crop is always meaningful.
    """
    img_w = draw(_image_width_strategy)
    img_h = draw(_image_height_strategy)

    bbox = draw(_bounding_box_strategy(max_w=img_w, max_h=img_h))
    object_type = draw(_object_type_strategy)
    confidence = draw(_confidence_strategy)
    age = draw(st.integers(min_value=0, max_value=1000))
    is_new = draw(st.booleans())

    track = Track(
        track_id=str(uuid.uuid4()),
        object_type=object_type,
        bounding_box=bbox,
        confidence=confidence,
        age=age,
        is_new=is_new,
    )

    # Generate a random numpy image (BGR, uint8)
    image = np.random.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)
    frame_number = draw(st.integers(min_value=0, max_value=100_000))

    frame = Frame(
        camera_id="cam-test",
        timestamp=datetime(2025, 1, 15, 14, 30, 0, 123000, tzinfo=timezone.utc),
        image=image,
        frame_number=frame_number,
    )

    camera_config = CameraConfig(
        camera_id="cam-test",
        uri="rtsp://0.0.0.0:554/stream",
        tenant_id="tenant-test",
        site_id="site-test",
        confidence_threshold=0.5,
        monitored_classes=["person", "vehicle"],
    )

    return track, frame, camera_config


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestStructuredEventSchemaCompleteness:
    """Property 2: Structured Event Schema Completeness.

    **Validates: Requirements 1.1, 2.3**
    """

    @given(data=_track_and_frame_strategy())
    @settings(max_examples=20)
    def test_encoded_event_has_all_required_fields_with_correct_types(
        self,
        data: tuple,
    ) -> None:
        """For any random Track and Frame, ``EventEncoder.encode()`` produces
        a ``StructuredEvent`` with all required fields present, non-None,
        and of the correct type."""

        track, frame, camera_config = data
        encoder = EventEncoder(camera_config)
        event = encoder.encode(track, frame)

        # --- All required fields are present and non-None ---
        assert event.event_id is not None, "event_id must not be None"
        assert event.camera_id is not None, "camera_id must not be None"
        assert event.tenant_id is not None, "tenant_id must not be None"
        assert event.site_id is not None, "site_id must not be None"
        assert event.timestamp is not None, "timestamp must not be None"
        assert event.object_type is not None, "object_type must not be None"
        assert event.track_id is not None, "track_id must not be None"
        assert event.bounding_box is not None, "bounding_box must not be None"
        assert event.confidence is not None, "confidence must not be None"
        assert event.frame_crop is not None, "frame_crop must not be None"

        # --- event_id is a valid UUID string ---
        parsed_uuid = uuid.UUID(event.event_id)
        assert str(parsed_uuid) == event.event_id, (
            f"event_id {event.event_id!r} is not a canonical UUID string"
        )

        # --- frame_crop is a non-empty string containing valid base64 ---
        assert isinstance(event.frame_crop, str), "frame_crop must be a string"
        assert len(event.frame_crop) > 0, "frame_crop must be non-empty"
        decoded_bytes = base64.b64decode(event.frame_crop)
        assert len(decoded_bytes) > 0, "decoded frame_crop must be non-empty"

        # --- confidence is a float in [0.0, 1.0] ---
        assert isinstance(event.confidence, float), "confidence must be a float"
        assert 0.0 <= event.confidence <= 1.0, (
            f"confidence {event.confidence} must be in [0.0, 1.0]"
        )

        # --- bounding_box has x, y, width, height as integers ---
        bb = event.bounding_box
        assert isinstance(bb.x, int), f"bounding_box.x must be int, got {type(bb.x)}"
        assert isinstance(bb.y, int), f"bounding_box.y must be int, got {type(bb.y)}"
        assert isinstance(bb.width, int), f"bounding_box.width must be int, got {type(bb.width)}"
        assert isinstance(bb.height, int), f"bounding_box.height must be int, got {type(bb.height)}"

        # --- timestamp is a datetime ---
        assert isinstance(event.timestamp, datetime), (
            f"timestamp must be a datetime, got {type(event.timestamp)}"
        )
