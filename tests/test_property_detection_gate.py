"""Property-based test for Detection Gate Correctness.

# Feature: agentic-ai-cctv-monitoring, Property 1: Detection Gate Correctness

**Validates: Requirements 1.2, 1.3**

For random confidence (0.0–1.0), random object_type, random threshold, and random
monitored_classes subset, the gate returns ``True`` iff
``confidence >= threshold AND object_type in monitored_classes``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.detection_engine import apply_detection_gate
from agentic_cctv.models import BoundingBox, CameraConfig, RawDetection

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

_confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_threshold_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_object_type_strategy = st.sampled_from(COMMON_OBJECT_TYPES)
_monitored_classes_strategy = st.lists(
    st.sampled_from(COMMON_OBJECT_TYPES),
    min_size=0,
    max_size=len(COMMON_OBJECT_TYPES),
    unique=True,
)

# A fixed bounding box — irrelevant to the gate logic but required by the model
_fixed_bbox = BoundingBox(x=0, y=0, width=100, height=100)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestDetectionGateCorrectness:
    """Property 1: Detection Gate Correctness.

    **Validates: Requirements 1.2, 1.3**
    """

    @given(
        confidence=_confidence_strategy,
        object_type=_object_type_strategy,
        threshold=_threshold_strategy,
        monitored_classes=_monitored_classes_strategy,
    )
    @settings(max_examples=20)
    def test_gate_returns_true_iff_confidence_gte_threshold_and_type_in_classes(
        self,
        confidence: float,
        object_type: str,
        threshold: float,
        monitored_classes: list[str],
    ) -> None:
        """For any combination of confidence, object_type, threshold, and
        monitored_classes, ``apply_detection_gate`` returns ``True`` iff
        ``confidence >= threshold AND object_type in monitored_classes``."""

        raw_detection = RawDetection(
            object_type=object_type,
            bounding_box=_fixed_bbox,
            confidence=confidence,
        )

        camera_config = CameraConfig(
            camera_id="test-cam",
            uri="rtsp://0.0.0.0:554/stream",
            tenant_id="test-tenant",
            site_id="test-site",
            confidence_threshold=threshold,
            monitored_classes=monitored_classes,
        )

        result = apply_detection_gate(raw_detection, camera_config)
        expected = confidence >= threshold and object_type in monitored_classes

        assert result == expected, (
            f"Gate returned {result}, expected {expected} for "
            f"confidence={confidence}, threshold={threshold}, "
            f"object_type={object_type!r}, monitored_classes={monitored_classes!r}"
        )
