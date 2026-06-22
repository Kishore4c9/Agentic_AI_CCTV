"""Unit tests for the DetectionEngine and apply_detection_gate."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from agentic_cctv.detection_engine import DetectionEngine, apply_detection_gate
from agentic_cctv.models import BoundingBox, CameraConfig, Detection, Frame, RawDetection
from agentic_cctv.runtimes import InferenceRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOX = BoundingBox(x=10, y=20, width=100, height=200)


def _camera_config(
    threshold: float = 0.5,
    classes: list[str] | None = None,
) -> CameraConfig:
    return CameraConfig(
        camera_id="cam-01",
        uri="rtsp://localhost/stream",
        tenant_id="tenant-1",
        site_id="site-1",
        confidence_threshold=threshold,
        monitored_classes=classes if classes is not None else ["person", "vehicle"],
    )


def _raw(object_type: str = "person", confidence: float = 0.8) -> RawDetection:
    return RawDetection(object_type=object_type, bounding_box=_BOX, confidence=confidence)


def _frame() -> Frame:
    return Frame(
        camera_id="cam-01",
        timestamp=datetime(2025, 1, 15, 12, 0, 0),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_number=1,
    )


class StubRuntime:
    """A minimal InferenceRuntime that returns pre-configured detections."""

    def __init__(self, detections: list[RawDetection] | None = None) -> None:
        self._detections = detections or []

    def load_model(self, model_path: str) -> None:  # noqa: ARG002
        pass

    def infer(self, image: np.ndarray) -> list[RawDetection]:  # noqa: ARG002
        return self._detections


# ---------------------------------------------------------------------------
# apply_detection_gate tests
# ---------------------------------------------------------------------------


class TestApplyDetectionGate:
    """Tests for the standalone apply_detection_gate function."""

    def test_passes_when_above_threshold_and_in_classes(self) -> None:
        cfg = _camera_config(threshold=0.5, classes=["person"])
        assert apply_detection_gate(_raw("person", 0.8), cfg) is True

    def test_passes_at_exact_threshold(self) -> None:
        cfg = _camera_config(threshold=0.7, classes=["person"])
        assert apply_detection_gate(_raw("person", 0.7), cfg) is True

    def test_fails_just_below_threshold(self) -> None:
        cfg = _camera_config(threshold=0.7, classes=["person"])
        assert apply_detection_gate(_raw("person", 0.6999), cfg) is False

    def test_fails_when_class_not_monitored(self) -> None:
        cfg = _camera_config(threshold=0.3, classes=["vehicle"])
        assert apply_detection_gate(_raw("person", 0.9), cfg) is False

    def test_fails_when_both_conditions_unmet(self) -> None:
        cfg = _camera_config(threshold=0.9, classes=["vehicle"])
        assert apply_detection_gate(_raw("person", 0.5), cfg) is False

    def test_passes_with_multiple_monitored_classes(self) -> None:
        cfg = _camera_config(threshold=0.5, classes=["person", "vehicle", "dog"])
        assert apply_detection_gate(_raw("dog", 0.6), cfg) is True

    def test_fails_with_empty_monitored_classes(self) -> None:
        cfg = _camera_config(threshold=0.1, classes=[])
        assert apply_detection_gate(_raw("person", 0.9), cfg) is False


# ---------------------------------------------------------------------------
# DetectionEngine tests
# ---------------------------------------------------------------------------


class TestDetectionEngine:
    """Tests for the DetectionEngine.detect method."""

    def test_returns_empty_list_when_no_detections(self) -> None:
        engine = DetectionEngine(_camera_config(), StubRuntime([]))
        result = engine.detect(_frame())
        assert result == []

    def test_all_detections_pass_gate(self) -> None:
        raws = [_raw("person", 0.9), _raw("vehicle", 0.8)]
        engine = DetectionEngine(
            _camera_config(threshold=0.5, classes=["person", "vehicle"]),
            StubRuntime(raws),
        )
        result = engine.detect(_frame())
        assert len(result) == 2
        assert all(d.passed_gate for d in result)

    def test_no_detections_pass_gate(self) -> None:
        raws = [_raw("person", 0.3), _raw("cat", 0.9)]
        engine = DetectionEngine(
            _camera_config(threshold=0.5, classes=["person"]),
            StubRuntime(raws),
        )
        result = engine.detect(_frame())
        assert len(result) == 2
        assert not result[0].passed_gate  # person below threshold
        assert not result[1].passed_gate  # cat not in monitored classes

    def test_mixed_gate_results(self) -> None:
        raws = [
            _raw("person", 0.9),   # passes
            _raw("person", 0.3),   # below threshold
            _raw("cat", 0.9),      # not monitored
            _raw("vehicle", 0.7),  # passes
        ]
        engine = DetectionEngine(
            _camera_config(threshold=0.5, classes=["person", "vehicle"]),
            StubRuntime(raws),
        )
        result = engine.detect(_frame())
        assert len(result) == 4
        assert result[0].passed_gate is True
        assert result[1].passed_gate is False
        assert result[2].passed_gate is False
        assert result[3].passed_gate is True

    def test_detection_fields_match_raw(self) -> None:
        raw = _raw("person", 0.85)
        engine = DetectionEngine(_camera_config(), StubRuntime([raw]))
        result = engine.detect(_frame())
        assert len(result) == 1
        det = result[0]
        assert det.object_type == raw.object_type
        assert det.bounding_box == raw.bounding_box
        assert det.confidence == raw.confidence

    def test_returns_detection_type(self) -> None:
        engine = DetectionEngine(_camera_config(), StubRuntime([_raw()]))
        result = engine.detect(_frame())
        assert isinstance(result[0], Detection)

    def test_stub_runtime_satisfies_protocol(self) -> None:
        """StubRuntime should satisfy the InferenceRuntime protocol."""
        assert isinstance(StubRuntime(), InferenceRuntime)
