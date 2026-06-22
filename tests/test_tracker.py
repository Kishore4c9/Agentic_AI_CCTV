"""Unit tests for the Tracker module."""

from __future__ import annotations

from datetime import datetime

import pytest

from agentic_cctv.models import BoundingBox, Detection, Frame, Track
from agentic_cctv.tracker import Tracker, _iou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(frame_number: int = 0) -> Frame:
    """Create a minimal Frame for testing."""
    return Frame(
        camera_id="cam-test",
        timestamp=datetime(2025, 1, 15, 12, 0, 0),
        image=None,
        frame_number=frame_number,
    )


def _make_detection(
    x: int = 100,
    y: int = 100,
    w: int = 50,
    h: int = 50,
    confidence: float = 0.9,
    object_type: str = "person",
    passed_gate: bool = True,
) -> Detection:
    return Detection(
        object_type=object_type,
        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
        confidence=confidence,
        passed_gate=passed_gate,
    )


# ---------------------------------------------------------------------------
# IoU helper tests
# ---------------------------------------------------------------------------

class TestIoU:
    def test_identical_boxes(self) -> None:
        box = BoundingBox(x=0, y=0, width=100, height=100)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        a = BoundingBox(x=0, y=0, width=50, height=50)
        b = BoundingBox(x=200, y=200, width=50, height=50)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = BoundingBox(x=0, y=0, width=100, height=100)
        b = BoundingBox(x=50, y=50, width=100, height=100)
        # Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        assert _iou(a, b) == pytest.approx(2500 / 17500)

    def test_zero_area_box(self) -> None:
        a = BoundingBox(x=0, y=0, width=0, height=0)
        b = BoundingBox(x=0, y=0, width=100, height=100)
        assert _iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tracker construction
# ---------------------------------------------------------------------------

class TestTrackerInit:
    def test_default_algorithm(self) -> None:
        tracker = Tracker()
        assert tracker._algorithm == "deepsort"
        assert tracker._max_age == 30

    def test_bytetrack_algorithm(self) -> None:
        tracker = Tracker(algorithm="bytetrack")
        assert tracker._algorithm == "bytetrack"

    def test_custom_max_age(self) -> None:
        tracker = Tracker(max_age=10)
        assert tracker._max_age == 10

    def test_unsupported_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported tracking algorithm"):
            Tracker(algorithm="unknown")


# ---------------------------------------------------------------------------
# Tracker.update behaviour
# ---------------------------------------------------------------------------

class TestTrackerUpdate:
    def test_no_detections_returns_empty(self) -> None:
        tracker = Tracker()
        tracks = tracker.update([], _make_frame())
        assert tracks == []

    def test_only_gated_detections_are_tracked(self) -> None:
        tracker = Tracker()
        detections = [
            _make_detection(passed_gate=True),
            _make_detection(x=300, passed_gate=False),
        ]
        tracks = tracker.update(detections, _make_frame())
        assert len(tracks) == 1

    def test_new_track_has_is_new_true(self) -> None:
        tracker = Tracker()
        tracks = tracker.update([_make_detection()], _make_frame())
        assert len(tracks) == 1
        assert tracks[0].is_new is True
        assert tracks[0].age == 0

    def test_track_id_is_uuid(self) -> None:
        import uuid

        tracker = Tracker()
        tracks = tracker.update([_make_detection()], _make_frame())
        # Should not raise
        uuid.UUID(tracks[0].track_id)

    def test_track_persistence_across_frames(self) -> None:
        """Same detection location across frames should keep the same track_id."""
        tracker = Tracker()
        det = _make_detection(x=100, y=100, w=50, h=50)

        tracks1 = tracker.update([det], _make_frame(0))
        assert len(tracks1) == 1
        tid = tracks1[0].track_id

        tracks2 = tracker.update([det], _make_frame(1))
        assert len(tracks2) == 1
        assert tracks2[0].track_id == tid
        assert tracks2[0].is_new is False
        assert tracks2[0].age == 1

    def test_is_new_only_on_first_frame(self) -> None:
        tracker = Tracker()
        det = _make_detection()

        t1 = tracker.update([det], _make_frame(0))
        assert t1[0].is_new is True

        t2 = tracker.update([det], _make_frame(1))
        assert t2[0].is_new is False

        t3 = tracker.update([det], _make_frame(2))
        assert t3[0].is_new is False

    def test_multiple_distinct_tracks(self) -> None:
        tracker = Tracker()
        d1 = _make_detection(x=0, y=0, w=50, h=50)
        d2 = _make_detection(x=500, y=500, w=50, h=50)

        tracks = tracker.update([d1, d2], _make_frame())
        assert len(tracks) == 2
        assert tracks[0].track_id != tracks[1].track_id

    def test_occlusion_handling_within_max_age(self) -> None:
        """Track should survive max_age frames without a matching detection."""
        tracker = Tracker(max_age=5)
        det = _make_detection(x=100, y=100, w=50, h=50)

        tracks = tracker.update([det], _make_frame(0))
        tid = tracks[0].track_id

        # 5 frames with no detections — track should still be alive
        for i in range(1, 6):
            tracks = tracker.update([], _make_frame(i))
            assert len(tracks) == 1
            assert tracks[0].track_id == tid

        # 6th frame without detection — track should be removed
        tracks = tracker.update([], _make_frame(6))
        assert len(tracks) == 0

    def test_occlusion_recovery(self) -> None:
        """Track should recover if detection reappears within max_age."""
        tracker = Tracker(max_age=5)
        det = _make_detection(x=100, y=100, w=50, h=50)

        tracks = tracker.update([det], _make_frame(0))
        tid = tracks[0].track_id

        # 3 frames without detection
        for i in range(1, 4):
            tracker.update([], _make_frame(i))

        # Detection reappears
        tracks = tracker.update([det], _make_frame(4))
        assert len(tracks) == 1
        assert tracks[0].track_id == tid

    def test_max_age_30_default(self) -> None:
        """Default max_age=30 should keep track alive for 30 empty frames."""
        tracker = Tracker()  # max_age=30
        det = _make_detection()

        tracker.update([det], _make_frame(0))

        # 30 frames with no detections — track should still be alive
        for i in range(1, 31):
            tracks = tracker.update([], _make_frame(i))
            assert len(tracks) == 1, f"Track lost at frame {i}"

        # 31st frame — track should be removed
        tracks = tracker.update([], _make_frame(31))
        assert len(tracks) == 0

    def test_track_fields_match_detection(self) -> None:
        tracker = Tracker()
        det = _make_detection(
            x=10, y=20, w=30, h=40, confidence=0.85, object_type="vehicle"
        )
        tracks = tracker.update([det], _make_frame())
        t = tracks[0]
        assert t.object_type == "vehicle"
        assert t.bounding_box == det.bounding_box
        assert t.confidence == 0.85

    def test_age_increments_each_frame(self) -> None:
        tracker = Tracker()
        det = _make_detection()

        for i in range(5):
            tracks = tracker.update([det], _make_frame(i))
            assert tracks[0].age == i

    def test_confidence_updates_on_match(self) -> None:
        tracker = Tracker()
        det1 = _make_detection(confidence=0.8)
        det2 = _make_detection(confidence=0.95)

        tracker.update([det1], _make_frame(0))
        tracks = tracker.update([det2], _make_frame(1))
        assert tracks[0].confidence == 0.95
