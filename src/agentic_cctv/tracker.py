"""Multi-object tracker for the Agentic AI CCTV Monitoring Framework.

Assigns persistent UUID-based track IDs to detections that pass the Detection
Gate, using IoU (Intersection over Union) matching to associate detections
across frames.  Supports configurable ``max_age`` for occlusion handling.

Both ``algorithm="deepsort"`` and ``algorithm="bytetrack"`` use the built-in
IoU tracker for v1 (external libraries are not required).

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from agentic_cctv.models import BoundingBox, Detection, Frame, Track


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.width, b.y + b.height

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a.width * a.height
    area_b = b.width * b.height
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


@dataclass
class _InternalTrack:
    """Internal bookkeeping for an active track."""

    track_id: str
    object_type: str
    bounding_box: BoundingBox
    confidence: float
    age: int  # frames since first seen
    frames_since_update: int  # frames since last matched detection
    is_new: bool  # True only on the very first frame


_SUPPORTED_ALGORITHMS = {"deepsort", "bytetrack"}
_IOU_THRESHOLD = 0.3


class Tracker:
    """IoU-based multi-object tracker.

    Parameters
    ----------
    algorithm:
        Tracking algorithm name.  Both ``"deepsort"`` and ``"bytetrack"``
        use the built-in IoU tracker in v1.
    max_age:
        Maximum number of frames a track is kept alive without a matching
        detection (occlusion tolerance).  Defaults to 30.
    """

    def __init__(self, algorithm: str = "deepsort", max_age: int = 30) -> None:
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported tracking algorithm {algorithm!r}. "
                f"Supported: {sorted(_SUPPORTED_ALGORITHMS)}"
            )
        self._algorithm = algorithm
        self._max_age = max_age
        self._active_tracks: list[_InternalTrack] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        """Update tracker state and return current tracks.

        Only detections with ``passed_gate=True`` are considered.  Each
        gated detection is matched to existing tracks via IoU.  Unmatched
        detections spawn new tracks.  Tracks that have not been matched
        for more than ``max_age`` frames are removed.

        Parameters
        ----------
        detections:
            Detection results from the current frame (may include
            detections that did *not* pass the gate).
        frame:
            The current video frame (used for metadata; image data is not
            required by the IoU tracker).

        Returns
        -------
        list[Track]
            Snapshot of all currently active tracks after this update.
        """
        gated = [d for d in detections if d.passed_gate]

        # Build IoU cost matrix and perform greedy matching
        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        # Compute all IoU scores
        iou_pairs: list[tuple[float, int, int]] = []
        for ti, trk in enumerate(self._active_tracks):
            for di, det in enumerate(gated):
                score = _iou(trk.bounding_box, det.bounding_box)
                if score >= _IOU_THRESHOLD:
                    iou_pairs.append((score, ti, di))

        # Greedy matching: highest IoU first
        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        for score, ti, di in iou_pairs:
            if ti in matched_track_indices or di in matched_det_indices:
                continue
            matched_track_indices.add(ti)
            matched_det_indices.add(di)

            # Update matched track
            det = gated[di]
            trk = self._active_tracks[ti]
            trk.bounding_box = det.bounding_box
            trk.confidence = det.confidence
            trk.object_type = det.object_type
            trk.age += 1
            trk.frames_since_update = 0
            trk.is_new = False

        # Age unmatched tracks
        for ti, trk in enumerate(self._active_tracks):
            if ti not in matched_track_indices:
                trk.frames_since_update += 1
                trk.age += 1
                trk.is_new = False

        # Create new tracks for unmatched detections
        for di, det in enumerate(gated):
            if di not in matched_det_indices:
                new_track = _InternalTrack(
                    track_id=str(uuid.uuid4()),
                    object_type=det.object_type,
                    bounding_box=det.bounding_box,
                    confidence=det.confidence,
                    age=0,
                    frames_since_update=0,
                    is_new=True,
                )
                self._active_tracks.append(new_track)

        # Remove stale tracks (exceeded max_age without update)
        self._active_tracks = [
            trk
            for trk in self._active_tracks
            if trk.frames_since_update <= self._max_age
        ]

        # Build output
        return [
            Track(
                track_id=trk.track_id,
                object_type=trk.object_type,
                bounding_box=trk.bounding_box,
                confidence=trk.confidence,
                age=trk.age,
                is_new=trk.is_new,
            )
            for trk in self._active_tracks
        ]
