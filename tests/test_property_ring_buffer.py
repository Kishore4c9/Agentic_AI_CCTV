"""Property-based tests for FrameRingBuffer.

Feature: vlm-video-snippet, Property 3: Ring Buffer Circular Overwrite Preserves Most Recent Frames
Feature: vlm-video-snippet, Property 4: Ring Buffer Time-Range Retrieval Returns Correct Frames
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from hypothesis import given, settings, strategies as st

from agentic_cctv.models import Frame
from agentic_cctv.video_feeder import FrameRingBuffer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_base_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_frame(index: int) -> Frame:
    """Create a lightweight synthetic frame with a unique timestamp."""
    return Frame(
        camera_id="cam-test",
        timestamp=_base_time + timedelta(milliseconds=index * 100),
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        frame_number=index,
    )


# ---------------------------------------------------------------------------
# Property 3: Circular Overwrite Preserves Most Recent Frames
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(
    capacity=st.integers(min_value=2, max_value=200),
    extra=st.integers(min_value=1, max_value=200),
)
def test_circular_overwrite_preserves_most_recent(capacity: int, extra: int) -> None:
    """**Validates: Requirements 2.2**

    For any capacity N and any sequence of M frames pushed (M > N),
    the buffer contains exactly the last N frames in chronological order,
    and size == N.
    """
    total = capacity + extra  # M > N guaranteed
    buf = FrameRingBuffer(capacity)

    frames = [_make_frame(i) for i in range(total)]
    for f in frames:
        buf.push(f)

    assert buf.size == capacity

    stored = buf.get_all_frames()
    expected = frames[-capacity:]

    assert len(stored) == capacity
    for stored_f, expected_f in zip(stored, expected):
        assert stored_f.frame_number == expected_f.frame_number
        assert stored_f.timestamp == expected_f.timestamp


# ---------------------------------------------------------------------------
# Property 4: Time-Range Retrieval Returns Correct Frames
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(
    n_frames=st.integers(min_value=1, max_value=50),
    start_idx=st.integers(min_value=0, max_value=49),
    end_idx=st.integers(min_value=0, max_value=49),
)
def test_time_range_retrieval_returns_correct_frames(
    n_frames: int, start_idx: int, end_idx: int
) -> None:
    """**Validates: Requirements 2.3**

    For any buffer with frames having distinct timestamps, and any time range
    [start, end], get_frames_in_range returns exactly those frames with
    start <= timestamp <= end, in chronological order.
    """
    # Clamp indices to valid range
    start_idx = min(start_idx, n_frames - 1)
    end_idx = min(end_idx, n_frames - 1)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    buf = FrameRingBuffer(n_frames + 10)  # capacity larger than frames
    frames = [_make_frame(i) for i in range(n_frames)]
    for f in frames:
        buf.push(f)

    start_time = frames[start_idx].timestamp
    end_time = frames[end_idx].timestamp

    result = buf.get_frames_in_range(start_time, end_time)

    # Expected: all frames with timestamps in [start_time, end_time]
    expected = [f for f in frames if start_time <= f.timestamp <= end_time]

    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert r.frame_number == e.frame_number
        assert r.timestamp == e.timestamp
