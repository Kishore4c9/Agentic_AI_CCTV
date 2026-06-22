"""Unit tests for the VideoFeeder component.

Tests cover:
- Source type detection (RTSP, USB index, file path)
- Frame-skip throttling
- Async start/stop lifecycle
- Health metrics tracking
- get_frame returns latest frame or None
- Connection error handling (graceful degradation)
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from agentic_cctv.models import CameraConfig, Frame
from agentic_cctv.video_feeder import HealthMetrics, VideoFeeder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera_config(**overrides) -> CameraConfig:
    """Create a CameraConfig with sensible defaults, applying *overrides*."""
    defaults = dict(
        camera_id="cam-test-01",
        uri="./test_video.mp4",
        tenant_id="tenant-test",
        site_id="site-test",
        confidence_threshold=0.7,
        monitored_classes=["person"],
        inference_runtime="pytorch",
        model_path="./models/yolov8n.pt",
        tracker_algorithm="deepsort",
        frame_skip=1,
    )
    defaults.update(overrides)
    return CameraConfig(**defaults)


class FakeVideoCapture:
    """A lightweight stand-in for cv2.VideoCapture that yields a fixed number
    of synthetic frames and then signals end-of-stream."""

    def __init__(self, total_frames: int = 10, width: int = 640, height: int = 480):
        self._total = total_frames
        self._read_count = 0
        self._width = width
        self._height = height
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if not self._opened or self._read_count >= self._total:
            return False, None
        self._read_count += 1
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self._opened = False


# ---------------------------------------------------------------------------
# HealthMetrics tests
# ---------------------------------------------------------------------------


class TestHealthMetrics:
    def test_initial_values(self):
        hm = HealthMetrics()
        snap = hm.snapshot()
        assert snap["fps"] == 0.0
        assert snap["connected"] is False
        assert snap["total_frames_read"] == 0
        assert snap["total_frames_emitted"] == 0
        assert snap["last_frame_time"] is None

    def test_update_and_snapshot(self):
        hm = HealthMetrics()
        hm.update(fps=15.0, connected=True, total_frames_read=100)
        snap = hm.snapshot()
        assert snap["fps"] == 15.0
        assert snap["connected"] is True
        assert snap["total_frames_read"] == 100

    def test_thread_safety(self):
        """Concurrent updates should not raise or corrupt state."""
        hm = HealthMetrics()
        errors = []

        def writer():
            try:
                for i in range(200):
                    hm.update(total_frames_read=i)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    hm.snapshot()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# VideoFeeder — source type detection
# ---------------------------------------------------------------------------


class TestSourceTypeDetection:
    def test_usb_index_detected(self):
        vf = VideoFeeder(_make_camera_config(uri="0"))
        assert not vf._is_file_source()

    def test_rtsp_detected(self):
        vf = VideoFeeder(_make_camera_config(uri="rtsp://192.168.1.1:554/stream"))
        assert not vf._is_file_source()

    def test_file_path_detected(self):
        vf = VideoFeeder(_make_camera_config(uri="./videos/test.mp4"))
        assert vf._is_file_source()

    def test_absolute_file_path(self):
        vf = VideoFeeder(_make_camera_config(uri="/home/user/video.avi"))
        assert vf._is_file_source()


# ---------------------------------------------------------------------------
# VideoFeeder — frame-skip throttling
# ---------------------------------------------------------------------------


class TestFrameSkipThrottling:
    @pytest.mark.asyncio
    async def test_frame_skip_1_emits_all(self):
        """With frame_skip=1, every frame should be emitted."""
        total = 5
        config = _make_camera_config(frame_skip=1)
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(total)):
            await vf.start()
            # Wait for the capture thread to finish (file source stops itself).
            await asyncio.sleep(0.5)
            await vf.stop()

        assert vf.health.total_frames_read == total
        assert vf.health.total_frames_emitted == total

    @pytest.mark.asyncio
    async def test_frame_skip_3_emits_every_third(self):
        """With frame_skip=3, only every 3rd frame should be emitted."""
        total = 9
        config = _make_camera_config(frame_skip=3)
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(total)):
            await vf.start()
            await asyncio.sleep(0.5)
            await vf.stop()

        assert vf.health.total_frames_read == total
        assert vf.health.total_frames_emitted == 3  # frames 3, 6, 9

    @pytest.mark.asyncio
    async def test_frame_skip_larger_than_total(self):
        """If frame_skip > total frames, only 1 frame (or 0) should be emitted."""
        total = 2
        config = _make_camera_config(frame_skip=5)
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(total)):
            await vf.start()
            await asyncio.sleep(0.5)
            await vf.stop()

        assert vf.health.total_frames_read == total
        # frame_counter goes 1, 2 — neither is divisible by 5
        assert vf.health.total_frames_emitted == 0


# ---------------------------------------------------------------------------
# VideoFeeder — async lifecycle
# ---------------------------------------------------------------------------


class TestAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        config = _make_camera_config()
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(50)):
            await vf.start()
            assert vf._running is True
            assert vf.health.snapshot()["connected"] is True

            await vf.stop()
            assert vf._running is False
            assert vf._cap is None

    @pytest.mark.asyncio
    async def test_double_start_raises(self):
        config = _make_camera_config()
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(100)):
            await vf.start()
            with pytest.raises(RuntimeError, match="already running"):
                await vf.start()
            await vf.stop()

    @pytest.mark.asyncio
    async def test_start_with_failed_capture(self):
        """If the video source cannot be opened, start should not crash."""
        config = _make_camera_config()
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=None):
            await vf.start()
            # Should not be running since capture failed.
            assert vf._running is False
            assert vf.health.snapshot()["connected"] is False


# ---------------------------------------------------------------------------
# VideoFeeder — get_frame
# ---------------------------------------------------------------------------


class TestGetFrame:
    @pytest.mark.asyncio
    async def test_get_frame_returns_none_before_start(self):
        config = _make_camera_config()
        vf = VideoFeeder(config)
        assert vf.get_frame() is None

    @pytest.mark.asyncio
    async def test_get_frame_returns_frame_after_capture(self):
        config = _make_camera_config(frame_skip=1)
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(5)):
            await vf.start()
            await asyncio.sleep(0.5)
            frame = vf.get_frame()
            await vf.stop()

        assert frame is not None
        assert isinstance(frame, Frame)
        assert frame.camera_id == "cam-test-01"
        assert isinstance(frame.image, np.ndarray)
        assert frame.image.shape == (480, 640, 3)
        assert frame.frame_number > 0

    @pytest.mark.asyncio
    async def test_frame_has_utc_timestamp(self):
        config = _make_camera_config(frame_skip=1)
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(3)):
            await vf.start()
            await asyncio.sleep(0.5)
            frame = vf.get_frame()
            await vf.stop()

        assert frame is not None
        assert frame.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# VideoFeeder — health metrics
# ---------------------------------------------------------------------------


class TestHealthMetricsIntegration:
    @pytest.mark.asyncio
    async def test_health_tracks_frame_counts(self):
        total = 6
        config = _make_camera_config(frame_skip=2)
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(total)):
            await vf.start()
            await asyncio.sleep(0.5)
            await vf.stop()

        snap = vf.health.snapshot()
        assert snap["total_frames_read"] == total
        assert snap["total_frames_emitted"] == 3  # frames 2, 4, 6

    @pytest.mark.asyncio
    async def test_health_connected_false_after_stop(self):
        config = _make_camera_config()
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(5)):
            await vf.start()
            await asyncio.sleep(0.3)
            await vf.stop()

        assert vf.health.snapshot()["connected"] is False


# ---------------------------------------------------------------------------
# VideoFeeder — connection error handling
# ---------------------------------------------------------------------------


class TestConnectionErrorHandling:
    @pytest.mark.asyncio
    async def test_open_capture_exception_returns_none(self):
        """If cv2.VideoCapture raises, _open_capture should return None."""
        config = _make_camera_config(uri="rtsp://bad-host:554/stream")
        vf = VideoFeeder(config)

        with patch("agentic_cctv.video_feeder.cv2.VideoCapture", side_effect=Exception("connection refused")):
            cap = vf._open_capture()
            assert cap is None

    @pytest.mark.asyncio
    async def test_file_source_stops_at_eof(self):
        """A file-based source should stop cleanly at end of file."""
        config = _make_camera_config(uri="./test.mp4", frame_skip=1)
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(3)):
            await vf.start()
            await asyncio.sleep(0.5)
            # The thread should have stopped itself.
            assert vf._running is False
            await vf.stop()


# ---------------------------------------------------------------------------
# FrameRingBuffer tests
# ---------------------------------------------------------------------------

from agentic_cctv.video_feeder import FrameRingBuffer


class TestFrameRingBuffer:
    def test_empty_buffer_returns_empty_list(self):
        buf = FrameRingBuffer(10)
        result = buf.get_frames_in_range(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        assert result == []
        assert buf.size == 0

    def test_single_frame_push_and_retrieve(self):
        buf = FrameRingBuffer(5)
        frame = Frame(
            camera_id="cam-01",
            timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            frame_number=1,
        )
        buf.push(frame)
        assert buf.size == 1
        all_frames = buf.get_all_frames()
        assert len(all_frames) == 1
        assert all_frames[0].frame_number == 1

    def test_exact_capacity_boundary(self):
        """Push exactly N frames into capacity-N buffer."""
        n = 5
        buf = FrameRingBuffer(n)
        for i in range(n):
            buf.push(Frame(
                camera_id="cam-01",
                timestamp=datetime(2025, 1, 15, 12, 0, i, tzinfo=timezone.utc),
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                frame_number=i,
            ))
        assert buf.size == n
        assert buf.capacity == n
        all_frames = buf.get_all_frames()
        assert len(all_frames) == n
        assert [f.frame_number for f in all_frames] == list(range(n))

    def test_get_all_frames_returns_chronological_order(self):
        buf = FrameRingBuffer(3)
        for i in range(5):
            buf.push(Frame(
                camera_id="cam-01",
                timestamp=datetime(2025, 1, 15, 12, 0, i, tzinfo=timezone.utc),
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                frame_number=i,
            ))
        all_frames = buf.get_all_frames()
        assert [f.frame_number for f in all_frames] == [2, 3, 4]


# ---------------------------------------------------------------------------
# VideoFeeder ring buffer integration tests
# ---------------------------------------------------------------------------


class TestVideoFeederRingBuffer:
    def test_image_mode_no_ring_buffer(self):
        """vlm_input_mode='image' does not allocate ring buffer."""
        config = _make_camera_config(vlm_input_mode="image")
        vf = VideoFeeder(config)
        assert vf.ring_buffer is None

    def test_video_mode_allocates_ring_buffer(self):
        """vlm_input_mode='video' allocates ring buffer with correct capacity."""
        config = _make_camera_config(
            vlm_input_mode="video",
            vlm_video_duration_seconds=10,
            frame_skip=3,
        )
        vf = VideoFeeder(config)
        assert vf.ring_buffer is not None
        # capacity = int(10 * (30.0 / 3)) + 1 = 101
        assert vf.ring_buffer.capacity == 101

    @pytest.mark.asyncio
    async def test_video_mode_pushes_frames_to_ring_buffer(self):
        """In video mode, captured frames are pushed to the ring buffer."""
        config = _make_camera_config(
            vlm_input_mode="video",
            vlm_video_duration_seconds=5,
            frame_skip=1,
        )
        vf = VideoFeeder(config)

        with patch.object(vf, "_open_capture", return_value=FakeVideoCapture(5)):
            await vf.start()
            await asyncio.sleep(0.5)
            await vf.stop()

        assert vf.ring_buffer is not None
        assert vf.ring_buffer.size == 5
