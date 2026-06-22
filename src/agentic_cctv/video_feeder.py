"""Video Feeder for the Agentic AI CCTV Monitoring Framework.

Captures frames from camera sources (RTSP streams, USB devices, or video files)
and feeds them to the Detection Engine.  Runs frame capture in a background
thread and exposes an async ``start``/``stop`` lifecycle.

Uses ``from __future__ import annotations`` for PEP 604 union-type syntax
on Python 3.9+.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from agentic_cctv.models import CameraConfig, Frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame Ring Buffer
# ---------------------------------------------------------------------------


class FrameRingBuffer:
    """Thread-safe circular buffer for storing recent video frames.

    Parameters
    ----------
    capacity:
        Maximum number of frames to store. Derived from
        vlm_video_duration_seconds * effective_fps.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("FrameRingBuffer capacity must be >= 1")
        self._capacity = capacity
        self._buffer: collections.deque[Frame] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, frame: Frame) -> None:
        """Add a frame to the buffer, overwriting the oldest if at capacity.

        Thread-safe: acquires internal lock before mutation.
        """
        with self._lock:
            self._buffer.append(frame)

    def get_frames_in_range(
        self, start_time: datetime, end_time: datetime
    ) -> list[Frame]:
        """Return all buffered frames with timestamps in [start_time, end_time].

        Thread-safe: acquires internal lock for the duration of the read.
        Returns a copy of the frame list (not a view into the buffer).
        """
        with self._lock:
            return [
                f for f in self._buffer
                if start_time <= f.timestamp <= end_time
            ]

    def get_all_frames(self) -> list[Frame]:
        """Return all buffered frames in chronological order."""
        with self._lock:
            return list(self._buffer)

    @property
    def size(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def capacity(self) -> int:
        """Maximum number of frames the buffer can hold."""
        return self._capacity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health metrics container
# ---------------------------------------------------------------------------


class HealthMetrics:
    """Thread-safe container for VideoFeeder health metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.connected: bool = False
        self.total_frames_read: int = 0
        self.total_frames_emitted: int = 0
        self.last_frame_time: Optional[float] = None

    def snapshot(self) -> dict:
        """Return a copy of the current metrics as a plain dict."""
        with self._lock:
            return {
                "fps": self.fps,
                "connected": self.connected,
                "total_frames_read": self.total_frames_read,
                "total_frames_emitted": self.total_frames_emitted,
                "last_frame_time": self.last_frame_time,
            }

    def update(self, **kwargs) -> None:  # noqa: ANN003
        """Update one or more metric fields atomically."""
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# VideoFeeder
# ---------------------------------------------------------------------------


class VideoFeeder:
    """Captures frames from a camera source and makes them available via
    :meth:`get_frame`.

    Parameters
    ----------
    camera_config:
        Per-camera configuration containing the URI (RTSP URL, USB index
        string, or file path), ``frame_skip``, and camera identifiers.
    """

    def __init__(self, camera_config: CameraConfig) -> None:
        self._config = camera_config
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[Frame] = None
        self._frame_lock = threading.Lock()
        self._frame_counter: int = 0
        self._emitted_counter: int = 0
        self.health = HealthMetrics()

        # Allocate ring buffer only when vlm_input_mode == "video"
        self._ring_buffer: Optional[FrameRingBuffer] = None
        if camera_config.vlm_input_mode == "video":
            capacity = self._compute_buffer_capacity(camera_config)
            self._ring_buffer = FrameRingBuffer(capacity)

    @property
    def ring_buffer(self) -> Optional[FrameRingBuffer]:
        """Return the frame ring buffer, or None if not in video mode."""
        return self._ring_buffer

    def _compute_buffer_capacity(self, config: CameraConfig) -> int:
        """Compute ring buffer capacity from config."""
        estimated_native_fps = 30.0
        effective_fps = estimated_native_fps / max(1, config.frame_skip)
        return int(config.vlm_video_duration_seconds * effective_fps) + 1

    # -- public API ---------------------------------------------------------

    async def start(self) -> None:
        """Open the video source and begin capturing frames in a background
        thread.

        Raises
        ------
        RuntimeError
            If the feeder is already running.
        """
        if self._running:
            raise RuntimeError(
                f"VideoFeeder for camera '{self._config.camera_id}' "
                "is already running."
            )

        self._cap = self._open_capture()
        if self._cap is None or not self._cap.isOpened():
            logger.error(
                "Failed to open video source '%s' for camera '%s'.",
                self._config.uri,
                self._config.camera_id,
            )
            self.health.update(connected=False)
            return

        self.health.update(connected=True)
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"VideoFeeder-{self._config.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "VideoFeeder started for camera '%s' (source: %s, frame_skip: %d).",
            self._config.camera_id,
            self._config.uri,
            self._config.frame_skip,
        )

    async def stop(self) -> None:
        """Stop the capture thread and release the video source."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.health.update(connected=False)
        logger.info(
            "VideoFeeder stopped for camera '%s'.",
            self._config.camera_id,
        )

    def get_frame(self) -> Optional[Frame]:
        """Return the most recently captured frame, or ``None`` if no frame
        is available yet."""
        with self._frame_lock:
            return self._latest_frame

    # -- internal -----------------------------------------------------------

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Create an OpenCV ``VideoCapture`` for the configured URI.

        Supports:
        - RTSP URLs  (e.g. ``"rtsp://..."``).
        - USB camera indices (e.g. ``"0"``, ``"1"``).
        - File paths (e.g. ``"./video.mp4"``).
        """
        uri = self._config.uri
        try:
            # USB camera index — the URI is a non-negative integer string.
            if uri.isdigit():
                cap = cv2.VideoCapture(int(uri))
            else:
                cap = cv2.VideoCapture(uri)
            return cap
        except Exception:
            logger.exception(
                "Error opening video source '%s' for camera '%s'.",
                uri,
                self._config.camera_id,
            )
            return None

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames, apply frame-skip
        throttling, and update :attr:`_latest_frame`."""
        fps_window_start = time.monotonic()
        fps_frame_count = 0
        frame_skip = max(1, self._config.frame_skip)

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.warning(
                    "VideoCapture lost for camera '%s'. Attempting reconnect…",
                    self._config.camera_id,
                )
                self.health.update(connected=False)
                self._attempt_reconnect()
                continue

            ret, image = self._cap.read()
            if not ret:
                # End of file for video files, or transient read failure.
                if self._is_file_source():
                    logger.info(
                        "End of video file for camera '%s'.",
                        self._config.camera_id,
                    )
                    self._running = False
                    break
                logger.warning(
                    "Frame read failed for camera '%s'. Retrying…",
                    self._config.camera_id,
                )
                self.health.update(connected=False)
                time.sleep(0.1)
                continue

            self.health.update(connected=True)
            self._frame_counter += 1
            self.health.update(total_frames_read=self._frame_counter)

            # Frame-skip throttling: only emit every Nth frame.
            if self._frame_counter % frame_skip != 0:
                continue

            self._emitted_counter += 1
            now = time.monotonic()
            frame = Frame(
                camera_id=self._config.camera_id,
                timestamp=datetime.now(timezone.utc),
                image=image,
                frame_number=self._emitted_counter,
            )

            with self._frame_lock:
                self._latest_frame = frame

            # Push to ring buffer if in video mode
            if self._ring_buffer is not None:
                self._ring_buffer.push(frame)

            self.health.update(
                total_frames_emitted=self._emitted_counter,
                last_frame_time=now,
            )

            # FPS calculation over a rolling 1-second window.
            fps_frame_count += 1
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                self.health.update(fps=fps_frame_count / elapsed)
                fps_window_start = now
                fps_frame_count = 0

    def _attempt_reconnect(self) -> None:
        """Try to re-open the video source after a connection loss."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # Back off briefly before retrying.
        time.sleep(1.0)
        self._cap = self._open_capture()
        if self._cap is not None and self._cap.isOpened():
            logger.info(
                "Reconnected to video source '%s' for camera '%s'.",
                self._config.uri,
                self._config.camera_id,
            )
            self.health.update(connected=True)
        else:
            logger.warning(
                "Reconnect failed for camera '%s'. Will retry…",
                self._config.camera_id,
            )

    def _is_file_source(self) -> bool:
        """Return ``True`` if the configured URI looks like a local file path
        rather than a live stream or USB index."""
        uri = self._config.uri
        if uri.isdigit():
            return False
        if uri.lower().startswith("rtsp://"):
            return False
        return True
