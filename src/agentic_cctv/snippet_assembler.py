"""Snippet Assembler for the Agentic AI CCTV Monitoring Framework.

Extracts frames from a FrameRingBuffer around a detection event timestamp
and encodes them into a base64-encoded MP4 byte string via OpenCV VideoWriter.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from datetime import timedelta
from datetime import datetime
from typing import TYPE_CHECKING

import cv2

from agentic_cctv.models import Frame

if TYPE_CHECKING:
    from agentic_cctv.video_feeder import FrameRingBuffer

logger = logging.getLogger(__name__)


class SnippetAssemblyError(Exception):
    """Raised when video snippet assembly fails."""

    pass


class SnippetAssembler:
    """Assembles video snippets from buffered frames.

    Parameters
    ----------
    fps:
        Frame rate for the output MP4 video. Should match the camera's
        effective frame rate (native_fps / frame_skip).
    codec:
        FourCC codec string. Defaults to "mp4v" (MPEG-4 Part 2).
    """

    def __init__(self, fps: float, codec: str = "mp4v") -> None:
        self._fps = fps
        self._codec = codec

    def assemble(
        self,
        ring_buffer: "FrameRingBuffer",
        event_timestamp: datetime,
        duration_seconds: float,
    ) -> str:
        """Extract frames centred on event_timestamp and encode as base64 MP4.

        Parameters
        ----------
        ring_buffer:
            The camera's frame ring buffer.
        event_timestamp:
            The detection event timestamp to centre the snippet around.
        duration_seconds:
            Total snippet duration in seconds.

        Returns
        -------
        str
            Base64-encoded MP4 byte string.

        Raises
        ------
        SnippetAssemblyError
            If no frames are available or encoding fails.
        """
        half_duration = duration_seconds / 2.0
        start_time = event_timestamp - timedelta(seconds=half_duration)
        end_time = event_timestamp + timedelta(seconds=half_duration)

        frames = ring_buffer.get_frames_in_range(start_time, end_time)

        if not frames:
            raise SnippetAssemblyError(
                "No frames available in ring buffer for the requested time range"
            )

        # Warn if fewer frames than expected
        expected_frames = int(duration_seconds * self._fps)
        if len(frames) < expected_frames:
            logger.warning(
                "Snippet assembly: got %d frames, expected ~%d for %.1fs duration",
                len(frames),
                expected_frames,
                duration_seconds,
            )

        mp4_bytes = self._encode_frames_to_mp4(frames, self._fps)
        return base64.b64encode(mp4_bytes).decode("ascii")

    def _encode_frames_to_mp4(self, frames: list[Frame], fps: float) -> bytes:
        """Encode a list of frames into MP4 bytes using OpenCV VideoWriter.

        Uses a temporary file as OpenCV's VideoWriter requires a file path.
        The file is deleted after reading.
        """
        if not frames:
            raise SnippetAssemblyError("Cannot encode empty frame list")

        # Get dimensions from the first frame
        first_image = frames[0].image
        height, width = first_image.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*self._codec)

        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(tmp_fd)
            tmp_fd = None

            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise SnippetAssemblyError(
                    f"Failed to open VideoWriter with codec '{self._codec}'"
                )

            try:
                for frame in frames:
                    writer.write(frame.image)
            finally:
                writer.release()

            with open(tmp_path, "rb") as f:
                mp4_bytes = f.read()

            if not mp4_bytes:
                raise SnippetAssemblyError("VideoWriter produced empty output")

            return mp4_bytes

        except SnippetAssemblyError:
            raise
        except Exception as exc:
            raise SnippetAssemblyError(
                f"Failed to encode frames to MP4: {exc}"
            ) from exc
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
