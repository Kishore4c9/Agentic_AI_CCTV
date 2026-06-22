"""Unit tests for the SnippetAssembler module."""

from __future__ import annotations

import base64
import os
import tempfile
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from agentic_cctv.models import Frame
from agentic_cctv.snippet_assembler import SnippetAssembler, SnippetAssemblyError
from agentic_cctv.video_feeder import FrameRingBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_base_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_frame(index: int, width: int = 64, height: int = 48) -> Frame:
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Frame(
        camera_id="cam-test",
        timestamp=_base_time + timedelta(seconds=index),
        image=image,
        frame_number=index,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnippetAssembler:
    def test_known_5_frame_input_produces_valid_mp4(self):
        """5-frame input produces valid decodable MP4 output."""
        buf = FrameRingBuffer(10)
        for i in range(5):
            buf.push(_make_frame(i, width=64, height=48))

        assembler = SnippetAssembler(fps=10.0)
        event_ts = _base_time + timedelta(seconds=2)
        result_b64 = assembler.assemble(buf, event_ts, 6.0)

        # Decode base64
        mp4_bytes = base64.b64decode(result_b64)
        assert len(mp4_bytes) > 0

        # Verify it's a valid MP4 by decoding with OpenCV
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            os.write(tmp_fd, mp4_bytes)
            os.close(tmp_fd)

            cap = cv2.VideoCapture(tmp_path)
            assert cap.isOpened()
            ret, frame = cap.read()
            assert ret
            assert frame.shape == (48, 64, 3)
            cap.release()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def test_partial_buffer_assembles_with_warning(self, caplog):
        """Fewer frames than duration assembles with available frames and logs warning."""
        buf = FrameRingBuffer(10)
        # Only 2 frames for a 10-second duration
        for i in range(2):
            buf.push(_make_frame(i))

        assembler = SnippetAssembler(fps=10.0)
        event_ts = _base_time + timedelta(seconds=0.5)

        with caplog.at_level("WARNING"):
            result_b64 = assembler.assemble(buf, event_ts, 10.0)

        assert len(result_b64) > 0
        assert "got 2 frames" in caplog.text

    def test_zero_frames_raises_snippet_assembly_error(self):
        """Empty buffer raises SnippetAssemblyError."""
        buf = FrameRingBuffer(10)
        assembler = SnippetAssembler(fps=10.0)
        event_ts = _base_time

        with pytest.raises(SnippetAssemblyError, match="No frames available"):
            assembler.assemble(buf, event_ts, 5.0)

    def test_base64_encoding_is_valid(self):
        """Output is valid base64."""
        buf = FrameRingBuffer(10)
        for i in range(3):
            buf.push(_make_frame(i))

        assembler = SnippetAssembler(fps=10.0)
        event_ts = _base_time + timedelta(seconds=1)
        result_b64 = assembler.assemble(buf, event_ts, 4.0)

        # Should not raise
        decoded = base64.b64decode(result_b64)
        assert isinstance(decoded, bytes)
        assert len(decoded) > 0
