"""Event Encoder for the Agentic AI CCTV Monitoring Framework.

Converts tracked objects into Structured_Events with base64-encoded JPEG
frame crops and publishes them to MQTT.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TYPE_CHECKING, runtime_checkable

import cv2
import numpy as np

from agentic_cctv.models import (
    BoundingBox,
    CameraConfig,
    Frame,
    StructuredEvent,
    Track,
)

if TYPE_CHECKING:
    from agentic_cctv.snippet_assembler import SnippetAssembler
    from agentic_cctv.video_feeder import FrameRingBuffer

logger = logging.getLogger(__name__)


@runtime_checkable
class MQTTPublisherProtocol(Protocol):
    """Protocol for MQTT publishing.

    Defines the minimal interface required by ``EventEncoder`` so that
    the real ``MQTTPublisher`` (implemented in a later task) can be
    used without a circular dependency.
    """

    async def publish(
        self, topic: str, payload: bytes, qos: int = 1, retain: bool = False
    ) -> None: ...


def _clip_bounding_box(bbox: BoundingBox, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    """Clip a bounding box to the frame boundaries.

    Returns (x1, y1, x2, y2) pixel coordinates clamped to [0, frame_w)
    and [0, frame_h) respectively.
    """
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(frame_w, bbox.x + bbox.width)
    y2 = min(frame_h, bbox.y + bbox.height)
    return x1, y1, x2, y2


def _crop_and_encode(image: np.ndarray, bbox: BoundingBox) -> str:
    """Crop the bounding box region from *image* and return a base64 JPEG string.

    The bounding box is clipped to the frame dimensions so that
    out-of-bounds coordinates do not cause errors.
    """
    frame_h, frame_w = image.shape[:2]
    x1, y1, x2, y2 = _clip_bounding_box(bbox, frame_h, frame_w)

    # Ensure we have a non-empty crop
    if x2 <= x1 or y2 <= y1:
        # Degenerate box — encode a 1×1 black pixel as fallback
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
    else:
        crop = image[y1:y2, x1:x2]

    success, buf = cv2.imencode(".jpg", crop)
    if not success:  # pragma: no cover – defensive; imencode rarely fails
        raise RuntimeError("cv2.imencode failed to encode crop as JPEG")

    return base64.b64encode(buf.tobytes()).decode("ascii")


def _structured_event_to_dict(event: StructuredEvent) -> dict[str, Any]:
    """Serialise a ``StructuredEvent`` to a JSON-compatible dict."""
    return {
        "event_id": event.event_id,
        "camera_id": event.camera_id,
        "tenant_id": event.tenant_id,
        "site_id": event.site_id,
        "timestamp": event.timestamp.isoformat(),
        "object_type": event.object_type,
        "track_id": event.track_id,
        "bounding_box": {
            "x": event.bounding_box.x,
            "y": event.bounding_box.y,
            "width": event.bounding_box.width,
            "height": event.bounding_box.height,
        },
        "confidence": event.confidence,
        "frame_crop": event.frame_crop,
        "video_snippet": event.video_snippet,
        "media_type": event.media_type,
    }


class EventEncoder:
    """Encodes tracked objects into ``StructuredEvent`` payloads.

    Parameters
    ----------
    camera_config:
        Camera configuration providing ``camera_id``, ``tenant_id``,
        and ``site_id`` for the events.
    mqtt_publisher:
        Optional MQTT publisher.  When provided, ``encode_and_publish``
        will publish the event JSON to the appropriate MQTT topic.
    snippet_assembler:
        Optional SnippetAssembler for video mode.
    ring_buffer:
        Optional FrameRingBuffer for video mode.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        mqtt_publisher: Optional[MQTTPublisherProtocol] = None,
        snippet_assembler: Optional["SnippetAssembler"] = None,
        ring_buffer: Optional["FrameRingBuffer"] = None,
    ) -> None:
        self._camera_config = camera_config
        self._mqtt_publisher = mqtt_publisher
        self._snippet_assembler = snippet_assembler
        self._ring_buffer = ring_buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, track: Track, frame: Frame) -> StructuredEvent:
        """Build a ``StructuredEvent`` from a track and its source frame.

        Steps:
        1. Crop the bounding box region from ``frame.image``.
        2. Encode the crop as a base64 JPEG string.
        3. Generate a UUID for ``event_id``.
        4. Populate all required fields from the track, frame, and
           camera configuration.
        5. If in video mode, assemble a video snippet.

        Parameters
        ----------
        track:
            The tracked object to encode.
        frame:
            The video frame from which the track was detected.

        Returns
        -------
        StructuredEvent
            A fully populated event ready for serialisation / publishing.
        """
        frame_crop_b64 = _crop_and_encode(frame.image, track.bounding_box)

        video_snippet: Optional[str] = None
        media_type: str = "image"

        if (
            self._camera_config.vlm_input_mode == "video"
            and self._snippet_assembler is not None
            and self._ring_buffer is not None
        ):
            try:
                from agentic_cctv.snippet_assembler import SnippetAssemblyError

                video_snippet = self._snippet_assembler.assemble(
                    self._ring_buffer,
                    frame.timestamp,
                    self._camera_config.vlm_video_duration_seconds,
                )
                media_type = "video"
            except Exception:
                logger.warning("Snippet assembly failed, falling back to image mode")
                media_type = "image"
                video_snippet = None

        return StructuredEvent(
            event_id=str(uuid.uuid4()),
            camera_id=self._camera_config.camera_id,
            tenant_id=self._camera_config.tenant_id,
            site_id=self._camera_config.site_id,
            timestamp=frame.timestamp,
            object_type=track.object_type,
            track_id=track.track_id,
            bounding_box=track.bounding_box,
            confidence=track.confidence,
            frame_crop=frame_crop_b64,
            video_snippet=video_snippet,
            media_type=media_type,
        )

    async def encode_and_publish(self, track: Track, frame: Frame) -> StructuredEvent:
        """Encode a track into a ``StructuredEvent`` and publish via MQTT.

        1. Calls :meth:`encode` to build the event.
        2. If an ``mqtt_publisher`` is configured, serialises the event
           to JSON and publishes it to
           ``{tenant_id}/{site_id}/{camera_id}/events`` with QoS 1.
        3. Returns the ``StructuredEvent``.

        Parameters
        ----------
        track:
            The tracked object to encode.
        frame:
            The video frame from which the track was detected.

        Returns
        -------
        StructuredEvent
            The encoded event (also published if a publisher is set).
        """
        event = self.encode(track, frame)

        if self._mqtt_publisher is not None:
            topic = (
                f"{self._camera_config.tenant_id}"
                f"/{self._camera_config.site_id}"
                f"/{self._camera_config.camera_id}"
                f"/events"
            )
            payload = json.dumps(_structured_event_to_dict(event)).encode("utf-8")
            await self._mqtt_publisher.publish(topic, payload, qos=1, retain=False)

        return event
