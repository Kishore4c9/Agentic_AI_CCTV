"""VLM Reasoner for the Agentic AI CCTV Monitoring Framework.

Invokes the configured VLM backend for events that pass both the Detection
Gate and Context Gate.  Implements retry-once-then-fallback logic and
validates VLM responses against the expected schema.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentic_cctv.models import (
    CameraConfig,
    IdentifiedObject,
    SceneUnderstanding,
    StructuredEvent,
)
from agentic_cctv.timeseries_db import TimeSeriesDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid enum values for schema validation
# ---------------------------------------------------------------------------

VALID_THREAT_LEVELS = frozenset({"none", "low", "medium", "high", "critical"})
VALID_ACTIONS = frozenset({"alert", "log", "summarise", "escalate"})


# ---------------------------------------------------------------------------
# Schema validation (module-level function for independent testing)
# ---------------------------------------------------------------------------


def validate_vlm_response(response: dict) -> bool:
    """Validate a VLM response dict against the expected schema.

    Validation rules:
    - ``scene_description`` must be a string
    - ``threat_level`` must be one of: "none", "low", "medium", "high", "critical"
    - ``objects_identified`` must be a list
    - ``recommended_action`` must be one of: "alert", "log", "summarise", "escalate"
    - ``confidence`` must be a float (or int) in [0, 1]

    Parameters
    ----------
    response:
        The VLM response dictionary to validate.

    Returns
    -------
    bool
        ``True`` if the response is valid, ``False`` otherwise.
    """
    if not isinstance(response, dict):
        return False

    # scene_description — must be a string
    scene_desc = response.get("scene_description")
    if not isinstance(scene_desc, str):
        return False

    # threat_level — must be in allowed set
    threat = response.get("threat_level")
    if threat not in VALID_THREAT_LEVELS:
        return False

    # objects_identified — must be a list
    objects = response.get("objects_identified")
    if not isinstance(objects, list):
        return False

    # recommended_action — must be in allowed set
    action = response.get("recommended_action")
    if action not in VALID_ACTIONS:
        return False

    # confidence — must be a number in [0, 1]
    confidence = response.get("confidence")
    if not isinstance(confidence, (int, float)):
        return False
    if isinstance(confidence, bool):
        return False
    if confidence < 0 or confidence > 1:
        return False

    return True


# ---------------------------------------------------------------------------
# VLMReasoner
# ---------------------------------------------------------------------------


class VLMReasoner:
    """Invokes the configured VLM backend and produces SceneUnderstanding.

    Implements retry-once-then-fallback logic:
    1. First call to VLM backend
    2. On failure → retry once
    3. On second failure → fall back to rule-based classification
    4. Log all failures to TimeSeriesDB

    Parameters
    ----------
    backend:
        The VLM backend to use for analysis.
    vector_db:
        Optional VectorDB for storing VLM embeddings.
    timeseries_db:
        Optional TimeSeriesDB for logging failures.
    """

    def __init__(
        self,
        backend: Any,  # VLMBackend protocol
        vector_db: Optional[Any] = None,  # VectorDB
        timeseries_db: Optional[TimeSeriesDB] = None,
        camera_configs: Optional[dict[str, CameraConfig]] = None,
    ) -> None:
        self._backend = backend
        self._vector_db = vector_db
        self._timeseries_db = timeseries_db
        self._camera_configs: dict[str, CameraConfig] = camera_configs or {}

    async def reason(
        self,
        event: StructuredEvent,
        matched_rules: Optional[list[str]] = None,
    ) -> SceneUnderstanding:
        """Analyse an event using the VLM backend with retry and fallback.

        Parameters
        ----------
        event:
            The structured event to analyse.
        matched_rules:
            Optional list of matched rule IDs from the Context Filter,
            used for fallback rule-based classification.

        Returns
        -------
        SceneUnderstanding
            The structured scene understanding result.
        """
        event_context = {
            "event_id": event.event_id,
            "camera_id": event.camera_id,
            "tenant_id": event.tenant_id,
            "site_id": event.site_id,
            "timestamp": event.timestamp.isoformat(),
            "object_type": event.object_type,
            "track_id": event.track_id,
            "confidence": event.confidence,
        }

        # Determine media to send based on camera config
        camera_config = self._camera_configs.get(event.camera_id)
        use_video = (
            camera_config is not None
            and camera_config.vlm_input_mode == "video"
            and event.video_snippet is not None
        )

        if use_video:
            media_data = event.video_snippet
            media_type = "video"
        else:
            media_data = event.frame_crop
            media_type = "image"

        # Attempt 1
        try:
            response = await self._backend.analyze(
                media_data, event_context, media_type=media_type
            )
            if validate_vlm_response(response):
                return self._build_scene_understanding(event, response)
            else:
                logger.warning(
                    "VLM response failed schema validation for event %s, retrying",
                    event.event_id,
                )
                raise ValueError("Schema validation failed")
        except Exception as exc:
            logger.warning(
                "VLM attempt 1 failed for event %s: %s", event.event_id, exc
            )
            self._log_failure(event, f"attempt_1: {exc}")

        # Attempt 2 (retry)
        try:
            response = await self._backend.analyze(
                media_data, event_context, media_type=media_type
            )
            if validate_vlm_response(response):
                return self._build_scene_understanding(event, response)
            else:
                logger.warning(
                    "VLM retry response failed schema validation for event %s",
                    event.event_id,
                )
                raise ValueError("Schema validation failed on retry")
        except Exception as exc:
            logger.warning(
                "VLM attempt 2 failed for event %s: %s", event.event_id, exc
            )
            self._log_failure(event, f"attempt_2: {exc}")

        # Fallback to rule-based classification
        logger.info(
            "Falling back to rule-based classification for event %s",
            event.event_id,
        )
        return self._rule_based_fallback(event, matched_rules or [])

    def _build_scene_understanding(
        self, event: StructuredEvent, response: dict
    ) -> SceneUnderstanding:
        """Build a SceneUnderstanding from a validated VLM response.

        Also stores the embedding in VectorDB if available.
        """
        objects_raw = response.get("objects_identified", [])
        objects: list[IdentifiedObject] = []
        for obj in objects_raw:
            if isinstance(obj, dict):
                objects.append(
                    IdentifiedObject(
                        type=obj.get("type", "unknown"),
                        action=obj.get("action", "unknown"),
                        location=obj.get("location", "unknown"),
                    )
                )

        embedding = response.get("embedding")

        scene = SceneUnderstanding(
            event_id=event.event_id,
            scene_description=response["scene_description"],
            threat_level=response["threat_level"],
            objects_identified=objects,
            recommended_action=response["recommended_action"],
            confidence=float(response["confidence"]),
            raw_response=response,
            embedding=embedding,
        )

        # Store embedding in VectorDB if available
        if self._vector_db is not None and embedding is not None:
            try:
                self._vector_db.store_embedding(
                    event_id=event.event_id,
                    embedding=embedding,
                    metadata={
                        "camera_id": event.camera_id,
                        "tenant_id": event.tenant_id,
                        "object_type": event.object_type,
                        "threat_level": scene.threat_level,
                        "timestamp": event.timestamp.isoformat(),
                    },
                )
            except Exception:
                logger.error(
                    "Failed to store embedding for event %s",
                    event.event_id,
                    exc_info=True,
                )

        return scene

    def _rule_based_fallback(
        self, event: StructuredEvent, matched_rules: list[str]
    ) -> SceneUnderstanding:
        """Produce a basic SceneUnderstanding using rule-based classification.

        Parameters
        ----------
        event:
            The structured event.
        matched_rules:
            List of matched rule IDs from the Context Filter.

        Returns
        -------
        SceneUnderstanding
            A basic scene understanding with threat_level based on confidence
            and recommended_action="log".
        """
        # Determine threat level from event confidence
        if event.confidence >= 0.9:
            threat_level = "high"
        elif event.confidence >= 0.7:
            threat_level = "medium"
        elif event.confidence >= 0.5:
            threat_level = "low"
        else:
            threat_level = "none"

        rule_info = f" (matched rules: {', '.join(matched_rules)})" if matched_rules else ""

        return SceneUnderstanding(
            event_id=event.event_id,
            scene_description=(
                f"Rule-based fallback: {event.object_type} detected by camera "
                f"{event.camera_id} with confidence {event.confidence:.2f}{rule_info}"
            ),
            threat_level=threat_level,
            objects_identified=[
                IdentifiedObject(
                    type=event.object_type,
                    action="detected",
                    location="unknown",
                )
            ],
            recommended_action="log",
            confidence=event.confidence,
            raw_response={"fallback": True, "matched_rules": matched_rules},
            embedding=None,
        )

    def _log_failure(self, event: StructuredEvent, error_msg: str) -> None:
        """Log a VLM failure to the TimeSeriesDB.

        Persists the event with ``vlm_invoked=True`` context so failures
        are tracked in the time series data.
        """
        if self._timeseries_db is not None:
            try:
                self._timeseries_db.insert_event(
                    event,
                    detection_gate_passed=True,
                    context_gate_passed=True,
                )
                logger.debug(
                    "Logged VLM failure for event %s: %s",
                    event.event_id,
                    error_msg,
                )
            except Exception:
                logger.error(
                    "Failed to log VLM failure for event %s",
                    event.event_id,
                    exc_info=True,
                )
