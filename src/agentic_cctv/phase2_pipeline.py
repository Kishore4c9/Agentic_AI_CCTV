"""Phase 2 pipeline wiring: ContextFilter → VLMReasoner → OrchestrationAgent → AlertSystem.

Provides the ``ContextFilterSubscriber`` MQTT callback class and the
``Phase2Pipeline`` coordinator that wires all Phase 2 components together.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

from agentic_cctv.alert_system import AlertSystem
from agentic_cctv.context_filter import ContextFilter
from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    StructuredEvent,
)
from agentic_cctv.orchestration_agent import OrchestrationAgent
from agentic_cctv.vlm_reasoner import VLMReasoner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContextFilterSubscriber — MQTT callback for Phase 2 pipeline
# ---------------------------------------------------------------------------


class ContextFilterSubscriber:
    """MQTT subscriber callback that evaluates events via the ContextFilter
    and forwards passing events through the Phase 2 pipeline:
    ContextFilter → VLMReasoner → OrchestrationAgent → AlertSystem.

    Suppressed events are already logged to TimeSeriesDB by the ContextFilter
    internally.

    Parameters
    ----------
    context_filter:
        The :class:`ContextFilter` instance for rule evaluation.
    vlm_reasoner:
        The :class:`VLMReasoner` instance for scene understanding.
    orchestration_agent:
        The :class:`OrchestrationAgent` instance for action decisions.
    alert_system:
        The :class:`AlertSystem` instance for alert delivery.
    loop:
        Optional asyncio event loop for scheduling async calls from
        synchronous MQTT callbacks.  If ``None``, a new loop is created.
    """

    def __init__(
        self,
        context_filter: ContextFilter,
        vlm_reasoner: VLMReasoner,
        orchestration_agent: OrchestrationAgent,
        alert_system: AlertSystem,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._context_filter = context_filter
        self._vlm_reasoner = vlm_reasoner
        self._orchestration_agent = orchestration_agent
        self._alert_system = alert_system
        self._loop = loop

    def __call__(self, topic: str, payload: bytes, qos: int) -> None:
        """Handle an incoming MQTT message by evaluating and processing the event.

        Parameters
        ----------
        topic:
            The MQTT topic the message was received on.
        payload:
            The raw message payload (JSON-encoded ``StructuredEvent``).
        qos:
            The QoS level of the received message.
        """
        try:
            event = self._deserialize_event(payload)
        except Exception as exc:
            logger.error(
                "Failed to deserialize event from topic %s: %s", topic, exc
            )
            return

        # Evaluate event against the camera's active RuleSet
        filter_result = self._context_filter.evaluate(event)

        if not filter_result.passed:
            logger.debug(
                "Event %s suppressed by ContextFilter (reason: %s)",
                event.event_id,
                filter_result.suppressed_reason,
            )
            return

        # Event passed the Context Gate — forward through Phase 2 pipeline
        logger.info(
            "Event %s passed ContextFilter (matched rules: %s), "
            "forwarding to VLMReasoner",
            event.event_id,
            filter_result.matched_rules,
        )

        try:
            self._process_event_async(event, filter_result.matched_rules)
        except Exception:
            logger.error(
                "Failed to process event %s through Phase 2 pipeline",
                event.event_id,
                exc_info=True,
            )

    def _process_event_async(
        self, event: StructuredEvent, matched_rules: list[str]
    ) -> None:
        """Schedule the async Phase 2 pipeline processing.

        Handles the sync-to-async bridge required because MQTT callbacks
        are synchronous but VLMReasoner.reason() and AlertSystem.send_alert()
        are async.
        """
        loop = self._loop
        if loop is not None and loop.is_running():
            # Schedule on the existing running loop
            future = asyncio.run_coroutine_threadsafe(
                self._run_pipeline(event, matched_rules), loop
            )
            # Wait for completion to ensure pipeline finishes before returning
            try:
                future.result(timeout=60)
            except Exception:
                logger.error(
                    "Phase 2 pipeline timed out or failed for event %s",
                    event.event_id,
                    exc_info=True,
                )
        else:
            # No running loop — try to get the current loop or create one
            try:
                current_loop = asyncio.get_event_loop()
                if current_loop.is_running():
                    # We're inside a running loop but no loop was provided
                    # Use run_coroutine_threadsafe with the current loop
                    future = asyncio.run_coroutine_threadsafe(
                        self._run_pipeline(event, matched_rules), current_loop
                    )
                    try:
                        future.result(timeout=60)
                    except Exception:
                        logger.error(
                            "Phase 2 pipeline failed for event %s",
                            event.event_id,
                            exc_info=True,
                        )
                else:
                    current_loop.run_until_complete(
                        self._run_pipeline(event, matched_rules)
                    )
            except RuntimeError:
                # No event loop exists — create a temporary one
                asyncio.run(self._run_pipeline(event, matched_rules))

    async def _run_pipeline(
        self, event: StructuredEvent, matched_rules: list[str]
    ) -> None:
        """Execute the full Phase 2 pipeline for a single event.

        ContextFilter (already passed) → VLMReasoner → OrchestrationAgent → AlertSystem.
        """
        # Step 1: VLM Reasoning
        try:
            scene = await self._vlm_reasoner.reason(event, matched_rules)
        except Exception:
            logger.error(
                "VLMReasoner failed for event %s",
                event.event_id,
                exc_info=True,
            )
            return

        # Step 2: Orchestration Agent decides action
        try:
            action_result = self._orchestration_agent.process(scene, event)
        except Exception:
            logger.error(
                "OrchestrationAgent failed for event %s",
                event.event_id,
                exc_info=True,
            )
            return

        logger.info(
            "OrchestrationAgent decided action=%s for event %s",
            action_result.action,
            event.event_id,
        )

        # Step 3: Execute alert if action is "alert" and payload exists
        if action_result.action == "alert" and action_result.alert_payload is not None:
            try:
                alert_result = await self._alert_system.send_alert(
                    action_result.alert_payload
                )
                logger.info(
                    "Alert sent for event %s: delivered=%s, channels=%s",
                    event.event_id,
                    alert_result.delivered,
                    alert_result.channels,
                )
            except Exception:
                logger.error(
                    "AlertSystem failed for event %s",
                    event.event_id,
                    exc_info=True,
                )

    @staticmethod
    def _deserialize_event(payload: bytes) -> StructuredEvent:
        """Deserialize a JSON payload into a StructuredEvent.

        Parameters
        ----------
        payload:
            JSON-encoded event bytes.

        Returns
        -------
        StructuredEvent
            The deserialized event.

        Raises
        ------
        ValueError, KeyError, json.JSONDecodeError
            If the payload cannot be parsed.
        """
        data = json.loads(payload)
        bbox_data = data.get("bounding_box", {})
        bounding_box = BoundingBox(
            x=int(bbox_data.get("x", 0)),
            y=int(bbox_data.get("y", 0)),
            width=int(bbox_data.get("width", 0)),
            height=int(bbox_data.get("height", 0)),
        )

        timestamp_str = data.get("timestamp", "")
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(timestamp_str)

        return StructuredEvent(
            event_id=data["event_id"],
            camera_id=data["camera_id"],
            tenant_id=data["tenant_id"],
            site_id=data["site_id"],
            timestamp=timestamp,
            object_type=data["object_type"],
            track_id=data["track_id"],
            bounding_box=bounding_box,
            confidence=float(data["confidence"]),
            frame_crop=data.get("frame_crop", ""),
        )
