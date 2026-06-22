"""Event Summarizer for the Agentic AI CCTV Monitoring Framework.

Generates natural language summaries of events on hourly and daily schedules.
Queries the Time_Series_DB for events and alerts within a time window,
generates summaries via the VLM/LLM backend, and delivers them through
the Alert System channels.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.

Requirements: 8.5, 17.2
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

from agentic_cctv.models import AlertPayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Backend Protocol for summary generation
# ---------------------------------------------------------------------------


class SummaryLLMBackend(Protocol):
    """Protocol for LLM backends used for summary generation.

    Any object with an ``analyze`` method accepting ``(image_b64, event_context)``
    can serve as a summary backend.  The ``image_b64`` parameter is passed as
    an empty string since summaries are text-only.
    """

    async def analyze(self, image_b64: str, event_context: dict) -> dict:
        """Generate a response from the LLM given event context."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# EventSummarizer
# ---------------------------------------------------------------------------


class EventSummarizer:
    """Generates natural language event summaries on hourly and daily schedules.

    Queries the :class:`~agentic_cctv.timeseries_db.TimeSeriesDB` for events
    and alerts within a time window, builds a statistical summary, optionally
    enriches it via an LLM backend, and delivers the summary through the
    :class:`~agentic_cctv.alert_system.AlertSystem`.

    Parameters
    ----------
    timeseries_db:
        The TimeSeriesDB instance to query events and alerts from.
    alert_system:
        The AlertSystem instance to deliver summaries through.
    llm_backend:
        Optional VLM/LLM backend for generating natural language summaries.
        If ``None``, only statistical summaries are produced.
    tenant_id:
        Default tenant ID for scoping queries.  Defaults to ``"default"``.
    site_id:
        Default site ID for summary metadata.  Defaults to ``"default"``.
    """

    def __init__(
        self,
        timeseries_db: Any,
        alert_system: Any,
        llm_backend: Optional[SummaryLLMBackend] = None,
        tenant_id: str = "default",
        site_id: str = "default",
    ) -> None:
        self._db = timeseries_db
        self._alert_system = alert_system
        self._llm_backend = llm_backend
        self._tenant_id = tenant_id
        self._site_id = site_id

        # Background scheduler tasks
        self._hourly_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._daily_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False

    # ------------------------------------------------------------------
    # Public API — query and summarize
    # ------------------------------------------------------------------

    def query_events_in_window(
        self,
        start_time: datetime,
        end_time: datetime,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query events from the TimeSeriesDB within a time window.

        Parameters
        ----------
        start_time:
            Start of the time window (inclusive).
        end_time:
            End of the time window (exclusive).
        tenant_id:
            Optional tenant filter.  Uses the instance default if ``None``.

        Returns
        -------
        list[dict]
            List of event rows as dictionaries.
        """
        tid = tenant_id or self._tenant_id
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()

        conn = self._db._conn
        cursor = conn.execute(
            """
            SELECT * FROM events
            WHERE tenant_id = ?
              AND timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp ASC
            """,
            (tid, start_iso, end_iso),
        )
        return [dict(row) for row in cursor.fetchall()]

    def query_alerts_in_window(
        self,
        start_time: datetime,
        end_time: datetime,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query alerts from the TimeSeriesDB within a time window.

        Parameters
        ----------
        start_time:
            Start of the time window (inclusive).
        end_time:
            End of the time window (exclusive).
        tenant_id:
            Optional tenant filter.  Uses the instance default if ``None``.

        Returns
        -------
        list[dict]
            List of alert rows as dictionaries.
        """
        tid = tenant_id or self._tenant_id
        # Use space-separated format to match SQLite CURRENT_TIMESTAMP format
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        conn = self._db._conn
        cursor = conn.execute(
            """
            SELECT * FROM alerts
            WHERE tenant_id = ?
              AND created_at >= ?
              AND created_at < ?
            ORDER BY created_at ASC
            """,
            (tid, start_str, end_str),
        )
        return [dict(row) for row in cursor.fetchall()]

    def build_statistical_summary(
        self,
        events: List[Dict[str, Any]],
        alerts: List[Dict[str, Any]],
        window_label: str,
    ) -> str:
        """Build a statistical summary string from events and alerts.

        Parameters
        ----------
        events:
            List of event dicts from the time window.
        alerts:
            List of alert dicts from the time window.
        window_label:
            Human-readable label for the time window (e.g., "Hourly", "Daily").

        Returns
        -------
        str
            A formatted statistical summary.
        """
        if not events and not alerts:
            return f"{window_label} Summary: No events or alerts recorded in this period."

        total_events = len(events)
        total_alerts = len(alerts)

        # Count events by type
        type_counts: Counter = Counter()
        for ev in events:
            type_counts[ev.get("object_type", "unknown")] += 1

        # Count events by camera
        camera_counts: Counter = Counter()
        for ev in events:
            camera_counts[ev.get("camera_id", "unknown")] += 1

        # Count alerts by threat level
        threat_counts: Counter = Counter()
        for al in alerts:
            threat_counts[al.get("threat_level", "unknown")] += 1

        lines: List[str] = [
            f"{window_label} Summary:",
            f"  Total events: {total_events}",
            f"  Total alerts: {total_alerts}",
        ]

        if type_counts:
            type_parts = ", ".join(
                f"{obj_type}: {count}" for obj_type, count in type_counts.most_common()
            )
            lines.append(f"  Events by type: {type_parts}")

        if camera_counts:
            cam_parts = ", ".join(
                f"{cam}: {count}" for cam, count in camera_counts.most_common()
            )
            lines.append(f"  Events by camera: {cam_parts}")

        if threat_counts:
            threat_parts = ", ".join(
                f"{level}: {count}" for level, count in threat_counts.most_common()
            )
            lines.append(f"  Alerts by threat level: {threat_parts}")

        return "\n".join(lines)

    async def generate_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        window_label: str = "Summary",
        tenant_id: Optional[str] = None,
    ) -> str:
        """Generate a natural language summary for a time window.

        Queries events and alerts, builds a statistical summary, and
        optionally enriches it via the LLM backend.  Falls back to the
        statistical summary if the LLM call fails.

        Parameters
        ----------
        start_time:
            Start of the time window.
        end_time:
            End of the time window.
        window_label:
            Label for the summary (e.g., "Hourly", "Daily").
        tenant_id:
            Optional tenant filter.

        Returns
        -------
        str
            The generated summary text.
        """
        events = self.query_events_in_window(start_time, end_time, tenant_id)
        alerts = self.query_alerts_in_window(start_time, end_time, tenant_id)

        statistical_summary = self.build_statistical_summary(
            events, alerts, window_label
        )

        # If no LLM backend or no events, return statistical summary directly
        if self._llm_backend is None or (not events and not alerts):
            return statistical_summary

        # Try LLM-enriched summary
        try:
            prompt_context = {
                "task": "event_summary",
                "window_label": window_label,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_events": len(events),
                "total_alerts": len(alerts),
                "statistical_summary": statistical_summary,
            }

            result = await self._llm_backend.analyze("", prompt_context)

            # Extract summary text from LLM response
            summary_text = result.get("scene_description", "")
            if not summary_text:
                summary_text = result.get("summary", "")
            if not summary_text:
                # LLM returned no usable text — fall back
                logger.warning(
                    "LLM returned empty summary; falling back to statistical summary."
                )
                return statistical_summary

            return summary_text

        except Exception as exc:
            logger.warning(
                "LLM summary generation failed (%s); falling back to statistical summary.",
                exc,
            )
            return statistical_summary

    async def deliver_summary(
        self,
        summary_text: str,
        window_label: str = "Summary",
    ) -> bool:
        """Deliver a summary through the Alert System channels.

        Creates an :class:`AlertPayload` with the summary text and sends
        it via the alert system.

        Parameters
        ----------
        summary_text:
            The summary text to deliver.
        window_label:
            Label for the summary type (used in alert metadata).

        Returns
        -------
        bool
            ``True`` if the summary was delivered successfully.
        """
        payload = AlertPayload(
            alert_id=f"summary-{uuid.uuid4().hex[:12]}",
            event_id=f"summary-{window_label.lower().replace(' ', '-')}",
            camera_id="system",
            tenant_id=self._tenant_id,
            site_id=self._site_id,
            timestamp=datetime.utcnow(),
            alert_type=f"{window_label.lower()}_summary",
            description=summary_text,
            threat_level="none",
            frame_crop_url=None,
            scene_understanding=None,
        )

        try:
            result = await self._alert_system.send_alert(payload)
            if result.delivered:
                logger.info(
                    "%s summary delivered via channels: %s",
                    window_label,
                    result.channels,
                )
                return True
            else:
                logger.warning(
                    "%s summary delivery suppressed or failed.", window_label
                )
                return False
        except Exception as exc:
            logger.error(
                "Failed to deliver %s summary: %s", window_label, exc
            )
            return False

    async def generate_and_deliver(
        self,
        start_time: datetime,
        end_time: datetime,
        window_label: str = "Summary",
        tenant_id: Optional[str] = None,
    ) -> str:
        """Generate a summary and deliver it via the alert system.

        Convenience method that combines :meth:`generate_summary` and
        :meth:`deliver_summary`.

        Parameters
        ----------
        start_time:
            Start of the time window.
        end_time:
            End of the time window.
        window_label:
            Label for the summary.
        tenant_id:
            Optional tenant filter.

        Returns
        -------
        str
            The generated summary text.
        """
        summary = await self.generate_summary(
            start_time, end_time, window_label, tenant_id
        )
        await self.deliver_summary(summary, window_label)
        return summary

    # ------------------------------------------------------------------
    # Scheduler — background periodic summary generation
    # ------------------------------------------------------------------

    async def start_scheduler(self) -> None:
        """Start the background hourly and daily summary scheduler tasks."""
        if self._running:
            logger.warning("EventSummarizer scheduler is already running.")
            return

        self._running = True
        self._hourly_task = asyncio.create_task(self._hourly_loop())
        self._daily_task = asyncio.create_task(self._daily_loop())
        logger.info("EventSummarizer scheduler started (hourly + daily).")

    async def stop_scheduler(self) -> None:
        """Stop the background scheduler tasks."""
        self._running = False

        if self._hourly_task is not None:
            self._hourly_task.cancel()
            try:
                await self._hourly_task
            except asyncio.CancelledError:
                pass
            self._hourly_task = None

        if self._daily_task is not None:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except asyncio.CancelledError:
                pass
            self._daily_task = None

        logger.info("EventSummarizer scheduler stopped.")

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the scheduler is currently running."""
        return self._running

    async def _hourly_loop(self) -> None:
        """Background loop that generates hourly summaries."""
        try:
            while self._running:
                await asyncio.sleep(3600)  # Wait 1 hour
                if not self._running:
                    break
                try:
                    now = datetime.utcnow()
                    start_time = now - timedelta(hours=1)
                    await self.generate_and_deliver(
                        start_time=start_time,
                        end_time=now,
                        window_label="Hourly",
                    )
                except Exception:
                    logger.exception("Error generating hourly summary.")
        except asyncio.CancelledError:
            pass

    async def _daily_loop(self) -> None:
        """Background loop that generates daily summaries."""
        try:
            while self._running:
                await asyncio.sleep(86400)  # Wait 24 hours
                if not self._running:
                    break
                try:
                    now = datetime.utcnow()
                    start_time = now - timedelta(days=1)
                    await self.generate_and_deliver(
                        start_time=start_time,
                        end_time=now,
                        window_label="Daily",
                    )
                except Exception:
                    logger.exception("Error generating daily summary.")
        except asyncio.CancelledError:
            pass
