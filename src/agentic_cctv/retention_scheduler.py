"""Retention scheduler for the Agentic AI CCTV Monitoring Framework.

Runs data retention tasks as periodic asyncio background tasks:
- Purge raw events older than 90 days (after aggregating them)
- Purge aggregated data older than 1 year
- Purge orphaned VectorDB embeddings lifecycle-linked to TimeSeriesDB

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from agentic_cctv.timeseries_db import TimeSeriesDB

logger = logging.getLogger(__name__)

# Default retention periods
_DEFAULT_RAW_EVENTS_DAYS = 90
_DEFAULT_AGGREGATED_EVENTS_DAYS = 365
_DEFAULT_INTERVAL_SECONDS = 86400  # 24 hours


class RetentionScheduler:
    """Scheduled background task that enforces data retention policies.

    Coordinates retention enforcement across TimeSeriesDB and VectorDB:

    1. Aggregate raw events older than ``raw_events_days`` into daily summaries
    2. Purge the aggregated raw events from the events table
    3. Purge aggregated data older than ``aggregated_events_days``
    4. Purge orphaned VectorDB embeddings whose events no longer exist

    Parameters
    ----------
    timeseries_db:
        The :class:`TimeSeriesDB` instance to enforce retention on.
    vector_db:
        Optional VectorDB instance.  If ``None``, VectorDB retention is
        skipped (ChromaDB is an optional dependency).
    raw_events_days:
        Number of days to retain raw events (default 90).
    aggregated_events_days:
        Number of days to retain aggregated event data (default 365).
    interval_seconds:
        How often to run retention checks in seconds (default 86400 = 24h).
    """

    def __init__(
        self,
        timeseries_db: TimeSeriesDB,
        vector_db: Optional[object] = None,
        raw_events_days: int = _DEFAULT_RAW_EVENTS_DAYS,
        aggregated_events_days: int = _DEFAULT_AGGREGATED_EVENTS_DAYS,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._tsdb = timeseries_db
        self._vdb = vector_db
        self._raw_events_days = raw_events_days
        self._aggregated_events_days = aggregated_events_days
        self._interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the scheduler loop is currently active."""
        return self._running

    async def start(self) -> None:
        """Start the retention scheduler as a background asyncio task."""
        if self._running:
            logger.warning("RetentionScheduler is already running.")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "RetentionScheduler started (interval=%ds, raw_events=%dd, "
            "aggregated=%dd).",
            self._interval_seconds,
            self._raw_events_days,
            self._aggregated_events_days,
        )

    async def stop(self) -> None:
        """Stop the retention scheduler and cancel the background task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("RetentionScheduler stopped.")

    async def _loop(self) -> None:
        """Main scheduler loop — runs retention, then sleeps."""
        try:
            while self._running:
                await self.run_retention()
                await asyncio.sleep(self._interval_seconds)
        except asyncio.CancelledError:
            pass

    async def run_retention(self) -> dict[str, int]:
        """Execute a single retention cycle.

        This method can also be called directly for testing or manual
        retention runs.

        Returns
        -------
        dict[str, int]
            Summary of actions taken with keys:
            ``aggregated_groups``, ``purged_raw_events``,
            ``purged_aggregated``, ``purged_embeddings``.
        """
        now = datetime.utcnow()
        results: dict[str, int] = {
            "aggregated_groups": 0,
            "purged_raw_events": 0,
            "purged_aggregated": 0,
            "purged_embeddings": 0,
        }

        # Step 1: Aggregate raw events older than retention period
        raw_cutoff = now - timedelta(days=self._raw_events_days)
        raw_cutoff_iso = raw_cutoff.isoformat()

        try:
            results["aggregated_groups"] = self._tsdb.aggregate_events(
                raw_cutoff_iso
            )
        except Exception:
            logger.exception("Failed to aggregate events.")

        # Step 2: Purge raw events (and related alerts/heartbeats)
        try:
            results["purged_raw_events"] = self._tsdb.purge_raw_events(
                raw_cutoff_iso
            )
        except Exception:
            logger.exception("Failed to purge raw events.")

        # Step 3: Purge aggregated data older than 1 year
        agg_cutoff = now - timedelta(days=self._aggregated_events_days)
        agg_cutoff_date = agg_cutoff.strftime("%Y-%m-%d")

        try:
            results["purged_aggregated"] = self._tsdb.purge_aggregated_events(
                agg_cutoff_date
            )
        except Exception:
            logger.exception("Failed to purge aggregated events.")

        # Step 4: Purge orphaned VectorDB embeddings
        if self._vdb is not None:
            try:
                valid_ids = self._tsdb.get_all_event_ids()
                results["purged_embeddings"] = self._vdb.purge_orphaned_embeddings(  # type: ignore[union-attr]
                    valid_ids
                )
            except Exception:
                logger.exception("Failed to purge orphaned embeddings.")

        logger.info(
            "Retention cycle complete: aggregated=%d, purged_raw=%d, "
            "purged_agg=%d, purged_embeddings=%d.",
            results["aggregated_groups"],
            results["purged_raw_events"],
            results["purged_aggregated"],
            results["purged_embeddings"],
        )
        return results
