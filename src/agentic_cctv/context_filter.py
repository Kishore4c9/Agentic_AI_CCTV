"""Context Filter for the Agentic AI CCTV Monitoring Framework.

Evaluates ``StructuredEvent`` objects against per-camera ``RuleSet`` documents
loaded from the ``RuleStore``.  Implements the Context Gate — only events that
match at least one rule (and are not suppressed) proceed to VLM reasoning.

Supports:
- ``object_type`` matching
- ``min_confidence`` threshold
- ``time_window`` (time-of-day) matching
- ``zone`` (point-in-polygon) matching
- ``compound`` conditions (AND / OR)
- ``suppress_if`` logic
- Hot-reload of rules within 5 seconds via ``reload_rules``

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from agentic_cctv.models import (
    CompoundCondition,
    FilterResult,
    Rule,
    RuleSet,
    StructuredEvent,
    SuppressCondition,
    TimeWindow,
    Zone,
)
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point-in-polygon (ray-casting algorithm)
# ---------------------------------------------------------------------------


def point_in_polygon(px: float, py: float, polygon: list[list[int]]) -> bool:
    """Determine if point (px, py) is inside a polygon using ray casting.

    Parameters
    ----------
    px, py:
        The point coordinates.
    polygon:
        A list of ``[x, y]`` vertices defining the polygon.

    Returns
    -------
    bool
        ``True`` if the point is inside the polygon.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        # Check if the ray from (px, py) going right crosses edge (i, j)
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i

    return inside


# ---------------------------------------------------------------------------
# Time-of-day matching
# ---------------------------------------------------------------------------


def time_in_window(event_time: datetime, window: TimeWindow) -> bool:
    """Check if the event's time-of-day falls within a TimeWindow.

    Handles windows that wrap past midnight (e.g., 22:00 → 06:00).

    Parameters
    ----------
    event_time:
        The event timestamp.
    window:
        The time window with ``start`` and ``end`` in ``HH:MM`` format.

    Returns
    -------
    bool
        ``True`` if the event time falls within the window.
    """
    start_parts = window.start.split(":")
    end_parts = window.end.split(":")
    start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
    end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
    event_minutes = event_time.hour * 60 + event_time.minute

    if start_minutes <= end_minutes:
        # Normal window (e.g., 08:00 → 18:00)
        return start_minutes <= event_minutes <= end_minutes
    else:
        # Wrapping window (e.g., 22:00 → 06:00)
        return event_minutes >= start_minutes or event_minutes <= end_minutes


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------


def _matches_rule(event: StructuredEvent, rule: Rule) -> bool:
    """Check if a StructuredEvent matches a single Rule's conditions.

    A rule matches if ALL of its non-None conditions are satisfied:
    - ``object_type``: event's object_type equals the rule's
    - ``min_confidence``: event's confidence >= rule's min_confidence
    - ``time_window``: event's timestamp falls within the window
    - ``zone``: event's bounding_box center falls within the polygon
    - ``compound``: compound sub-conditions are evaluated recursively

    Parameters
    ----------
    event:
        The structured event to evaluate.
    rule:
        The rule to match against.

    Returns
    -------
    bool
        ``True`` if all non-None conditions match.
    """
    # Object type filter
    if rule.object_type is not None:
        if event.object_type != rule.object_type:
            return False

    # Confidence threshold
    if rule.min_confidence is not None:
        if event.confidence < rule.min_confidence:
            return False

    # Time window
    if rule.time_window is not None:
        if not time_in_window(event.timestamp, rule.time_window):
            return False

    # Zone (point-in-polygon on bounding box center)
    if rule.zone is not None and rule.zone.polygon:
        cx = event.bounding_box.x + event.bounding_box.width / 2
        cy = event.bounding_box.y + event.bounding_box.height / 2
        if not point_in_polygon(cx, cy, rule.zone.polygon):
            return False

    # Compound conditions
    if rule.compound is not None:
        if not _matches_compound(event, rule.compound):
            return False

    return True


def _matches_compound(event: StructuredEvent, compound: CompoundCondition) -> bool:
    """Evaluate a compound condition against an event.

    Parameters
    ----------
    event:
        The structured event.
    compound:
        The compound condition with an operator (``"and"`` or ``"or"``)
        and a list of sub-condition dicts.

    Returns
    -------
    bool
        Result of the compound evaluation.
    """
    if not compound.conditions:
        return True

    results = []
    for cond_dict in compound.conditions:
        # Each sub-condition is a dict with optional keys matching Rule fields
        sub_rule = Rule(
            rule_id="__compound_sub__",
            object_type=cond_dict.get("object_type"),
            min_confidence=cond_dict.get("min_confidence"),
        )
        # Handle time_window in sub-condition
        tw_data = cond_dict.get("time_window")
        if tw_data:
            sub_rule.time_window = TimeWindow(
                start=tw_data["start"], end=tw_data["end"]
            )
        # Handle zone in sub-condition
        zone_data = cond_dict.get("zone")
        if zone_data:
            sub_rule.zone = Zone(polygon=zone_data.get("polygon", []))

        results.append(_matches_rule(event, sub_rule))

    if compound.operator == "or":
        return any(results)
    else:  # "and" is the default
        return all(results)


def _matches_suppress(
    event: StructuredEvent, suppress: SuppressCondition
) -> bool:
    """Check if a suppress_if condition matches the event.

    Parameters
    ----------
    event:
        The structured event.
    suppress:
        The suppress condition.

    Returns
    -------
    bool
        ``True`` if the suppress condition matches (event should be suppressed).
    """
    if suppress.object_type is not None:
        if event.object_type != suppress.object_type:
            return False

    if suppress.time_window is not None:
        if not time_in_window(event.timestamp, suppress.time_window):
            return False

    return True


# ---------------------------------------------------------------------------
# ContextFilter
# ---------------------------------------------------------------------------


class ContextFilter:
    """Evaluates StructuredEvents against per-camera RuleSets.

    Parameters
    ----------
    rule_store:
        The :class:`RuleStore` to load rulesets from.
    timeseries_db:
        Optional :class:`TimeSeriesDB` for logging suppressed events.
    """

    def __init__(
        self,
        rule_store: RuleStore,
        timeseries_db: Optional[TimeSeriesDB] = None,
    ) -> None:
        self._rule_store = rule_store
        self._timeseries_db = timeseries_db
        self._cache: dict[str, RuleSet] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl = 5.0  # seconds — hot-reload within 5s
        self._lock = threading.Lock()

    def evaluate(self, event: StructuredEvent) -> FilterResult:
        """Evaluate a StructuredEvent against the camera's active RuleSet.

        Parameters
        ----------
        event:
            The event to evaluate.

        Returns
        -------
        FilterResult
            The evaluation result indicating whether the event passed the
            Context Gate, which rules matched, and any suppression reason.
        """
        ruleset = self._get_ruleset(event.camera_id)
        if ruleset is None or not ruleset.rules:
            # No rules configured — suppress by default
            result = FilterResult(
                passed=False,
                matched_rules=[],
                suppressed_reason="no_active_ruleset",
            )
            self._log_suppressed(event, result)
            return result

        matched_rule_ids: list[str] = []
        for rule in ruleset.rules:
            if _matches_rule(event, rule):
                # Check suppress_if on this matching rule
                if rule.suppress_if is not None and _matches_suppress(
                    event, rule.suppress_if
                ):
                    result = FilterResult(
                        passed=False,
                        matched_rules=[rule.rule_id],
                        suppressed_reason="suppress_if",
                    )
                    self._log_suppressed(event, result)
                    return result
                matched_rule_ids.append(rule.rule_id)

        if matched_rule_ids:
            return FilterResult(
                passed=True,
                matched_rules=matched_rule_ids,
                suppressed_reason=None,
            )

        # No rules matched
        result = FilterResult(
            passed=False,
            matched_rules=[],
            suppressed_reason="no_matching_rules",
        )
        self._log_suppressed(event, result)
        return result

    def reload_rules(self, camera_id: str) -> None:
        """Force an immediate reload of the ruleset for a camera.

        Parameters
        ----------
        camera_id:
            The camera whose rules should be reloaded.
        """
        with self._lock:
            self._cache.pop(camera_id, None)
            self._cache_timestamps.pop(camera_id, None)
        logger.info("Cleared ruleset cache for camera %s", camera_id)

    # -- internal helpers ---------------------------------------------------

    def _get_ruleset(self, camera_id: str) -> Optional[RuleSet]:
        """Get the active ruleset for a camera, using a TTL cache.

        The cache ensures hot-reload within ``_cache_ttl`` seconds.
        """
        now = time.monotonic()
        with self._lock:
            cached_ts = self._cache_timestamps.get(camera_id)
            if cached_ts is not None and (now - cached_ts) < self._cache_ttl:
                return self._cache.get(camera_id)

        # Cache miss or expired — reload from store
        ruleset = self._rule_store.get_active_ruleset(camera_id)
        with self._lock:
            if ruleset is not None:
                self._cache[camera_id] = ruleset
            else:
                self._cache.pop(camera_id, None)
            self._cache_timestamps[camera_id] = now
        return ruleset

    def _log_suppressed(
        self, event: StructuredEvent, result: FilterResult
    ) -> None:
        """Log a suppressed event to the Time Series DB if available."""
        if self._timeseries_db is not None:
            try:
                self._timeseries_db.insert_event(
                    event,
                    detection_gate_passed=True,
                    context_gate_passed=False,
                )
                logger.debug(
                    "Logged suppressed event %s (reason: %s)",
                    event.event_id,
                    result.suppressed_reason,
                )
            except Exception:
                logger.error(
                    "Failed to log suppressed event %s",
                    event.event_id,
                    exc_info=True,
                )
