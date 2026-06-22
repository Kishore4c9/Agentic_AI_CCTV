"""Prompt Compiler for the Agentic AI CCTV Monitoring Framework.

Converts natural language monitoring prompts into structured ``RuleSet`` JSON
documents via a pluggable LLM backend.  Supports scoped application to single
cameras, camera groups, or entire sites.

The ``LLMClient`` protocol mirrors the ``VLMBackend`` pattern — any callable
that accepts a prompt string and returns a string response can serve as the
LLM backend, making it trivial to mock in tests.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Protocol, runtime_checkable

from agentic_cctv.models import (
    BoundingBox,
    CompiledRuleSet,
    CompoundCondition,
    PromptScope,
    Rule,
    RuleSet,
    StructuredEvent,
    SuppressCondition,
    TimeWindow,
    Zone,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLMClient Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM backends used by the Prompt Compiler.

    Implementations accept a system prompt and a user prompt, and return
    the LLM's text response.  This mirrors the ``VLMBackend`` pattern so
    the LLM can be easily mocked in tests.
    """

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response from the LLM.

        Parameters
        ----------
        system_prompt:
            Instructions for the LLM describing the expected output format.
        user_prompt:
            The user's natural language monitoring prompt.

        Returns
        -------
        str
            The LLM's text response (expected to be valid JSON).
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# TestResult dataclass (stub for Task 27.1)
# ---------------------------------------------------------------------------


@dataclass
class HistoryTestResult:
    """Result of testing a RuleSet against historical events.

    This is a stub for v1 — full implementation in Task 27.1.
    """

    camera_id: str
    days_tested: int
    total_events: int = 0
    matched_events: int = 0
    expected_alert_rate: float = 0.0
    sample_matches: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RuleStore / ContextFilter protocol stubs for type hints
# ---------------------------------------------------------------------------

# We import the real classes but use Optional so the compiler can work
# without them wired up.
try:
    from agentic_cctv.rule_store import RuleStore
    from agentic_cctv.context_filter import ContextFilter, _matches_rule, _matches_suppress
    from agentic_cctv.timeseries_db import TimeSeriesDB
except ImportError:  # pragma: no cover
    RuleStore = Any  # type: ignore[assignment,misc]
    ContextFilter = Any  # type: ignore[assignment,misc]
    TimeSeriesDB = Any  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a CCTV monitoring rule compiler. Convert the user's natural language \
monitoring prompt into a structured JSON rule set.

Return ONLY a valid JSON object with these exact fields:
{
  "rules": [
    {
      "rule_id": "rule-<unique_id>",
      "object_type": "<string or null>",
      "min_confidence": <float 0.0-1.0 or null>,
      "time_window": {"start": "HH:MM", "end": "HH:MM"} or null,
      "zone": {"polygon": [[x1,y1],[x2,y2],...]} or null,
      "suppress_if": {
        "object_type": "<string or null>",
        "time_window": {"start": "HH:MM", "end": "HH:MM"} or null
      } or null,
      "compound": {
        "operator": "and" or "or",
        "conditions": [...]
      } or null
    }
  ],
  "explanation": "<human-readable explanation of what these rules do>",
  "confidence": <float 0.0-1.0 indicating how confident you are in the compilation>
}

Guidelines:
- Each rule should capture one logical condition from the prompt.
- Use object_type for filtering by object class (e.g., "person", "vehicle", "animal").
- Use min_confidence to set detection confidence thresholds.
- Use time_window for time-of-day restrictions (24-hour HH:MM format).
- Use suppress_if to prevent alerts under certain conditions.
- Use compound conditions for complex AND/OR logic.
- Set zone to null unless the prompt explicitly mentions spatial regions.
- Return ONLY valid JSON, no markdown fences or extra text.\
"""


# ---------------------------------------------------------------------------
# PromptCompiler
# ---------------------------------------------------------------------------


class PromptCompiler:
    """Converts natural language prompts into structured RuleSets.

    Parameters
    ----------
    llm_client:
        An LLM backend implementing the :class:`LLMClient` protocol.
    rule_store:
        Optional :class:`RuleStore` for persisting compiled rulesets.
    context_filter:
        Optional :class:`ContextFilter` for reloading rules after activation.
    timeseries_db:
        Optional :class:`TimeSeriesDB` for querying historical events
        (used by :meth:`test_against_history`).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        rule_store: Optional[RuleStore] = None,
        context_filter: Optional[ContextFilter] = None,
        timeseries_db: Optional[TimeSeriesDB] = None,
    ) -> None:
        self._llm_client = llm_client
        self._rule_store = rule_store
        self._context_filter = context_filter
        self._timeseries_db = timeseries_db

    async def compile(
        self, prompt: str, scope: PromptScope
    ) -> CompiledRuleSet:
        """Compile a natural language prompt into a structured RuleSet.

        Calls the LLM to convert the prompt, parses the JSON response,
        and returns a :class:`CompiledRuleSet` with explanation and
        confidence metadata.

        Parameters
        ----------
        prompt:
            The operator's natural language monitoring prompt.
        scope:
            The scope to apply the rules to (camera, camera_group, or site).

        Returns
        -------
        CompiledRuleSet
            The compiled ruleset with explanation and confidence.

        Raises
        ------
        ValueError
            If the LLM returns invalid or unparseable JSON.
        RuntimeError
            If the LLM call fails (timeout, network error, etc.).
        """
        user_prompt = (
            f"Monitoring prompt: {prompt}\n"
            f"Scope: {scope.scope_type} targeting {scope.target_ids}"
        )

        try:
            raw_response = await self._llm_client.generate(
                _SYSTEM_PROMPT, user_prompt
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        # Parse the LLM response into structured data
        parsed = _parse_llm_response(raw_response)

        rules = parsed["rules"]
        explanation = parsed["explanation"]
        confidence = parsed["confidence"]

        # Build a RuleSet for the first target camera (or a generic one)
        camera_id = scope.target_ids[0] if scope.target_ids else "unscoped"
        version_id = f"rs-{uuid.uuid4().hex[:12]}"

        ruleset = RuleSet(
            version_id=version_id,
            camera_id=camera_id,
            rules=rules,
            created_at=datetime.utcnow(),
        )

        compiled = CompiledRuleSet(
            ruleset=ruleset,
            original_prompt=prompt,
            explanation=explanation,
            confidence=confidence,
        )

        logger.info(
            "Compiled prompt into %d rules (confidence=%.2f) for scope %s",
            len(rules),
            confidence,
            scope.scope_type,
        )
        return compiled

    async def confirm_and_activate(
        self, compiled: CompiledRuleSet, scope: PromptScope
    ) -> list[str]:
        """Persist a compiled ruleset and activate it for all target cameras.

        For scope ``"camera"``, applies to the single target camera.
        For ``"camera_group"`` or ``"site"``, applies to all cameras in
        ``scope.target_ids``.

        Parameters
        ----------
        compiled:
            The compiled ruleset to activate.
        scope:
            The scope defining which cameras to apply the rules to.

        Returns
        -------
        list[str]
            The version IDs of the saved rulesets (one per camera).

        Raises
        ------
        RuntimeError
            If no RuleStore is configured.
        """
        if self._rule_store is None:
            raise RuntimeError(
                "Cannot activate rules: no RuleStore configured"
            )

        version_ids: list[str] = []

        for camera_id in scope.target_ids:
            # Create a camera-specific copy of the ruleset
            version_id = f"rs-{uuid.uuid4().hex[:12]}"
            camera_ruleset = RuleSet(
                version_id=version_id,
                camera_id=camera_id,
                rules=compiled.ruleset.rules,
                created_at=datetime.utcnow(),
            )

            saved_id = self._rule_store.save_ruleset(
                camera_id=camera_id,
                ruleset=camera_ruleset,
                original_prompt=compiled.original_prompt,
            )
            version_ids.append(saved_id)

            # Trigger context filter reload for this camera
            if self._context_filter is not None:
                self._context_filter.reload_rules(camera_id)

            logger.info(
                "Activated ruleset %s for camera %s", saved_id, camera_id
            )

        return version_ids

    async def test_against_history(
        self, ruleset: RuleSet, camera_id: str, days: int = 7
    ) -> HistoryTestResult:
        """Test a ruleset against historical events without activating it.

        Queries historical events from the :class:`TimeSeriesDB` for the
        given camera and time range, evaluates each event against the
        provided :class:`RuleSet` using the Context Filter's rule matching
        logic, and returns a :class:`HistoryTestResult` with expected alert
        volume statistics.  This is a dry-run — rules are NOT activated.

        Parameters
        ----------
        ruleset:
            The ruleset to test.
        camera_id:
            The camera to test against.
        days:
            Number of days of history to test against.

        Returns
        -------
        HistoryTestResult
            Statistics about how the ruleset would perform against
            historical events.

        Raises
        ------
        RuntimeError
            If no :class:`TimeSeriesDB` is configured.
        """
        if self._timeseries_db is None:
            raise RuntimeError(
                "Cannot test against history: no TimeSeriesDB configured"
            )

        # Compute time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        # Query historical events
        event_dicts = self._timeseries_db.get_events_by_time_range(
            camera_id=camera_id,
            start_iso=start_time.isoformat(),
            end_iso=end_time.isoformat(),
        )

        total_events = len(event_dicts)
        matched_events = 0
        sample_matches: list[dict[str, Any]] = []

        for event_dict in event_dicts:
            # Reconstruct a StructuredEvent from the DB dict
            structured_event = _reconstruct_event(event_dict)

            # Evaluate against the ruleset rules
            if _event_matches_ruleset(structured_event, ruleset):
                matched_events += 1
                if len(sample_matches) < 10:
                    sample_matches.append(event_dict)

        expected_alert_rate = (
            matched_events / total_events if total_events > 0 else 0.0
        )

        logger.info(
            "test_against_history for camera %s (%d days): "
            "%d/%d events matched (rate=%.3f)",
            camera_id,
            days,
            matched_events,
            total_events,
            expected_alert_rate,
        )

        return HistoryTestResult(
            camera_id=camera_id,
            days_tested=days,
            total_events=total_events,
            matched_events=matched_events,
            expected_alert_rate=expected_alert_rate,
            sample_matches=sample_matches,
        )


# ---------------------------------------------------------------------------
# Historical event helpers
# ---------------------------------------------------------------------------


def _reconstruct_event(event_dict: dict[str, Any]) -> StructuredEvent:
    """Reconstruct a :class:`StructuredEvent` from a database row dict.

    Parameters
    ----------
    event_dict:
        A dict as returned by :meth:`TimeSeriesDB.get_events` or
        :meth:`TimeSeriesDB.get_events_by_time_range`.

    Returns
    -------
    StructuredEvent
        The reconstructed event.
    """
    bbox_raw = event_dict.get("bounding_box", "{}")
    if isinstance(bbox_raw, str):
        bbox_data = json.loads(bbox_raw)
    else:
        bbox_data = bbox_raw

    timestamp_str = event_dict.get("timestamp", "")
    if isinstance(timestamp_str, str):
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(timestamp_str)
    else:
        timestamp = timestamp_str

    return StructuredEvent(
        event_id=event_dict.get("event_id", ""),
        camera_id=event_dict.get("camera_id", ""),
        tenant_id=event_dict.get("tenant_id", ""),
        site_id=event_dict.get("site_id", ""),
        timestamp=timestamp,
        object_type=event_dict.get("object_type", ""),
        track_id=event_dict.get("track_id", ""),
        bounding_box=BoundingBox(
            x=int(bbox_data.get("x", 0)),
            y=int(bbox_data.get("y", 0)),
            width=int(bbox_data.get("width", 0)),
            height=int(bbox_data.get("height", 0)),
        ),
        confidence=float(event_dict.get("confidence", 0.0)),
        frame_crop=event_dict.get("frame_crop", ""),
    )


def _event_matches_ruleset(event: StructuredEvent, ruleset: RuleSet) -> bool:
    """Check if an event matches a ruleset (at least one rule matches and
    is not suppressed).

    Uses the same logic as the :class:`ContextFilter`: an event matches
    if at least one rule's conditions are satisfied AND the matching rule's
    ``suppress_if`` condition does NOT also match.

    Parameters
    ----------
    event:
        The structured event to evaluate.
    ruleset:
        The ruleset to evaluate against.

    Returns
    -------
    bool
        ``True`` if the event would trigger an alert under this ruleset.
    """
    for rule in ruleset.rules:
        if _matches_rule(event, rule):
            # Check suppress_if
            if rule.suppress_if is not None and _matches_suppress(
                event, rule.suppress_if
            ):
                continue  # This rule matches but is suppressed; try others
            return True
    return False


# ---------------------------------------------------------------------------
# LLM response parsing helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Parse the LLM's raw text response into structured rule data.

    Parameters
    ----------
    raw:
        The raw text response from the LLM.

    Returns
    -------
    dict
        A dict with keys ``"rules"`` (list[Rule]), ``"explanation"`` (str),
        and ``"confidence"`` (float).

    Raises
    ------
    ValueError
        If the response is not valid JSON or is missing required fields.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (possibly with language tag)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")

    if "rules" not in data:
        raise ValueError("LLM response missing required 'rules' field")

    if not isinstance(data["rules"], list):
        raise ValueError("'rules' field must be a list")

    # Parse rules
    rules: list[Rule] = []
    for i, rule_data in enumerate(data["rules"]):
        try:
            rule = _parse_rule(rule_data, index=i)
            rules.append(rule)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid rule at index {i}: {exc}"
            ) from exc

    explanation = data.get("explanation", "No explanation provided")
    if not isinstance(explanation, str):
        explanation = str(explanation)

    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "rules": rules,
        "explanation": explanation,
        "confidence": confidence,
    }


def _parse_rule(data: dict[str, Any], index: int = 0) -> Rule:
    """Parse a single rule dict into a :class:`Rule` dataclass.

    Parameters
    ----------
    data:
        The rule dict from the LLM response.
    index:
        The rule's position in the list (used for default rule_id).

    Returns
    -------
    Rule
        The parsed rule.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Rule must be a dict, got {type(data).__name__}")

    rule_id = data.get("rule_id", f"rule-{uuid.uuid4().hex[:8]}")

    # object_type
    object_type = data.get("object_type")
    if object_type is not None and not isinstance(object_type, str):
        object_type = str(object_type)

    # min_confidence
    min_confidence = data.get("min_confidence")
    if min_confidence is not None:
        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            min_confidence = None

    # time_window
    time_window = None
    tw_data = data.get("time_window")
    if tw_data is not None and isinstance(tw_data, dict):
        start = tw_data.get("start", "")
        end = tw_data.get("end", "")
        if start and end:
            time_window = TimeWindow(start=str(start), end=str(end))

    # zone
    zone = None
    zone_data = data.get("zone")
    if zone_data is not None and isinstance(zone_data, dict):
        polygon = zone_data.get("polygon", [])
        if isinstance(polygon, list) and polygon:
            zone = Zone(polygon=polygon)

    # suppress_if
    suppress_if = None
    si_data = data.get("suppress_if")
    if si_data is not None and isinstance(si_data, dict):
        si_tw = None
        si_tw_data = si_data.get("time_window")
        if si_tw_data is not None and isinstance(si_tw_data, dict):
            si_start = si_tw_data.get("start", "")
            si_end = si_tw_data.get("end", "")
            if si_start and si_end:
                si_tw = TimeWindow(start=str(si_start), end=str(si_end))
        suppress_if = SuppressCondition(
            object_type=si_data.get("object_type"),
            time_window=si_tw,
        )

    # compound
    compound = None
    comp_data = data.get("compound")
    if comp_data is not None and isinstance(comp_data, dict):
        compound = CompoundCondition(
            operator=comp_data.get("operator", "and"),
            conditions=comp_data.get("conditions", []),
        )

    return Rule(
        rule_id=rule_id,
        object_type=object_type,
        min_confidence=min_confidence,
        time_window=time_window,
        zone=zone,
        suppress_if=suppress_if,
        compound=compound,
    )
