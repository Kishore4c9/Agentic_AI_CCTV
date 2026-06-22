"""Property-based test for Context Filter Rule Evaluation Correctness.

# Feature: agentic-ai-cctv-monitoring, Property 5: Context Filter Rule Evaluation Correctness

**Validates: Requirements 4.1, 4.2, 4.3, 4.5**

For random StructuredEvents and random RuleSets, the filter returns
``passed=True`` iff at least one rule matches AND no matching rule's
``suppress_if`` also matches.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentic_cctv.context_filter import (
    ContextFilter,
    _matches_rule,
    _matches_suppress,
    point_in_polygon,
    time_in_window,
)
from agentic_cctv.models import (
    BoundingBox,
    FilterResult,
    Rule,
    RuleSet,
    StructuredEvent,
    SuppressCondition,
    TimeWindow,
    Zone,
)
from agentic_cctv.rule_store import RuleStore

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_object_types = ["person", "vehicle", "animal", "package", "bicycle", "fire"]

_object_type_strategy = st.sampled_from(_object_types)

_confidence_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

_hh_mm_strategy = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59),
)

_time_window_strategy = st.builds(
    TimeWindow, start=_hh_mm_strategy, end=_hh_mm_strategy
)

_bounding_box_strategy = st.builds(
    BoundingBox,
    x=st.integers(min_value=0, max_value=1000),
    y=st.integers(min_value=0, max_value=1000),
    width=st.integers(min_value=10, max_value=500),
    height=st.integers(min_value=10, max_value=500),
)

_timestamp_strategy = st.builds(
    lambda h, m: datetime(2025, 6, 15, h, m, 0),
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59),
)

_camera_id_strategy = st.from_regex(r"cam-[a-z0-9]{3,8}", fullmatch=True)

_suppress_condition_strategy = st.builds(
    SuppressCondition,
    object_type=st.one_of(st.none(), _object_type_strategy),
    time_window=st.one_of(st.none(), _time_window_strategy),
)

_rule_strategy = st.builds(
    Rule,
    rule_id=st.builds(lambda: f"rule-{uuid.uuid4().hex[:8]}"),
    object_type=st.one_of(st.none(), _object_type_strategy),
    min_confidence=st.one_of(st.none(), _confidence_strategy),
    time_window=st.one_of(st.none(), _time_window_strategy),
    zone=st.none(),  # Zone tested separately to keep strategies manageable
    suppress_if=st.one_of(st.none(), _suppress_condition_strategy),
    compound=st.none(),
)

_rules_list_strategy = st.lists(_rule_strategy, min_size=1, max_size=5)

_event_strategy = st.builds(
    StructuredEvent,
    event_id=st.builds(lambda: f"evt-{uuid.uuid4().hex[:8]}"),
    camera_id=_camera_id_strategy,
    tenant_id=st.just("tenant-test"),
    site_id=st.just("site-test"),
    timestamp=_timestamp_strategy,
    object_type=_object_type_strategy,
    track_id=st.builds(lambda: f"trk-{uuid.uuid4().hex[:8]}"),
    bounding_box=_bounding_box_strategy,
    confidence=_confidence_strategy,
    frame_crop=st.just("dGVzdA=="),  # base64 "test"
)


# ---------------------------------------------------------------------------
# Reference implementation for oracle comparison
# ---------------------------------------------------------------------------


def _oracle_evaluate(event: StructuredEvent, rules: list[Rule]) -> FilterResult:
    """Reference implementation of the Context Filter evaluation logic.

    This is the "oracle" that the property test compares against.
    It implements the same logic described in the design document:
    - A rule matches if ALL its non-None conditions are satisfied
    - If a matching rule has suppress_if and the suppress condition matches,
      the event is suppressed
    - The event passes if at least one rule matches without suppression
    """
    matched_ids: list[str] = []

    for rule in rules:
        if _rule_matches_oracle(event, rule):
            # Check suppress_if
            if rule.suppress_if is not None:
                if _suppress_matches_oracle(event, rule.suppress_if):
                    return FilterResult(
                        passed=False,
                        matched_rules=[rule.rule_id],
                        suppressed_reason="suppress_if",
                    )
            matched_ids.append(rule.rule_id)

    if matched_ids:
        return FilterResult(
            passed=True,
            matched_rules=matched_ids,
            suppressed_reason=None,
        )

    return FilterResult(
        passed=False,
        matched_rules=[],
        suppressed_reason="no_matching_rules",
    )


def _rule_matches_oracle(event: StructuredEvent, rule: Rule) -> bool:
    """Oracle: does the event match this rule's conditions?"""
    if rule.object_type is not None and event.object_type != rule.object_type:
        return False
    if rule.min_confidence is not None and event.confidence < rule.min_confidence:
        return False
    if rule.time_window is not None:
        if not time_in_window(event.timestamp, rule.time_window):
            return False
    if rule.zone is not None and rule.zone.polygon:
        cx = event.bounding_box.x + event.bounding_box.width / 2
        cy = event.bounding_box.y + event.bounding_box.height / 2
        if not point_in_polygon(cx, cy, rule.zone.polygon):
            return False
    return True


def _suppress_matches_oracle(
    event: StructuredEvent, suppress: SuppressCondition
) -> bool:
    """Oracle: does the suppress condition match?"""
    if suppress.object_type is not None and event.object_type != suppress.object_type:
        return False
    if suppress.time_window is not None:
        if not time_in_window(event.timestamp, suppress.time_window):
            return False
    return True


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestContextFilterRuleEvaluation:
    """Property 5: Context Filter Rule Evaluation Correctness.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.5**
    """

    @given(
        event=_event_strategy,
        rules=_rules_list_strategy,
    )
    @settings(max_examples=20)
    def test_filter_matches_oracle(
        self,
        event: StructuredEvent,
        rules: list[Rule],
    ) -> None:
        """For any random StructuredEvent and random RuleSet:

        1. Compute the expected result using the oracle.
        2. Create a RuleStore with the rules, create a ContextFilter.
        3. Evaluate the event through the ContextFilter.
        4. Assert the ContextFilter's ``passed`` matches the oracle's ``passed``.
        5. If passed, assert the matched rule IDs are the same.
        6. If suppressed, assert the suppression reason matches.
        """
        # 1. Oracle result
        expected = _oracle_evaluate(event, rules)

        # 2. Set up RuleStore and ContextFilter
        store = RuleStore(":memory:")
        try:
            camera_id = event.camera_id
            ruleset = RuleSet(
                version_id=f"rs-{uuid.uuid4().hex[:12]}",
                camera_id=camera_id,
                rules=rules,
                created_at=datetime.utcnow(),
            )
            store.save_ruleset(camera_id, ruleset)

            context_filter = ContextFilter(rule_store=store)

            # 3. Evaluate
            actual = context_filter.evaluate(event)

            # 4. passed must match
            assert actual.passed == expected.passed, (
                f"passed mismatch: expected {expected.passed}, got {actual.passed}. "
                f"Event: object_type={event.object_type}, confidence={event.confidence}, "
                f"time={event.timestamp.strftime('%H:%M')}. "
                f"Rules: {len(rules)} rules."
            )

            # 5. If passed, matched rule IDs should be the same set
            if actual.passed:
                assert set(actual.matched_rules) == set(expected.matched_rules), (
                    f"matched_rules mismatch: expected {expected.matched_rules}, "
                    f"got {actual.matched_rules}"
                )

            # 6. If suppressed via suppress_if, reason should match
            if expected.suppressed_reason == "suppress_if":
                assert actual.suppressed_reason == "suppress_if", (
                    f"Expected suppress_if suppression, got {actual.suppressed_reason}"
                )
        finally:
            store.close()

    @given(
        event=_event_strategy,
    )
    @settings(max_examples=20)
    def test_no_rules_means_not_passed(
        self,
        event: StructuredEvent,
    ) -> None:
        """When no ruleset is active for a camera, the event should not pass."""
        store = RuleStore(":memory:")
        try:
            context_filter = ContextFilter(rule_store=store)
            result = context_filter.evaluate(event)
            assert result.passed is False, (
                "Event should not pass when no ruleset is active"
            )
            assert result.suppressed_reason == "no_active_ruleset"
        finally:
            store.close()

    @given(
        event=_event_strategy,
    )
    @settings(max_examples=20)
    def test_empty_rules_means_not_passed(
        self,
        event: StructuredEvent,
    ) -> None:
        """When the active ruleset has zero rules, the event should not pass."""
        store = RuleStore(":memory:")
        try:
            camera_id = event.camera_id
            empty_ruleset = RuleSet(
                version_id=f"rs-{uuid.uuid4().hex[:12]}",
                camera_id=camera_id,
                rules=[],
                created_at=datetime.utcnow(),
            )
            store.save_ruleset(camera_id, empty_ruleset)

            context_filter = ContextFilter(rule_store=store)
            result = context_filter.evaluate(event)
            assert result.passed is False, (
                "Event should not pass when ruleset has no rules"
            )
        finally:
            store.close()
