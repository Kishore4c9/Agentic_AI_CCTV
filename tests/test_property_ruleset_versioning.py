"""Property-based test for Rule Set Version History and Rollback.

# Feature: agentic-ai-cctv-monitoring, Property 8: Rule Set Version History and Rollback

**Validates: Requirements 7.4**

For random sequences of RuleSet saves for a camera, the version history is
complete; rolling back to any version_id restores the exact RuleSet saved
with that version_id.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.models import (
    CompoundCondition,
    Rule,
    RuleSet,
    SuppressCondition,
    TimeWindow,
    Zone,
)
from agentic_cctv.rule_store import RuleStore

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_camera_id_strategy = st.from_regex(r"cam-[a-z0-9]{3,10}", fullmatch=True)

_rule_id_strategy = st.from_regex(r"rule-[a-z0-9]{3,8}", fullmatch=True)

_object_type_strategy = st.sampled_from(
    ["person", "vehicle", "animal", "package", "bicycle", "fire"]
)

_confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

_hh_mm_strategy = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59),
)

_time_window_strategy = st.builds(TimeWindow, start=_hh_mm_strategy, end=_hh_mm_strategy)

_zone_strategy = st.builds(
    Zone,
    polygon=st.lists(
        st.lists(st.integers(min_value=0, max_value=1920), min_size=2, max_size=2),
        min_size=3,
        max_size=8,
    ),
)

_suppress_condition_strategy = st.builds(
    SuppressCondition,
    object_type=st.one_of(st.none(), _object_type_strategy),
    time_window=st.one_of(st.none(), _time_window_strategy),
)

_rule_strategy = st.builds(
    Rule,
    rule_id=_rule_id_strategy,
    object_type=st.one_of(st.none(), _object_type_strategy),
    min_confidence=st.one_of(st.none(), _confidence_strategy),
    time_window=st.one_of(st.none(), _time_window_strategy),
    zone=st.one_of(st.none(), _zone_strategy),
    suppress_if=st.one_of(st.none(), _suppress_condition_strategy),
    compound=st.none(),  # keep compound simple for this property test
)

_rules_list_strategy = st.lists(_rule_strategy, min_size=0, max_size=5)


def _make_ruleset(camera_id: str, rules: list[Rule]) -> RuleSet:
    """Create a RuleSet with a unique version_id."""
    return RuleSet(
        version_id=f"rs-{uuid.uuid4().hex[:12]}",
        camera_id=camera_id,
        rules=rules,
        created_at=datetime.utcnow(),
    )


_ruleset_sequence_strategy = st.lists(
    _rules_list_strategy, min_size=1, max_size=10
)


# ---------------------------------------------------------------------------
# Helpers to compare rules (ignoring created_at / version_id)
# ---------------------------------------------------------------------------


def _rules_equal(a: list[Rule], b: list[Rule]) -> bool:
    """Check if two rule lists are structurally equal."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra.rule_id != rb.rule_id:
            return False
        if ra.object_type != rb.object_type:
            return False
        if ra.min_confidence != rb.min_confidence:
            return False
        # TimeWindow
        if (ra.time_window is None) != (rb.time_window is None):
            return False
        if ra.time_window and rb.time_window:
            if ra.time_window.start != rb.time_window.start:
                return False
            if ra.time_window.end != rb.time_window.end:
                return False
        # Zone
        if (ra.zone is None) != (rb.zone is None):
            return False
        if ra.zone and rb.zone:
            if ra.zone.polygon != rb.zone.polygon:
                return False
        # SuppressCondition
        if (ra.suppress_if is None) != (rb.suppress_if is None):
            return False
        if ra.suppress_if and rb.suppress_if:
            if ra.suppress_if.object_type != rb.suppress_if.object_type:
                return False
            if (ra.suppress_if.time_window is None) != (
                rb.suppress_if.time_window is None
            ):
                return False
            if ra.suppress_if.time_window and rb.suppress_if.time_window:
                if (
                    ra.suppress_if.time_window.start
                    != rb.suppress_if.time_window.start
                ):
                    return False
                if (
                    ra.suppress_if.time_window.end
                    != rb.suppress_if.time_window.end
                ):
                    return False
        # CompoundCondition
        if (ra.compound is None) != (rb.compound is None):
            return False
        if ra.compound and rb.compound:
            if ra.compound.operator != rb.compound.operator:
                return False
            if ra.compound.conditions != rb.compound.conditions:
                return False
    return True


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestRuleSetVersionHistoryAndRollback:
    """Property 8: Rule Set Version History and Rollback.

    **Validates: Requirements 7.4**
    """

    @given(
        camera_id=_camera_id_strategy,
        rule_sequences=_ruleset_sequence_strategy,
    )
    @settings(max_examples=20)
    def test_version_history_complete_and_rollback_restores(
        self,
        camera_id: str,
        rule_sequences: list[list[Rule]],
    ) -> None:
        """For any random sequence of RuleSet saves for a camera:

        1. Save each RuleSet and record the version_id.
        2. Assert the version history contains all saved version_ids.
        3. Assert only the last saved version is active.
        4. For each version_id in the history, rollback and verify the
           restored RuleSet has the exact same rules as the original.
        """
        store = RuleStore(":memory:")
        try:
            saved_versions: list[tuple[str, list[Rule]]] = []

            # 1. Save each RuleSet
            for rules in rule_sequences:
                ruleset = _make_ruleset(camera_id, rules)
                version_id = store.save_ruleset(camera_id, ruleset)
                saved_versions.append((version_id, rules))

            # 2. Version history contains all saved version_ids
            history = store.get_version_history(camera_id)
            history_ids = {v.version_id for v in history}
            for vid, _ in saved_versions:
                assert vid in history_ids, (
                    f"Version {vid} missing from history. "
                    f"History IDs: {history_ids}"
                )

            # 3. Only the last saved version is active
            active_versions = [v for v in history if v.is_active]
            assert len(active_versions) == 1, (
                f"Expected exactly 1 active version, got {len(active_versions)}"
            )
            last_vid = saved_versions[-1][0]
            assert active_versions[0].version_id == last_vid, (
                f"Active version should be {last_vid}, "
                f"got {active_versions[0].version_id}"
            )

            # 4. Rollback to each version and verify rules match
            for vid, original_rules in saved_versions:
                restored = store.rollback(camera_id, vid)
                assert _rules_equal(restored.rules, original_rules), (
                    f"Rollback to {vid}: rules do not match original. "
                    f"Original had {len(original_rules)} rules, "
                    f"restored has {len(restored.rules)} rules."
                )

                # After rollback, the restored version should be active
                active = store.get_active_ruleset(camera_id)
                assert active is not None, "No active ruleset after rollback"
                assert _rules_equal(active.rules, original_rules), (
                    f"Active ruleset after rollback to {vid} does not match"
                )
        finally:
            store.close()

    @given(
        camera_id=_camera_id_strategy,
        rule_sequences=_ruleset_sequence_strategy,
    )
    @settings(max_examples=20)
    def test_history_length_matches_saves_plus_rollbacks(
        self,
        camera_id: str,
        rule_sequences: list[list[Rule]],
    ) -> None:
        """The version history length equals the total number of save
        operations (initial saves + rollback-created saves)."""
        store = RuleStore(":memory:")
        try:
            total_saves = 0

            # Save all rulesets
            saved_versions: list[str] = []
            for rules in rule_sequences:
                ruleset = _make_ruleset(camera_id, rules)
                vid = store.save_ruleset(camera_id, ruleset)
                saved_versions.append(vid)
                total_saves += 1

            # Rollback to the first version (creates one more save)
            if saved_versions:
                store.rollback(camera_id, saved_versions[0])
                total_saves += 1

            history = store.get_version_history(camera_id)
            assert len(history) == total_saves, (
                f"History length {len(history)} != total saves {total_saves}"
            )
        finally:
            store.close()
