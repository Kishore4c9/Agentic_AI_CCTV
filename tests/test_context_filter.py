"""Unit tests for ContextFilter and RuleStore.

Tests specific rule combinations: time windows, zones (point-in-polygon edge
cases), suppress_if, compound conditions, hot-reload timing, and RuleStore
CRUD operations.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 7.3, 7.4
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime

import pytest

from agentic_cctv.context_filter import (
    ContextFilter,
    point_in_polygon,
    time_in_window,
)
from agentic_cctv.models import (
    BoundingBox,
    CompoundCondition,
    Rule,
    RuleSet,
    StructuredEvent,
    SuppressCondition,
    TimeWindow,
    Zone,
)
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rule_store() -> RuleStore:
    store = RuleStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def tsdb() -> TimeSeriesDB:
    db = TimeSeriesDB(":memory:")
    yield db
    db.close()


def _make_event(
    camera_id: str = "cam-lobby-01",
    object_type: str = "person",
    confidence: float = 0.9,
    hour: int = 23,
    minute: int = 30,
    bbox_x: int = 320,
    bbox_y: int = 240,
    bbox_w: int = 100,
    bbox_h: int = 200,
) -> StructuredEvent:
    return StructuredEvent(
        event_id=f"evt-{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        tenant_id="tenant-test",
        site_id="site-test",
        timestamp=datetime(2025, 6, 15, hour, minute, 0),
        object_type=object_type,
        track_id=f"trk-{uuid.uuid4().hex[:8]}",
        bounding_box=BoundingBox(x=bbox_x, y=bbox_y, width=bbox_w, height=bbox_h),
        confidence=confidence,
        frame_crop="dGVzdA==",
    )


def _make_ruleset(
    camera_id: str, rules: list[Rule], version_id: str | None = None
) -> RuleSet:
    return RuleSet(
        version_id=version_id or f"rs-{uuid.uuid4().hex[:12]}",
        camera_id=camera_id,
        rules=rules,
        created_at=datetime.utcnow(),
    )


# ===========================================================================
# RuleStore unit tests
# ===========================================================================


class TestRuleStore:
    """Unit tests for RuleStore CRUD operations."""

    def test_save_and_get_active(self, rule_store: RuleStore) -> None:
        """Save a ruleset and retrieve it as the active one."""
        rules = [Rule(rule_id="r1", object_type="person", min_confidence=0.8)]
        rs = _make_ruleset("cam-01", rules)
        vid = rule_store.save_ruleset("cam-01", rs)

        active = rule_store.get_active_ruleset("cam-01")
        assert active is not None
        assert active.version_id == vid
        assert len(active.rules) == 1
        assert active.rules[0].rule_id == "r1"
        assert active.rules[0].object_type == "person"
        assert active.rules[0].min_confidence == 0.8

    def test_no_active_returns_none(self, rule_store: RuleStore) -> None:
        """get_active_ruleset returns None when no ruleset exists."""
        assert rule_store.get_active_ruleset("cam-nonexistent") is None

    def test_save_deactivates_previous(self, rule_store: RuleStore) -> None:
        """Saving a new ruleset deactivates the previous one."""
        rs1 = _make_ruleset("cam-01", [Rule(rule_id="r1")])
        vid1 = rule_store.save_ruleset("cam-01", rs1)

        rs2 = _make_ruleset("cam-01", [Rule(rule_id="r2")])
        vid2 = rule_store.save_ruleset("cam-01", rs2)

        active = rule_store.get_active_ruleset("cam-01")
        assert active is not None
        assert active.version_id == vid2
        assert active.rules[0].rule_id == "r2"

    def test_version_history(self, rule_store: RuleStore) -> None:
        """Version history contains all saved versions in order."""
        vids = []
        for i in range(3):
            rs = _make_ruleset("cam-01", [Rule(rule_id=f"r{i}")])
            vid = rule_store.save_ruleset("cam-01", rs)
            vids.append(vid)

        history = rule_store.get_version_history("cam-01")
        assert len(history) == 3
        history_ids = [v.version_id for v in history]
        assert history_ids == vids

        # Only the last should be active
        active_count = sum(1 for v in history if v.is_active)
        assert active_count == 1
        assert history[-1].is_active is True

    def test_rollback_restores_rules(self, rule_store: RuleStore) -> None:
        """Rollback restores the rules from a previous version."""
        rs1 = _make_ruleset(
            "cam-01",
            [Rule(rule_id="r1", object_type="person", min_confidence=0.5)],
        )
        vid1 = rule_store.save_ruleset("cam-01", rs1)

        rs2 = _make_ruleset(
            "cam-01",
            [Rule(rule_id="r2", object_type="vehicle", min_confidence=0.9)],
        )
        rule_store.save_ruleset("cam-01", rs2)

        # Rollback to version 1
        restored = rule_store.rollback("cam-01", vid1)
        assert len(restored.rules) == 1
        assert restored.rules[0].rule_id == "r1"
        assert restored.rules[0].object_type == "person"
        assert restored.rules[0].min_confidence == 0.5

        # Active should now be the restored version
        active = rule_store.get_active_ruleset("cam-01")
        assert active is not None
        assert active.rules[0].rule_id == "r1"

    def test_rollback_nonexistent_version_raises(
        self, rule_store: RuleStore
    ) -> None:
        """Rollback to a nonexistent version raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            rule_store.rollback("cam-01", "rs-nonexistent")

    def test_rollback_wrong_camera_raises(self, rule_store: RuleStore) -> None:
        """Rollback with wrong camera_id raises ValueError."""
        rs = _make_ruleset("cam-01", [Rule(rule_id="r1")])
        vid = rule_store.save_ruleset("cam-01", rs)

        with pytest.raises(ValueError, match="belongs to camera"):
            rule_store.rollback("cam-02", vid)

    def test_save_with_complex_rules(self, rule_store: RuleStore) -> None:
        """Save and retrieve rules with time_window, zone, and suppress_if."""
        rules = [
            Rule(
                rule_id="r1",
                object_type="person",
                min_confidence=0.8,
                time_window=TimeWindow(start="22:00", end="06:00"),
                zone=Zone(polygon=[[0, 0], [640, 0], [640, 480], [0, 480]]),
                suppress_if=SuppressCondition(
                    object_type="person",
                    time_window=TimeWindow(start="08:00", end="18:00"),
                ),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        active = rule_store.get_active_ruleset("cam-01")
        assert active is not None
        r = active.rules[0]
        assert r.time_window is not None
        assert r.time_window.start == "22:00"
        assert r.time_window.end == "06:00"
        assert r.zone is not None
        assert len(r.zone.polygon) == 4
        assert r.suppress_if is not None
        assert r.suppress_if.object_type == "person"
        assert r.suppress_if.time_window is not None
        assert r.suppress_if.time_window.start == "08:00"

    def test_get_ruleset_by_version(self, rule_store: RuleStore) -> None:
        """get_ruleset_by_version returns the correct ruleset."""
        rs = _make_ruleset("cam-01", [Rule(rule_id="r1", object_type="vehicle")])
        vid = rule_store.save_ruleset("cam-01", rs)

        result = rule_store.get_ruleset_by_version(vid)
        assert result is not None
        assert result.version_id == vid
        assert result.rules[0].object_type == "vehicle"

    def test_get_ruleset_by_version_not_found(
        self, rule_store: RuleStore
    ) -> None:
        """get_ruleset_by_version returns None for unknown version."""
        assert rule_store.get_ruleset_by_version("rs-unknown") is None

    def test_multiple_cameras_isolated(self, rule_store: RuleStore) -> None:
        """Rulesets for different cameras are isolated."""
        rs1 = _make_ruleset("cam-01", [Rule(rule_id="r1", object_type="person")])
        rs2 = _make_ruleset("cam-02", [Rule(rule_id="r2", object_type="vehicle")])
        rule_store.save_ruleset("cam-01", rs1)
        rule_store.save_ruleset("cam-02", rs2)

        active1 = rule_store.get_active_ruleset("cam-01")
        active2 = rule_store.get_active_ruleset("cam-02")
        assert active1 is not None
        assert active2 is not None
        assert active1.rules[0].object_type == "person"
        assert active2.rules[0].object_type == "vehicle"


# ===========================================================================
# Point-in-polygon unit tests
# ===========================================================================


class TestPointInPolygon:
    """Unit tests for the point_in_polygon function."""

    def test_point_inside_rectangle(self) -> None:
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon(50, 50, polygon) is True

    def test_point_outside_rectangle(self) -> None:
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon(150, 50, polygon) is False

    def test_point_on_edge(self) -> None:
        """Points on edges may or may not be inside depending on ray casting."""
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Just inside
        assert point_in_polygon(1, 1, polygon) is True

    def test_triangle(self) -> None:
        polygon = [[0, 0], [200, 0], [100, 200]]
        assert point_in_polygon(100, 50, polygon) is True
        assert point_in_polygon(0, 200, polygon) is False

    def test_fewer_than_3_vertices(self) -> None:
        assert point_in_polygon(50, 50, [[0, 0], [100, 0]]) is False
        assert point_in_polygon(50, 50, []) is False

    def test_concave_polygon(self) -> None:
        """L-shaped polygon."""
        polygon = [[0, 0], [200, 0], [200, 100], [100, 100], [100, 200], [0, 200]]
        # Inside the L
        assert point_in_polygon(50, 50, polygon) is True
        assert point_in_polygon(50, 150, polygon) is True
        # Outside the L (in the concave notch)
        assert point_in_polygon(150, 150, polygon) is False


# ===========================================================================
# Time window unit tests
# ===========================================================================


class TestTimeInWindow:
    """Unit tests for the time_in_window function."""

    def test_normal_window_inside(self) -> None:
        """Time within a normal (non-wrapping) window."""
        tw = TimeWindow(start="08:00", end="18:00")
        assert time_in_window(datetime(2025, 1, 1, 12, 0), tw) is True

    def test_normal_window_outside(self) -> None:
        tw = TimeWindow(start="08:00", end="18:00")
        assert time_in_window(datetime(2025, 1, 1, 20, 0), tw) is False

    def test_normal_window_at_start(self) -> None:
        tw = TimeWindow(start="08:00", end="18:00")
        assert time_in_window(datetime(2025, 1, 1, 8, 0), tw) is True

    def test_normal_window_at_end(self) -> None:
        tw = TimeWindow(start="08:00", end="18:00")
        assert time_in_window(datetime(2025, 1, 1, 18, 0), tw) is True

    def test_wrapping_window_late_night(self) -> None:
        """22:00 → 06:00 wrapping window, event at 23:30."""
        tw = TimeWindow(start="22:00", end="06:00")
        assert time_in_window(datetime(2025, 1, 1, 23, 30), tw) is True

    def test_wrapping_window_early_morning(self) -> None:
        """22:00 → 06:00 wrapping window, event at 03:00."""
        tw = TimeWindow(start="22:00", end="06:00")
        assert time_in_window(datetime(2025, 1, 1, 3, 0), tw) is True

    def test_wrapping_window_outside(self) -> None:
        """22:00 → 06:00 wrapping window, event at 12:00."""
        tw = TimeWindow(start="22:00", end="06:00")
        assert time_in_window(datetime(2025, 1, 1, 12, 0), tw) is False

    def test_wrapping_window_at_start(self) -> None:
        tw = TimeWindow(start="22:00", end="06:00")
        assert time_in_window(datetime(2025, 1, 1, 22, 0), tw) is True

    def test_wrapping_window_at_end(self) -> None:
        tw = TimeWindow(start="22:00", end="06:00")
        assert time_in_window(datetime(2025, 1, 1, 6, 0), tw) is True

    def test_same_start_end(self) -> None:
        """When start == end, only that exact minute matches."""
        tw = TimeWindow(start="12:00", end="12:00")
        assert time_in_window(datetime(2025, 1, 1, 12, 0), tw) is True
        assert time_in_window(datetime(2025, 1, 1, 12, 1), tw) is False


# ===========================================================================
# ContextFilter unit tests
# ===========================================================================


class TestContextFilter:
    """Unit tests for ContextFilter evaluation logic."""

    def test_object_type_match(self, rule_store: RuleStore) -> None:
        """Event passes when object_type matches a rule."""
        rules = [Rule(rule_id="r1", object_type="person")]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)
        event = _make_event(camera_id="cam-01", object_type="person")
        result = cf.evaluate(event)
        assert result.passed is True
        assert "r1" in result.matched_rules

    def test_object_type_mismatch(self, rule_store: RuleStore) -> None:
        """Event does not pass when object_type doesn't match."""
        rules = [Rule(rule_id="r1", object_type="vehicle")]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)
        event = _make_event(camera_id="cam-01", object_type="person")
        result = cf.evaluate(event)
        assert result.passed is False

    def test_confidence_threshold(self, rule_store: RuleStore) -> None:
        """Event passes when confidence >= min_confidence."""
        rules = [Rule(rule_id="r1", min_confidence=0.8)]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        # Above threshold
        event_high = _make_event(camera_id="cam-01", confidence=0.9)
        assert cf.evaluate(event_high).passed is True

        # Exactly at threshold
        event_exact = _make_event(camera_id="cam-01", confidence=0.8)
        assert cf.evaluate(event_exact).passed is True

        # Below threshold
        event_low = _make_event(camera_id="cam-01", confidence=0.7)
        assert cf.evaluate(event_low).passed is False

    def test_time_window_match(self, rule_store: RuleStore) -> None:
        """Event passes when timestamp is within the time window."""
        rules = [
            Rule(
                rule_id="r1",
                time_window=TimeWindow(start="22:00", end="06:00"),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        # Inside window (23:30)
        event_in = _make_event(camera_id="cam-01", hour=23, minute=30)
        assert cf.evaluate(event_in).passed is True

        # Outside window (12:00)
        event_out = _make_event(camera_id="cam-01", hour=12, minute=0)
        assert cf.evaluate(event_out).passed is False

    def test_zone_match(self, rule_store: RuleStore) -> None:
        """Event passes when bounding box center is inside the zone polygon."""
        rules = [
            Rule(
                rule_id="r1",
                zone=Zone(polygon=[[0, 0], [640, 0], [640, 480], [0, 480]]),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        # Center at (370, 340) — inside
        event_in = _make_event(
            camera_id="cam-01", bbox_x=320, bbox_y=240, bbox_w=100, bbox_h=200
        )
        assert cf.evaluate(event_in).passed is True

        # Center at (1050, 550) — outside
        event_out = _make_event(
            camera_id="cam-01", bbox_x=1000, bbox_y=500, bbox_w=100, bbox_h=100
        )
        assert cf.evaluate(event_out).passed is False

    def test_suppress_if_blocks_matching_rule(
        self, rule_store: RuleStore
    ) -> None:
        """A matching rule with a matching suppress_if suppresses the event."""
        rules = [
            Rule(
                rule_id="r1",
                object_type="vehicle",
                suppress_if=SuppressCondition(
                    object_type="vehicle",
                    time_window=TimeWindow(start="08:00", end="18:00"),
                ),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        # Vehicle during business hours — suppressed
        event_suppressed = _make_event(
            camera_id="cam-01", object_type="vehicle", hour=12, minute=0
        )
        result = cf.evaluate(event_suppressed)
        assert result.passed is False
        assert result.suppressed_reason == "suppress_if"

        # Vehicle at night — not suppressed
        event_allowed = _make_event(
            camera_id="cam-01", object_type="vehicle", hour=23, minute=0
        )
        result = cf.evaluate(event_allowed)
        assert result.passed is True

    def test_suppress_if_different_object_type(
        self, rule_store: RuleStore
    ) -> None:
        """suppress_if with different object_type does not suppress."""
        rules = [
            Rule(
                rule_id="r1",
                object_type="person",
                suppress_if=SuppressCondition(object_type="vehicle"),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)
        event = _make_event(camera_id="cam-01", object_type="person")
        result = cf.evaluate(event)
        assert result.passed is True

    def test_compound_and_condition(self, rule_store: RuleStore) -> None:
        """Compound AND condition requires all sub-conditions to match."""
        rules = [
            Rule(
                rule_id="r1",
                compound=CompoundCondition(
                    operator="and",
                    conditions=[
                        {"object_type": "person"},
                        {"min_confidence": 0.8},
                    ],
                ),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        # Both match
        event_both = _make_event(
            camera_id="cam-01", object_type="person", confidence=0.9
        )
        assert cf.evaluate(event_both).passed is True

        # Only object_type matches
        event_low_conf = _make_event(
            camera_id="cam-01", object_type="person", confidence=0.5
        )
        assert cf.evaluate(event_low_conf).passed is False

        # Only confidence matches
        event_wrong_type = _make_event(
            camera_id="cam-01", object_type="vehicle", confidence=0.9
        )
        assert cf.evaluate(event_wrong_type).passed is False

    def test_compound_or_condition(self, rule_store: RuleStore) -> None:
        """Compound OR condition requires at least one sub-condition to match."""
        rules = [
            Rule(
                rule_id="r1",
                compound=CompoundCondition(
                    operator="or",
                    conditions=[
                        {"object_type": "person"},
                        {"object_type": "vehicle"},
                    ],
                ),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        # Person matches
        event_person = _make_event(camera_id="cam-01", object_type="person")
        assert cf.evaluate(event_person).passed is True

        # Vehicle matches
        event_vehicle = _make_event(camera_id="cam-01", object_type="vehicle")
        assert cf.evaluate(event_vehicle).passed is True

        # Animal doesn't match either
        event_animal = _make_event(camera_id="cam-01", object_type="animal")
        assert cf.evaluate(event_animal).passed is False

    def test_multiple_rules_any_match(self, rule_store: RuleStore) -> None:
        """Event passes if any rule matches."""
        rules = [
            Rule(rule_id="r1", object_type="person"),
            Rule(rule_id="r2", object_type="vehicle"),
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        event = _make_event(camera_id="cam-01", object_type="vehicle")
        result = cf.evaluate(event)
        assert result.passed is True
        assert "r2" in result.matched_rules

    def test_wildcard_rule_no_conditions(self, rule_store: RuleStore) -> None:
        """A rule with no conditions matches everything."""
        rules = [Rule(rule_id="r1")]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)
        event = _make_event(camera_id="cam-01", object_type="anything")
        result = cf.evaluate(event)
        assert result.passed is True

    def test_hot_reload(self, rule_store: RuleStore) -> None:
        """reload_rules clears the cache so new rules take effect."""
        rules_v1 = [Rule(rule_id="r1", object_type="person")]
        rs1 = _make_ruleset("cam-01", rules_v1)
        rule_store.save_ruleset("cam-01", rs1)

        cf = ContextFilter(rule_store=rule_store)

        # First evaluation caches the rules
        event_person = _make_event(camera_id="cam-01", object_type="person")
        assert cf.evaluate(event_person).passed is True

        event_vehicle = _make_event(camera_id="cam-01", object_type="vehicle")
        assert cf.evaluate(event_vehicle).passed is False

        # Update rules to match vehicles instead
        rules_v2 = [Rule(rule_id="r2", object_type="vehicle")]
        rs2 = _make_ruleset("cam-01", rules_v2)
        rule_store.save_ruleset("cam-01", rs2)

        # Force reload
        cf.reload_rules("cam-01")

        # Now vehicle should pass and person should not
        assert cf.evaluate(event_vehicle).passed is True
        assert cf.evaluate(event_person).passed is False

    def test_suppressed_event_logged_to_tsdb(
        self, rule_store: RuleStore, tsdb: TimeSeriesDB
    ) -> None:
        """Suppressed events are logged to the Time Series DB."""
        rules = [Rule(rule_id="r1", object_type="vehicle")]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store, timeseries_db=tsdb)

        # Person event won't match vehicle rule — suppressed
        event = _make_event(camera_id="cam-01", object_type="person")
        result = cf.evaluate(event)
        assert result.passed is False

        # Check it was logged to TSDB
        events = tsdb.get_events(camera_id="cam-01")
        assert len(events) == 1
        assert events[0]["event_id"] == event.event_id
        assert events[0]["context_gate_passed"] == 0  # False

    def test_no_active_ruleset_suppresses(self, rule_store: RuleStore) -> None:
        """Events are suppressed when no ruleset is active."""
        cf = ContextFilter(rule_store=rule_store)
        event = _make_event(camera_id="cam-01")
        result = cf.evaluate(event)
        assert result.passed is False
        assert result.suppressed_reason == "no_active_ruleset"

    def test_combined_conditions(self, rule_store: RuleStore) -> None:
        """Rule with multiple conditions: all must match."""
        rules = [
            Rule(
                rule_id="r1",
                object_type="person",
                min_confidence=0.8,
                time_window=TimeWindow(start="22:00", end="06:00"),
            )
        ]
        rs = _make_ruleset("cam-01", rules)
        rule_store.save_ruleset("cam-01", rs)

        cf = ContextFilter(rule_store=rule_store)

        # All conditions match
        event_match = _make_event(
            camera_id="cam-01",
            object_type="person",
            confidence=0.9,
            hour=23,
            minute=30,
        )
        assert cf.evaluate(event_match).passed is True

        # Wrong time
        event_wrong_time = _make_event(
            camera_id="cam-01",
            object_type="person",
            confidence=0.9,
            hour=12,
            minute=0,
        )
        assert cf.evaluate(event_wrong_time).passed is False

        # Low confidence
        event_low_conf = _make_event(
            camera_id="cam-01",
            object_type="person",
            confidence=0.5,
            hour=23,
            minute=30,
        )
        assert cf.evaluate(event_low_conf).passed is False

        # Wrong object type
        event_wrong_type = _make_event(
            camera_id="cam-01",
            object_type="vehicle",
            confidence=0.9,
            hour=23,
            minute=30,
        )
        assert cf.evaluate(event_wrong_type).passed is False
