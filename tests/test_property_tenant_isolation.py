"""Property-based test for Multi-Tenant Data Isolation.

# Feature: agentic-ai-cctv-monitoring, Property 12: Multi-Tenant Data Isolation

**Validates: Requirements 10.6**

For random events across two distinct tenants, queries scoped to one tenant
never return data belonging to the other tenant.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentic_cctv.models import (
    AlertPayload,
    BoundingBox,
    HeartbeatMessage,
    Rule,
    RuleSet,
    StructuredEvent,
)
from agentic_cctv.frame_crop_store import FrameCropStore
from agentic_cctv.rule_store import RuleStore
from agentic_cctv.timeseries_db import TimeSeriesDB

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Two distinct alphanumeric tenant IDs
_tenant_id_strategy = st.from_regex(r"tenant-[a-z0-9]{3,8}", fullmatch=True)

_camera_id_strategy = st.from_regex(r"cam-[a-z0-9]{3,8}", fullmatch=True)

_site_id_strategy = st.from_regex(r"site-[a-z0-9]{3,6}", fullmatch=True)

_object_type_strategy = st.sampled_from(
    ["person", "vehicle", "animal", "package", "bicycle", "fire"]
)

_confidence_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False
)

_bounding_box_strategy = st.builds(
    BoundingBox,
    x=st.integers(min_value=0, max_value=1920),
    y=st.integers(min_value=0, max_value=1080),
    width=st.integers(min_value=1, max_value=200),
    height=st.integers(min_value=1, max_value=200),
)

_event_id_strategy = st.builds(lambda: f"evt-{uuid.uuid4().hex[:12]}")

_alert_id_strategy = st.builds(lambda: f"alert-{uuid.uuid4().hex[:12]}")

_track_id_strategy = st.builds(lambda: f"trk-{uuid.uuid4().hex[:8]}")

_timestamp_strategy = st.builds(
    lambda days, hours, minutes: datetime(2024, 1, 1)
    + timedelta(days=days, hours=hours, minutes=minutes),
    days=st.integers(min_value=0, max_value=30),
    hours=st.integers(min_value=0, max_value=23),
    minutes=st.integers(min_value=0, max_value=59),
)

_threat_level_strategy = st.sampled_from(
    ["none", "low", "medium", "high", "critical"]
)

_rule_id_strategy = st.from_regex(r"rule-[a-z0-9]{3,8}", fullmatch=True)

_rule_strategy = st.builds(
    Rule,
    rule_id=_rule_id_strategy,
    object_type=st.one_of(st.none(), _object_type_strategy),
    min_confidence=st.one_of(st.none(), _confidence_strategy),
    time_window=st.none(),
    zone=st.none(),
    suppress_if=st.none(),
    compound=st.none(),
)

_rules_list_strategy = st.lists(_rule_strategy, min_size=0, max_size=3)

_crop_bytes_strategy = st.binary(min_size=10, max_size=500)

# Valid 64-hex-char encryption key (32 bytes)
_encryption_key_hex = "a" * 64  # deterministic valid key for tests


# ---------------------------------------------------------------------------
# Helper: build StructuredEvent
# ---------------------------------------------------------------------------


def _make_event(
    tenant_id: str,
    camera_id: str,
    site_id: str,
    event_id: str,
    object_type: str,
    confidence: float,
    bbox: BoundingBox,
    track_id: str,
    timestamp: datetime,
) -> StructuredEvent:
    return StructuredEvent(
        event_id=event_id,
        camera_id=camera_id,
        tenant_id=tenant_id,
        site_id=site_id,
        timestamp=timestamp,
        object_type=object_type,
        track_id=track_id,
        bounding_box=bbox,
        confidence=confidence,
        frame_crop="",
    )


def _make_alert(
    tenant_id: str,
    camera_id: str,
    site_id: str,
    event_id: str,
    alert_id: str,
    threat_level: str,
    timestamp: datetime,
) -> AlertPayload:
    return AlertPayload(
        alert_id=alert_id,
        event_id=event_id,
        camera_id=camera_id,
        tenant_id=tenant_id,
        site_id=site_id,
        timestamp=timestamp,
        alert_type="intrusion",
        description="Test alert",
        threat_level=threat_level,
        frame_crop_url=None,
    )


def _make_heartbeat(
    tenant_id: str,
    camera_id: str,
    site_id: str,
    timestamp: datetime,
) -> HeartbeatMessage:
    return HeartbeatMessage(
        camera_id=camera_id,
        tenant_id=tenant_id,
        site_id=site_id,
        timestamp=timestamp,
        cpu_percent=50.0,
        memory_percent=60.0,
        temperature_celsius=45.0,
        inference_latency_ms=20.0,
        gpu_utilization_percent=70.0,
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestMultiTenantDataIsolation:
    """Property 12: Multi-Tenant Data Isolation.

    **Validates: Requirements 10.6**
    """

    @given(
        tenant_a=_tenant_id_strategy,
        tenant_b=_tenant_id_strategy,
        camera_id=_camera_id_strategy,
        site_id=_site_id_strategy,
        object_type=_object_type_strategy,
        confidence=_confidence_strategy,
        bbox=_bounding_box_strategy,
        track_id_a=_track_id_strategy,
        track_id_b=_track_id_strategy,
        timestamp=_timestamp_strategy,
        threat_level=_threat_level_strategy,
    )
    @settings(max_examples=10)
    def test_timeseries_db_isolation(
        self,
        tenant_a: str,
        tenant_b: str,
        camera_id: str,
        site_id: str,
        object_type: str,
        confidence: float,
        bbox: BoundingBox,
        track_id_a: str,
        track_id_b: str,
        timestamp: datetime,
        threat_level: str,
    ) -> None:
        """TimeSeriesDB: events, alerts, and heartbeats for tenant A are
        never returned when querying for tenant B, and vice versa."""
        assume(tenant_a != tenant_b)

        db = TimeSeriesDB(":memory:")
        try:
            event_id_a = f"evt-a-{uuid.uuid4().hex[:8]}"
            event_id_b = f"evt-b-{uuid.uuid4().hex[:8]}"

            event_a = _make_event(
                tenant_a, camera_id, site_id, event_id_a,
                object_type, confidence, bbox, track_id_a, timestamp,
            )
            event_b = _make_event(
                tenant_b, camera_id, site_id, event_id_b,
                object_type, confidence, bbox, track_id_b, timestamp,
            )

            db.insert_event(event_a)
            db.insert_event(event_b)

            # Events isolation
            events_a = db.get_events(tenant_id=tenant_a)
            events_b = db.get_events(tenant_id=tenant_b)

            event_ids_a = {e["event_id"] for e in events_a}
            event_ids_b = {e["event_id"] for e in events_b}

            assert event_id_a in event_ids_a, "Tenant A event missing from tenant A query"
            assert event_id_b not in event_ids_a, "Tenant B event leaked into tenant A query"
            assert event_id_b in event_ids_b, "Tenant B event missing from tenant B query"
            assert event_id_a not in event_ids_b, "Tenant A event leaked into tenant B query"

            # Alerts isolation
            alert_id_a = f"alert-a-{uuid.uuid4().hex[:8]}"
            alert_id_b = f"alert-b-{uuid.uuid4().hex[:8]}"

            alert_a = _make_alert(
                tenant_a, camera_id, site_id, event_id_a,
                alert_id_a, threat_level, timestamp,
            )
            alert_b = _make_alert(
                tenant_b, camera_id, site_id, event_id_b,
                alert_id_b, threat_level, timestamp,
            )

            db.insert_alert(alert_a, ["push"])
            db.insert_alert(alert_b, ["push"])

            alerts_a = db.get_alerts(tenant_id=tenant_a)
            alerts_b = db.get_alerts(tenant_id=tenant_b)

            alert_ids_a = {a["alert_id"] for a in alerts_a}
            alert_ids_b = {a["alert_id"] for a in alerts_b}

            assert alert_id_a in alert_ids_a, "Tenant A alert missing from tenant A query"
            assert alert_id_b not in alert_ids_a, "Tenant B alert leaked into tenant A query"
            assert alert_id_b in alert_ids_b, "Tenant B alert missing from tenant B query"
            assert alert_id_a not in alert_ids_b, "Tenant A alert leaked into tenant B query"

            # Heartbeats isolation
            hb_a = _make_heartbeat(tenant_a, camera_id, site_id, timestamp)
            hb_b = _make_heartbeat(tenant_b, camera_id, site_id, timestamp)

            db.insert_heartbeat(hb_a)
            db.insert_heartbeat(hb_b)

            hbs_a = db.get_heartbeats(tenant_id=tenant_a)
            hbs_b = db.get_heartbeats(tenant_id=tenant_b)

            tenant_ids_in_hbs_a = {h["tenant_id"] for h in hbs_a}
            tenant_ids_in_hbs_b = {h["tenant_id"] for h in hbs_b}

            assert tenant_ids_in_hbs_a == {tenant_a}, (
                f"Heartbeats for tenant A contain unexpected tenants: {tenant_ids_in_hbs_a}"
            )
            assert tenant_ids_in_hbs_b == {tenant_b}, (
                f"Heartbeats for tenant B contain unexpected tenants: {tenant_ids_in_hbs_b}"
            )
        finally:
            db.close()

    @given(
        tenant_a=_tenant_id_strategy,
        tenant_b=_tenant_id_strategy,
        camera_id=_camera_id_strategy,
        rules_a=_rules_list_strategy,
        rules_b=_rules_list_strategy,
    )
    @settings(max_examples=10)
    def test_rule_store_isolation(
        self,
        tenant_a: str,
        tenant_b: str,
        camera_id: str,
        rules_a: list[Rule],
        rules_b: list[Rule],
    ) -> None:
        """RuleStore: rulesets saved for tenant A are not visible when
        querying with tenant B, even for the same camera_id."""
        assume(tenant_a != tenant_b)

        store = RuleStore(":memory:")
        try:
            ruleset_a = RuleSet(
                version_id=f"rs-a-{uuid.uuid4().hex[:8]}",
                camera_id=camera_id,
                rules=rules_a,
                created_at=datetime.utcnow(),
            )
            ruleset_b = RuleSet(
                version_id=f"rs-b-{uuid.uuid4().hex[:8]}",
                camera_id=camera_id,
                rules=rules_b,
                created_at=datetime.utcnow(),
            )

            vid_a = store.save_ruleset(camera_id, ruleset_a, tenant_id=tenant_a)
            vid_b = store.save_ruleset(camera_id, ruleset_b, tenant_id=tenant_b)

            # get_active_ruleset scoped by tenant
            active_a = store.get_active_ruleset(camera_id, tenant_id=tenant_a)
            active_b = store.get_active_ruleset(camera_id, tenant_id=tenant_b)

            assert active_a is not None, "Tenant A should have an active ruleset"
            assert active_b is not None, "Tenant B should have an active ruleset"
            assert active_a.version_id == vid_a, (
                f"Tenant A active ruleset should be {vid_a}, got {active_a.version_id}"
            )
            assert active_b.version_id == vid_b, (
                f"Tenant B active ruleset should be {vid_b}, got {active_b.version_id}"
            )

            # get_version_history scoped by tenant
            history_a = store.get_version_history(camera_id, tenant_id=tenant_a)
            history_b = store.get_version_history(camera_id, tenant_id=tenant_b)

            history_ids_a = {v.version_id for v in history_a}
            history_ids_b = {v.version_id for v in history_b}

            assert vid_a in history_ids_a, "Tenant A version missing from tenant A history"
            assert vid_b not in history_ids_a, "Tenant B version leaked into tenant A history"
            assert vid_b in history_ids_b, "Tenant B version missing from tenant B history"
            assert vid_a not in history_ids_b, "Tenant A version leaked into tenant B history"
        finally:
            store.close()

    @given(
        tenant_a=_tenant_id_strategy,
        tenant_b=_tenant_id_strategy,
        crop_a=_crop_bytes_strategy,
        crop_b=_crop_bytes_strategy,
        camera_id=_camera_id_strategy,
    )
    @settings(max_examples=10)
    def test_frame_crop_store_isolation(
        self,
        tenant_a: str,
        tenant_b: str,
        crop_a: bytes,
        crop_b: bytes,
        camera_id: str,
    ) -> None:
        """FrameCropStore: crops stored for tenant A are not accessible
        when querying with tenant B's tenant_id."""
        assume(tenant_a != tenant_b)

        tmp_dir = tempfile.mkdtemp()
        try:
            store = FrameCropStore(
                crop_path=tmp_dir,
                encryption_key_hex=_encryption_key_hex,
            )

            event_id_a = f"evt-a-{uuid.uuid4().hex[:8]}"
            event_id_b = f"evt-b-{uuid.uuid4().hex[:8]}"

            store.store_crop(event_id_a, tenant_a, camera_id, crop_a)
            store.store_crop(event_id_b, tenant_b, camera_id, crop_b)

            # get_crop_by_event_id with tenant scoping
            # Tenant A can access their own crop
            result_a_own = store.get_crop_by_event_id(event_id_a, tenant_id=tenant_a)
            assert result_a_own is not None, "Tenant A should access their own crop"
            assert result_a_own == crop_a, "Tenant A crop data mismatch"

            # Tenant A cannot access tenant B's crop
            result_a_cross = store.get_crop_by_event_id(event_id_b, tenant_id=tenant_a)
            assert result_a_cross is None, (
                "Tenant A should NOT be able to access tenant B's crop"
            )

            # Tenant B can access their own crop
            result_b_own = store.get_crop_by_event_id(event_id_b, tenant_id=tenant_b)
            assert result_b_own is not None, "Tenant B should access their own crop"
            assert result_b_own == crop_b, "Tenant B crop data mismatch"

            # Tenant B cannot access tenant A's crop
            result_b_cross = store.get_crop_by_event_id(event_id_a, tenant_id=tenant_b)
            assert result_b_cross is None, (
                "Tenant B should NOT be able to access tenant A's crop"
            )

            # get_crops_by_tenant returns only that tenant's crops
            crops_a = store.get_crops_by_tenant(tenant_a)
            crops_b = store.get_crops_by_tenant(tenant_b)

            crop_event_ids_a = {c["event_id"] for c in crops_a}
            crop_event_ids_b = {c["event_id"] for c in crops_b}

            assert event_id_a in crop_event_ids_a, "Tenant A crop missing from tenant A list"
            assert event_id_b not in crop_event_ids_a, "Tenant B crop leaked into tenant A list"
            assert event_id_b in crop_event_ids_b, "Tenant B crop missing from tenant B list"
            assert event_id_a not in crop_event_ids_b, "Tenant A crop leaked into tenant B list"

            store.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @given(
        tenant_a=_tenant_id_strategy,
        tenant_b=_tenant_id_strategy,
        camera_id=_camera_id_strategy,
    )
    @settings(max_examples=10)
    def test_vector_db_isolation(
        self,
        tenant_a: str,
        tenant_b: str,
        camera_id: str,
    ) -> None:
        """VectorDB: embeddings stored for tenant A are not returned
        when searching with tenant B's tenant_id."""
        chromadb = pytest.importorskip("chromadb")
        assume(tenant_a != tenant_b)

        from agentic_cctv.vector_db import VectorDB

        tmp_dir = tempfile.mkdtemp()
        try:
            vdb = VectorDB(tmp_dir)

            event_id_a = f"evt-a-{uuid.uuid4().hex[:8]}"
            event_id_b = f"evt-b-{uuid.uuid4().hex[:8]}"

            # Use simple deterministic embeddings that differ
            embedding_a = [1.0, 0.0, 0.0]
            embedding_b = [0.0, 1.0, 0.0]

            vdb.store_embedding(
                event_id=event_id_a,
                embedding=embedding_a,
                metadata={"camera_id": camera_id},
                tenant_id=tenant_a,
            )
            vdb.store_embedding(
                event_id=event_id_b,
                embedding=embedding_b,
                metadata={"camera_id": camera_id},
                tenant_id=tenant_b,
            )

            # Search scoped to tenant A
            results_a = vdb.search(
                query_embedding=embedding_a,
                n_results=10,
                tenant_id=tenant_a,
            )
            result_ids_a = {r["id"] for r in results_a}
            assert event_id_a in result_ids_a, "Tenant A embedding missing from tenant A search"
            assert event_id_b not in result_ids_a, "Tenant B embedding leaked into tenant A search"

            # Search scoped to tenant B
            results_b = vdb.search(
                query_embedding=embedding_b,
                n_results=10,
                tenant_id=tenant_b,
            )
            result_ids_b = {r["id"] for r in results_b}
            assert event_id_b in result_ids_b, "Tenant B embedding missing from tenant B search"
            assert event_id_a not in result_ids_b, "Tenant A embedding leaked into tenant B search"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
