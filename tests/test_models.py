"""Unit tests for shared data models."""

from __future__ import annotations

from datetime import datetime

from agentic_cctv.models import (
    ActionResult,
    AlertConfig,
    AlertPayload,
    AlertResult,
    BoundingBox,
    BrokerConfig,
    CameraConfig,
    CompiledRuleSet,
    CompoundCondition,
    CooldownConfig,
    Detection,
    DeviceHealth,
    FilterResult,
    Frame,
    HeartbeatMessage,
    IdentifiedObject,
    PromptScope,
    RawDetection,
    Rule,
    RuleSet,
    RuleSetVersion,
    SceneUnderstanding,
    SecurityConfig,
    StorageConfig,
    StructuredEvent,
    SuppressCondition,
    SystemConfig,
    TimeWindow,
    Track,
    VLMConfig,
    Zone,
)


class TestEdgeLayerModels:
    """Tests for Frame, BoundingBox, RawDetection, Detection, Track."""

    def test_frame_creation(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        frame = Frame(camera_id="cam-01", timestamp=ts, image=None, frame_number=42)
        assert frame.camera_id == "cam-01"
        assert frame.timestamp == ts
        assert frame.image is None
        assert frame.frame_number == 42

    def test_bounding_box_creation(self) -> None:
        bb = BoundingBox(x=10, y=20, width=100, height=200)
        assert bb.x == 10
        assert bb.y == 20
        assert bb.width == 100
        assert bb.height == 200

    def test_raw_detection_creation(self) -> None:
        bb = BoundingBox(x=0, y=0, width=50, height=50)
        raw = RawDetection(object_type="person", bounding_box=bb, confidence=0.95)
        assert raw.object_type == "person"
        assert raw.confidence == 0.95
        assert raw.bounding_box is bb

    def test_detection_with_gate_passed(self) -> None:
        bb = BoundingBox(x=0, y=0, width=50, height=50)
        det = Detection(object_type="vehicle", bounding_box=bb, confidence=0.8, passed_gate=True)
        assert det.passed_gate is True

    def test_detection_with_gate_failed(self) -> None:
        bb = BoundingBox(x=0, y=0, width=50, height=50)
        det = Detection(object_type="bird", bounding_box=bb, confidence=0.3, passed_gate=False)
        assert det.passed_gate is False

    def test_track_creation(self) -> None:
        bb = BoundingBox(x=100, y=200, width=60, height=120)
        track = Track(
            track_id="trk-abc123",
            object_type="person",
            bounding_box=bb,
            confidence=0.92,
            age=15,
            is_new=False,
        )
        assert track.track_id == "trk-abc123"
        assert track.age == 15
        assert track.is_new is False


class TestEventModels:
    """Tests for StructuredEvent and HeartbeatMessage."""

    def test_structured_event_all_fields(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        bb = BoundingBox(x=120, y=80, width=200, height=400)
        event = StructuredEvent(
            event_id="evt-001",
            camera_id="cam-lobby-01",
            tenant_id="tenant-acme",
            site_id="site-hq",
            timestamp=ts,
            object_type="person",
            track_id="trk-a1b2c3d4",
            bounding_box=bb,
            confidence=0.92,
            frame_crop="/9j/4AAQSkZJRg...",
        )
        assert event.event_id == "evt-001"
        assert event.camera_id == "cam-lobby-01"
        assert event.tenant_id == "tenant-acme"
        assert event.site_id == "site-hq"
        assert event.object_type == "person"
        assert event.track_id == "trk-a1b2c3d4"
        assert event.confidence == 0.92
        assert event.frame_crop.startswith("/9j/")

    def test_heartbeat_message_with_optional_fields(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        hb = HeartbeatMessage(
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=ts,
            cpu_percent=45.2,
            memory_percent=62.1,
            temperature_celsius=68.5,
            inference_latency_ms=35.2,
            gpu_utilization_percent=78.0,
        )
        assert hb.temperature_celsius == 68.5
        assert hb.gpu_utilization_percent == 78.0

    def test_heartbeat_message_without_optional_fields(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        hb = HeartbeatMessage(
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=ts,
            cpu_percent=45.2,
            memory_percent=62.1,
            temperature_celsius=None,
            inference_latency_ms=35.2,
            gpu_utilization_percent=None,
        )
        assert hb.temperature_celsius is None
        assert hb.gpu_utilization_percent is None


class TestVLMModels:
    """Tests for SceneUnderstanding and IdentifiedObject."""

    def test_scene_understanding_defaults(self) -> None:
        scene = SceneUnderstanding(
            event_id="evt-001",
            scene_description="A person walking",
            threat_level="low",
        )
        assert scene.objects_identified == []
        assert scene.recommended_action == "log"
        assert scene.confidence == 0.0
        assert scene.raw_response == {}
        assert scene.embedding is None

    def test_scene_understanding_full(self) -> None:
        obj = IdentifiedObject(type="person", action="running", location="parking lot")
        scene = SceneUnderstanding(
            event_id="evt-002",
            scene_description="Person running in parking lot",
            threat_level="medium",
            objects_identified=[obj],
            recommended_action="alert",
            confidence=0.85,
            raw_response={"raw": "data"},
            embedding=[0.1, 0.2, 0.3],
        )
        assert len(scene.objects_identified) == 1
        assert scene.objects_identified[0].type == "person"
        assert scene.embedding == [0.1, 0.2, 0.3]


class TestAlertModels:
    """Tests for AlertPayload, AlertResult, CooldownConfig."""

    def test_alert_payload_with_scene(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        scene = SceneUnderstanding(
            event_id="evt-001", scene_description="test", threat_level="high"
        )
        payload = AlertPayload(
            alert_id="a-001",
            event_id="evt-001",
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=ts,
            alert_type="intrusion",
            description="Intrusion detected",
            threat_level="high",
            frame_crop_url="https://example.com/crop.jpg",
            scene_understanding=scene,
        )
        assert payload.scene_understanding is not None
        assert payload.frame_crop_url is not None

    def test_alert_payload_without_optional(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        payload = AlertPayload(
            alert_id="a-002",
            event_id="evt-002",
            camera_id="cam-02",
            tenant_id="t1",
            site_id="s1",
            timestamp=ts,
            alert_type="fire",
            description="Fire detected",
            threat_level="critical",
            frame_crop_url=None,
        )
        assert payload.scene_understanding is None
        assert payload.frame_crop_url is None

    def test_alert_result_defaults(self) -> None:
        result = AlertResult(delivered=True)
        assert result.channels == []
        assert result.suppressed is False
        assert result.suppressed_count == 0

    def test_cooldown_config_defaults(self) -> None:
        config = CooldownConfig()
        assert config.default_seconds == 60
        assert config.per_type_overrides == {}

    def test_cooldown_config_custom(self) -> None:
        config = CooldownConfig(
            default_seconds=120,
            per_type_overrides={"intrusion": 30, "fire": 10},
        )
        assert config.default_seconds == 120
        assert config.per_type_overrides["fire"] == 10


class TestRuleModels:
    """Tests for Rule, RuleSet, FilterResult, and supporting types."""

    def test_rule_minimal(self) -> None:
        rule = Rule(rule_id="r-001")
        assert rule.object_type is None
        assert rule.min_confidence is None
        assert rule.time_window is None
        assert rule.zone is None
        assert rule.suppress_if is None
        assert rule.compound is None

    def test_rule_full(self) -> None:
        tw = TimeWindow(start="22:00", end="06:00")
        zone = Zone(polygon=[[0, 0], [640, 0], [640, 480], [0, 480]])
        suppress = SuppressCondition(object_type="vehicle", time_window=tw)
        rule = Rule(
            rule_id="r-002",
            object_type="person",
            min_confidence=0.8,
            time_window=tw,
            zone=zone,
            suppress_if=suppress,
        )
        assert rule.object_type == "person"
        assert rule.min_confidence == 0.8
        assert rule.time_window.start == "22:00"
        assert len(rule.zone.polygon) == 4

    def test_ruleset_default_factory(self) -> None:
        rs = RuleSet(version_id="v1", camera_id="cam-01")
        assert rs.rules == []
        assert isinstance(rs.created_at, datetime)

    def test_ruleset_with_rules(self) -> None:
        rule = Rule(rule_id="r-001", object_type="person")
        rs = RuleSet(version_id="v2", camera_id="cam-01", rules=[rule])
        assert len(rs.rules) == 1

    def test_filter_result_passed(self) -> None:
        fr = FilterResult(passed=True, matched_rules=["r-001", "r-002"])
        assert fr.passed is True
        assert len(fr.matched_rules) == 2
        assert fr.suppressed_reason is None

    def test_filter_result_suppressed(self) -> None:
        fr = FilterResult(passed=False, matched_rules=[], suppressed_reason="suppress_if")
        assert fr.passed is False
        assert fr.suppressed_reason == "suppress_if"

    def test_compound_condition(self) -> None:
        cc = CompoundCondition(operator="or", conditions=[{"type": "person"}])
        assert cc.operator == "or"
        assert len(cc.conditions) == 1

    def test_ruleset_version(self) -> None:
        ts = datetime(2025, 1, 15, 10, 0, 0)
        rsv = RuleSetVersion(version_id="v3", camera_id="cam-01", created_at=ts, is_active=True)
        assert rsv.is_active is True

    def test_compiled_ruleset(self) -> None:
        rs = RuleSet(version_id="v1", camera_id="cam-01")
        crs = CompiledRuleSet(
            ruleset=rs,
            original_prompt="detect people at night",
            explanation="Detects people between 22:00 and 06:00",
            confidence=0.95,
        )
        assert crs.original_prompt == "detect people at night"
        assert crs.confidence == 0.95


class TestConfigModels:
    """Tests for all configuration dataclasses."""

    def test_broker_config_defaults(self) -> None:
        cfg = BrokerConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 1883
        assert cfg.use_tls is False
        assert cfg.ca_cert is None
        assert cfg.username is None

    def test_camera_config_required_and_defaults(self) -> None:
        cfg = CameraConfig(
            camera_id="cam-01",
            uri="rtsp://192.168.1.100:554/stream1",
            tenant_id="t1",
            site_id="s1",
            confidence_threshold=0.7,
        )
        assert cfg.inference_runtime == "pytorch"
        assert cfg.tracker_algorithm == "deepsort"
        assert cfg.frame_skip == 3
        assert cfg.monitored_classes == []
        assert cfg.model_path == "./models/yolov8n.pt"

    def test_vlm_config_defaults(self) -> None:
        cfg = VLMConfig()
        assert cfg.backend == "cosmos"
        assert cfg.api_key == ""
        assert cfg.endpoint is None
        assert cfg.timeout_seconds == 30

    def test_storage_config_defaults(self) -> None:
        cfg = StorageConfig()
        assert cfg.timeseries_db == "sqlite"
        assert cfg.vector_db == "chromadb"
        assert cfg.retention["raw_events_days"] == 90
        assert cfg.retention["frame_crops_hours"] == 72

    def test_alert_config_defaults(self) -> None:
        cfg = AlertConfig()
        assert cfg.channels == ["push", "webhook"]
        assert cfg.webhook_url is None
        assert isinstance(cfg.cooldown, CooldownConfig)

    def test_security_config_defaults(self) -> None:
        cfg = SecurityConfig()
        assert cfg.tls_enabled is False
        assert cfg.oauth_enabled is False
        assert cfg.tenant_isolation is True

    def test_system_config_defaults(self) -> None:
        cfg = SystemConfig()
        assert cfg.deployment_profile == "single-machine"
        assert isinstance(cfg.mqtt, BrokerConfig)
        assert isinstance(cfg.vlm, VLMConfig)
        assert cfg.cameras == []
        assert isinstance(cfg.storage, StorageConfig)
        assert isinstance(cfg.alerts, AlertConfig)
        assert isinstance(cfg.security, SecurityConfig)

    def test_system_config_full(self) -> None:
        cam = CameraConfig(
            camera_id="cam-01",
            uri="rtsp://test",
            tenant_id="t1",
            site_id="s1",
            confidence_threshold=0.7,
            monitored_classes=["person", "vehicle"],
        )
        cfg = SystemConfig(
            deployment_profile="multi-machine",
            mqtt=BrokerConfig(host="broker.example.com", port=8883, use_tls=True),
            vlm=VLMConfig(backend="gpt4o", api_key="sk-test"),
            cameras=[cam],
        )
        assert cfg.deployment_profile == "multi-machine"
        assert cfg.mqtt.host == "broker.example.com"
        assert cfg.mqtt.use_tls is True
        assert cfg.vlm.backend == "gpt4o"
        assert len(cfg.cameras) == 1


class TestMiscModels:
    """Tests for remaining models: PromptScope, DeviceHealth, ActionResult."""

    def test_prompt_scope(self) -> None:
        ps = PromptScope(scope_type="site", target_ids=["site-hq"])
        assert ps.scope_type == "site"
        assert ps.target_ids == ["site-hq"]

    def test_prompt_scope_defaults(self) -> None:
        ps = PromptScope(scope_type="camera")
        assert ps.target_ids == []

    def test_device_health_online(self) -> None:
        dh = DeviceHealth(camera_id="cam-01", status="online")
        assert dh.last_heartbeat is None
        assert dh.metrics is None

    def test_device_health_with_metrics(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        hb = HeartbeatMessage(
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=ts,
            cpu_percent=45.0,
            memory_percent=60.0,
            temperature_celsius=None,
            inference_latency_ms=35.0,
            gpu_utilization_percent=None,
        )
        dh = DeviceHealth(
            camera_id="cam-01", status="online", last_heartbeat=ts, metrics=hb
        )
        assert dh.metrics is not None
        assert dh.metrics.cpu_percent == 45.0

    def test_action_result_defaults(self) -> None:
        ar = ActionResult(action="log")
        assert ar.alert_payload is None
        assert ar.context_update is None
        assert ar.cross_camera_refs == []

    def test_action_result_with_alert(self) -> None:
        ts = datetime(2025, 1, 15, 14, 30, 0)
        payload = AlertPayload(
            alert_id="a-001",
            event_id="evt-001",
            camera_id="cam-01",
            tenant_id="t1",
            site_id="s1",
            timestamp=ts,
            alert_type="intrusion",
            description="test",
            threat_level="high",
            frame_crop_url=None,
        )
        ar = ActionResult(action="alert", alert_payload=payload)
        assert ar.alert_payload is not None
        assert ar.alert_payload.alert_type == "intrusion"


class TestMutableDefaultIsolation:
    """Verify that mutable defaults are not shared between instances."""

    def test_ruleset_rules_not_shared(self) -> None:
        rs1 = RuleSet(version_id="v1", camera_id="cam-01")
        rs2 = RuleSet(version_id="v2", camera_id="cam-02")
        rs1.rules.append(Rule(rule_id="r-001"))
        assert len(rs2.rules) == 0

    def test_scene_understanding_objects_not_shared(self) -> None:
        s1 = SceneUnderstanding(event_id="e1", scene_description="a", threat_level="low")
        s2 = SceneUnderstanding(event_id="e2", scene_description="b", threat_level="low")
        s1.objects_identified.append(IdentifiedObject(type="x", action="y", location="z"))
        assert len(s2.objects_identified) == 0

    def test_system_config_cameras_not_shared(self) -> None:
        c1 = SystemConfig()
        c2 = SystemConfig()
        c1.cameras.append(
            CameraConfig(
                camera_id="cam-01",
                uri="test",
                tenant_id="t1",
                site_id="s1",
                confidence_threshold=0.5,
            )
        )
        assert len(c2.cameras) == 0

    def test_storage_config_retention_not_shared(self) -> None:
        s1 = StorageConfig()
        s2 = StorageConfig()
        s1.retention["custom_key"] = 999
        assert "custom_key" not in s2.retention
