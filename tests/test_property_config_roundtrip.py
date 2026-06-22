"""Property-based test for Configuration File Round-Trip.

# Feature: agentic-ai-cctv-monitoring, Property 14: Configuration File Round-Trip

**Validates: Requirements 18.1, 18.2, 18.3, 18.4**

For any valid SystemConfig object, serializing it to YAML and then deserializing
the YAML back SHALL produce a SystemConfig object equal to the original, with all
fields (VLM backend, API key, deployment profile, MQTT broker address/port, camera
URIs, thresholds, monitored classes) preserved.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.config_manager import ConfigManager
from agentic_cctv.models import (
    AlertConfig,
    BrokerConfig,
    CameraConfig,
    CooldownConfig,
    SecurityConfig,
    StorageConfig,
    SystemConfig,
    VLMConfig,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies for each nested config dataclass
# ---------------------------------------------------------------------------

# Reusable building blocks
_non_empty_str = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd"), whitelist_characters="-_./"),
    min_size=1,
    max_size=30,
)

_identifier_str = st.from_regex(r"[a-z][a-z0-9\-]{0,19}", fullmatch=True)


broker_config_strategy = st.builds(
    BrokerConfig,
    host=st.sampled_from(["localhost", "192.168.1.1", "broker.example.com"]),
    port=st.integers(min_value=1, max_value=65535),
    use_tls=st.booleans(),
    ca_cert=st.one_of(st.none(), _non_empty_str),
    client_cert=st.one_of(st.none(), _non_empty_str),
    client_key=st.one_of(st.none(), _non_empty_str),
    username=st.one_of(st.none(), _non_empty_str),
    password=st.one_of(st.none(), _non_empty_str),
)


vlm_config_strategy = st.builds(
    VLMConfig,
    backend=st.sampled_from(["cosmos", "gpt4o", "claude3", "gemini15"]),
    api_key=_non_empty_str,
    endpoint=st.one_of(st.none(), st.sampled_from([
        "https://integrate.api.nvidia.com/v1",
        "https://api.openai.com/v1",
    ])),
    timeout_seconds=st.integers(min_value=1, max_value=300),
)


camera_config_strategy = st.builds(
    CameraConfig,
    camera_id=_identifier_str,
    uri=st.from_regex(r"rtsp://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:[0-9]{1,5}/stream[0-9]", fullmatch=True),
    tenant_id=_identifier_str,
    site_id=_identifier_str,
    confidence_threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    monitored_classes=st.lists(
        st.sampled_from(["person", "vehicle", "animal", "bicycle", "fire"]),
        min_size=0,
        max_size=5,
    ),
    inference_runtime=st.sampled_from(["pytorch", "tensorrt"]),
    model_path=st.sampled_from(["./models/yolov8n.pt", "./models/yolov8s.pt"]),
    tracker_algorithm=st.sampled_from(["deepsort", "bytetrack"]),
    frame_skip=st.integers(min_value=1, max_value=30),
)


cooldown_config_strategy = st.builds(
    CooldownConfig,
    default_seconds=st.integers(min_value=1, max_value=3600),
    per_type_overrides=st.dictionaries(
        keys=st.sampled_from(["intrusion", "fire", "loitering", "vehicle"]),
        values=st.integers(min_value=1, max_value=3600),
        max_size=4,
    ),
)


storage_config_strategy = st.builds(
    StorageConfig,
    timeseries_db=st.sampled_from(["sqlite", "influxdb", "timescaledb"]),
    timeseries_path=st.sampled_from(["./data/events.db", "./data/ts.db"]),
    vector_db=st.sampled_from(["chromadb", "pinecone", "weaviate"]),
    vector_path=st.sampled_from(["./data/vectors", "./data/vdb"]),
    frame_crop_path=st.sampled_from(["./data/crops", "./data/frames"]),
    frame_crop_encryption_key=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=0,
        max_size=44,
    ),
    retention=st.fixed_dictionaries({
        "raw_events_days": st.integers(min_value=1, max_value=365),
        "aggregated_events_days": st.integers(min_value=1, max_value=730),
        "frame_crops_hours": st.integers(min_value=1, max_value=168),
    }),
)


alert_config_strategy = st.builds(
    AlertConfig,
    channels=st.lists(
        st.sampled_from(["push", "webhook", "sms", "in-app"]),
        min_size=1,
        max_size=4,
    ),
    webhook_url=st.one_of(st.none(), st.sampled_from([
        "https://hooks.example.com/cctv",
        "https://webhook.site/test",
    ])),
    cooldown=cooldown_config_strategy,
)


security_config_strategy = st.builds(
    SecurityConfig,
    tls_enabled=st.booleans(),
    oauth_enabled=st.booleans(),
    tenant_isolation=st.booleans(),
)


system_config_strategy = st.builds(
    SystemConfig,
    deployment_profile=st.sampled_from(["single-machine", "multi-machine", "edge-cloud-hybrid"]),
    mqtt=broker_config_strategy,
    vlm=vlm_config_strategy,
    cameras=st.lists(camera_config_strategy, min_size=0, max_size=3),
    storage=storage_config_strategy,
    alerts=alert_config_strategy,
    security=security_config_strategy,
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    """Property 14: Configuration File Round-Trip.

    **Validates: Requirements 18.1, 18.2, 18.3, 18.4**
    """

    @given(config=system_config_strategy)
    @settings(max_examples=20)
    def test_serialize_deserialize_roundtrip(self, config: SystemConfig) -> None:
        """For any valid SystemConfig, serialize to YAML then deserialize back
        and assert equality of all fields."""
        mgr = ConfigManager()

        # Serialize → YAML string
        yaml_str = mgr.to_yaml(config)

        # Deserialize ← YAML string
        restored = mgr.from_yaml(yaml_str)

        # Assert top-level field
        assert restored.deployment_profile == config.deployment_profile

        # Assert MQTT / BrokerConfig fields
        assert restored.mqtt.host == config.mqtt.host
        assert restored.mqtt.port == config.mqtt.port
        assert restored.mqtt.use_tls == config.mqtt.use_tls
        assert restored.mqtt.ca_cert == config.mqtt.ca_cert
        assert restored.mqtt.client_cert == config.mqtt.client_cert
        assert restored.mqtt.client_key == config.mqtt.client_key
        assert restored.mqtt.username == config.mqtt.username
        assert restored.mqtt.password == config.mqtt.password

        # Assert VLM config fields
        assert restored.vlm.backend == config.vlm.backend
        assert restored.vlm.api_key == config.vlm.api_key
        assert restored.vlm.endpoint == config.vlm.endpoint
        assert restored.vlm.timeout_seconds == config.vlm.timeout_seconds

        # Assert cameras list length and per-camera fields
        assert len(restored.cameras) == len(config.cameras)
        for orig_cam, rest_cam in zip(config.cameras, restored.cameras):
            assert rest_cam.camera_id == orig_cam.camera_id
            assert rest_cam.uri == orig_cam.uri
            assert rest_cam.tenant_id == orig_cam.tenant_id
            assert rest_cam.site_id == orig_cam.site_id
            assert rest_cam.confidence_threshold == orig_cam.confidence_threshold
            assert rest_cam.monitored_classes == orig_cam.monitored_classes
            assert rest_cam.inference_runtime == orig_cam.inference_runtime
            assert rest_cam.model_path == orig_cam.model_path
            assert rest_cam.tracker_algorithm == orig_cam.tracker_algorithm
            assert rest_cam.frame_skip == orig_cam.frame_skip

        # Assert storage config fields
        assert restored.storage.timeseries_db == config.storage.timeseries_db
        assert restored.storage.timeseries_path == config.storage.timeseries_path
        assert restored.storage.vector_db == config.storage.vector_db
        assert restored.storage.vector_path == config.storage.vector_path
        assert restored.storage.frame_crop_path == config.storage.frame_crop_path
        assert restored.storage.frame_crop_encryption_key == config.storage.frame_crop_encryption_key
        assert restored.storage.retention == config.storage.retention

        # Assert alert config fields
        assert restored.alerts.channels == config.alerts.channels
        assert restored.alerts.webhook_url == config.alerts.webhook_url
        assert restored.alerts.cooldown.default_seconds == config.alerts.cooldown.default_seconds
        assert restored.alerts.cooldown.per_type_overrides == config.alerts.cooldown.per_type_overrides

        # Assert security config fields
        assert restored.security.tls_enabled == config.security.tls_enabled
        assert restored.security.oauth_enabled == config.security.oauth_enabled
        assert restored.security.tenant_isolation == config.security.tenant_isolation
