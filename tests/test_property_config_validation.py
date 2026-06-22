"""Property-based test for Configuration Validation Error Reporting.

# Feature: agentic-ai-cctv-monitoring, Property 15: Configuration Validation Error Reporting

**Validates: Requirements 18.5**

For any valid SystemConfig with exactly one required field removed or set to an
invalid value, the configuration validator SHALL produce at least one
ValidationError whose field_path identifies the invalid field.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Tuple

from hypothesis import given, settings, assume
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
# Reuse Hypothesis strategies from test_property_config_roundtrip.py
# ---------------------------------------------------------------------------

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
    api_key=st.from_regex(r"nvapi-[a-z0-9]{10,20}", fullmatch=True),
    endpoint=st.one_of(st.none(), st.sampled_from([
        "https://integrate.api.nvidia.com/v1",
        "https://api.openai.com/v1",
    ])),
    timeout_seconds=st.integers(min_value=1, max_value=300),
)


camera_config_strategy = st.builds(
    CameraConfig,
    camera_id=_identifier_str,
    uri=st.from_regex(
        r"rtsp://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:[0-9]{1,5}/stream[0-9]",
        fullmatch=True,
    ),
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


# Strategy that always produces at least one camera (needed for camera mutations)
_system_config_with_cameras_strategy = st.builds(
    SystemConfig,
    deployment_profile=st.sampled_from(["single-machine", "multi-machine", "edge-cloud-hybrid"]),
    mqtt=broker_config_strategy,
    vlm=vlm_config_strategy,
    cameras=st.lists(camera_config_strategy, min_size=1, max_size=3),
    storage=storage_config_strategy,
    alerts=alert_config_strategy,
    security=security_config_strategy,
)


# ---------------------------------------------------------------------------
# Mutation definitions
#
# Each mutation is a tuple of:
#   (name, mutator_function, expected_field_path_substring)
#
# The mutator receives a deep-copied SystemConfig and returns the mutated
# config.  The expected_field_path_substring is checked against the
# ValidationError.field_path values returned by the validator.
# ---------------------------------------------------------------------------

MutationDef = Tuple[str, Callable[[SystemConfig, st.DataObject], SystemConfig], str]


def _mutate_deployment_profile(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    cfg.deployment_profile = data.draw(
        st.sampled_from(["invalid-profile", "cloud-only", "standalone", ""])
    )
    return cfg


def _mutate_vlm_backend(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    cfg.vlm.backend = data.draw(
        st.sampled_from(["invalid-backend", "openai", "llama", ""])
    )
    return cfg


def _mutate_vlm_api_key_empty(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    cfg.vlm.api_key = data.draw(st.sampled_from(["", "YOUR_API_KEY_HERE"]))
    return cfg


def _mutate_camera_id_empty(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    idx = data.draw(st.integers(min_value=0, max_value=len(cfg.cameras) - 1))
    cfg.cameras[idx].camera_id = ""
    return cfg


def _mutate_camera_uri_empty(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    idx = data.draw(st.integers(min_value=0, max_value=len(cfg.cameras) - 1))
    cfg.cameras[idx].uri = ""
    return cfg


def _mutate_camera_tenant_id_empty(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    idx = data.draw(st.integers(min_value=0, max_value=len(cfg.cameras) - 1))
    cfg.cameras[idx].tenant_id = ""
    return cfg


def _mutate_camera_site_id_empty(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    idx = data.draw(st.integers(min_value=0, max_value=len(cfg.cameras) - 1))
    cfg.cameras[idx].site_id = ""
    return cfg


def _mutate_camera_confidence_threshold(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    idx = data.draw(st.integers(min_value=0, max_value=len(cfg.cameras) - 1))
    cfg.cameras[idx].confidence_threshold = data.draw(
        st.one_of(
            st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        )
    )
    return cfg


def _mutate_camera_inference_runtime(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    idx = data.draw(st.integers(min_value=0, max_value=len(cfg.cameras) - 1))
    cfg.cameras[idx].inference_runtime = data.draw(
        st.sampled_from(["onnx", "openvino", "invalid", ""])
    )
    return cfg


def _mutate_camera_tracker_algorithm(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    idx = data.draw(st.integers(min_value=0, max_value=len(cfg.cameras) - 1))
    cfg.cameras[idx].tracker_algorithm = data.draw(
        st.sampled_from(["sort", "kalman", "invalid", ""])
    )
    return cfg


def _mutate_mqtt_port(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    cfg.mqtt.port = data.draw(
        st.one_of(
            st.just(0),
            st.integers(min_value=-65535, max_value=-1),
        )
    )
    return cfg


def _mutate_storage_timeseries_db(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    cfg.storage.timeseries_db = data.draw(
        st.sampled_from(["mysql", "postgres", "invalid", ""])
    )
    return cfg


def _mutate_storage_vector_db(cfg: SystemConfig, data: st.DataObject) -> SystemConfig:
    cfg.storage.vector_db = data.draw(
        st.sampled_from(["faiss", "milvus", "invalid", ""])
    )
    return cfg


# Map of mutation name → (mutator, expected field_path substring)
MUTATIONS: list[MutationDef] = [
    ("deployment_profile", _mutate_deployment_profile, "deployment_profile"),
    ("vlm.backend", _mutate_vlm_backend, "vlm.backend"),
    ("vlm.api_key", _mutate_vlm_api_key_empty, "vlm.api_key"),
    ("camera_id", _mutate_camera_id_empty, "camera_id"),
    ("camera_uri", _mutate_camera_uri_empty, "uri"),
    ("camera_tenant_id", _mutate_camera_tenant_id_empty, "tenant_id"),
    ("camera_site_id", _mutate_camera_site_id_empty, "site_id"),
    ("camera_confidence_threshold", _mutate_camera_confidence_threshold, "confidence_threshold"),
    ("camera_inference_runtime", _mutate_camera_inference_runtime, "inference_runtime"),
    ("camera_tracker_algorithm", _mutate_camera_tracker_algorithm, "tracker_algorithm"),
    ("mqtt.port", _mutate_mqtt_port, "mqtt.port"),
    ("storage.timeseries_db", _mutate_storage_timeseries_db, "storage.timeseries_db"),
    ("storage.vector_db", _mutate_storage_vector_db, "storage.vector_db"),
]

# Build a strategy that picks one mutation index
_mutation_index_strategy = st.integers(min_value=0, max_value=len(MUTATIONS) - 1)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestConfigValidationErrorReporting:
    """Property 15: Configuration Validation Error Reporting.

    **Validates: Requirements 18.5**
    """

    @given(
        config=_system_config_with_cameras_strategy,
        mutation_idx=_mutation_index_strategy,
        data=st.data(),
    )
    @settings(max_examples=20)
    def test_invalid_field_produces_validation_error(
        self,
        config: SystemConfig,
        mutation_idx: int,
        data: st.DataObject,
    ) -> None:
        """For a valid config with one field mutated to an invalid value,
        validate() returns at least one error identifying the mutated field."""
        mutation_name, mutator, expected_field_substr = MUTATIONS[mutation_idx]

        # Deep-copy so the original strategy-generated config is not modified
        mutated = copy.deepcopy(config)
        mutated = mutator(mutated, data)

        # Load the mutated config into ConfigManager via internal state
        mgr = ConfigManager()
        mgr._config = mutated

        errors = mgr.validate()

        # Assert at least one validation error was produced
        assert len(errors) >= 1, (
            f"Expected at least one ValidationError after mutating '{mutation_name}', "
            f"but got none."
        )

        # Assert at least one error's field_path identifies the mutated field
        field_paths = [e.field_path for e in errors]
        assert any(expected_field_substr in fp for fp in field_paths), (
            f"Expected at least one error with field_path containing "
            f"'{expected_field_substr}' after mutating '{mutation_name}', "
            f"but got field_paths: {field_paths}"
        )
