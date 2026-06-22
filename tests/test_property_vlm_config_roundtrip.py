"""Property-based test for Configuration Round-Trip Preserves New VLM Fields.

# Feature: vlm-video-snippet, Property 1: Configuration Round-Trip Preserves New Fields

**Validates: Requirements 1.1, 1.4**

For any valid CameraConfig with vlm_input_mode in {"image", "video"} and
vlm_video_duration_seconds in [1, 60], serializing the configuration to YAML
via ConfigManager.to_yaml and deserializing via ConfigManager.from_yaml SHALL
produce a CameraConfig with identical vlm_input_mode and
vlm_video_duration_seconds values.
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
# Hypothesis strategies
# ---------------------------------------------------------------------------

_identifier_str = st.from_regex(r"[a-z][a-z0-9\-]{0,19}", fullmatch=True)

camera_config_strategy = st.builds(
    CameraConfig,
    camera_id=_identifier_str,
    uri=st.from_regex(
        r"rtsp://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:[0-9]{1,5}/stream[0-9]",
        fullmatch=True,
    ),
    tenant_id=_identifier_str,
    site_id=_identifier_str,
    confidence_threshold=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    monitored_classes=st.lists(
        st.sampled_from(["person", "vehicle", "animal"]),
        min_size=0,
        max_size=3,
    ),
    inference_runtime=st.sampled_from(["pytorch", "tensorrt"]),
    model_path=st.sampled_from(["./models/yolov8n.pt", "./models/yolov8s.pt"]),
    tracker_algorithm=st.sampled_from(["deepsort", "bytetrack"]),
    frame_skip=st.integers(min_value=1, max_value=30),
    vlm_input_mode=st.sampled_from(["image", "video"]),
    vlm_video_duration_seconds=st.integers(min_value=1, max_value=60),
)

system_config_strategy = st.builds(
    SystemConfig,
    deployment_profile=st.sampled_from(
        ["single-machine", "multi-machine", "edge-cloud-hybrid"]
    ),
    mqtt=st.builds(
        BrokerConfig,
        host=st.just("localhost"),
        port=st.just(1883),
    ),
    vlm=st.builds(
        VLMConfig,
        backend=st.just("cosmos"),
        api_key=st.just("nvapi-test-key"),
    ),
    cameras=st.lists(camera_config_strategy, min_size=1, max_size=3),
    storage=st.builds(StorageConfig),
    alerts=st.builds(AlertConfig),
    security=st.builds(SecurityConfig),
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestVLMConfigRoundTrip:
    """Property 1: Configuration Round-Trip Preserves New Fields.

    **Validates: Requirements 1.1, 1.4**
    """

    @given(config=system_config_strategy)
    @settings(max_examples=20)
    def test_roundtrip_preserves_vlm_input_mode_and_duration(
        self, config: SystemConfig
    ) -> None:
        """For any valid SystemConfig with vlm_input_mode and
        vlm_video_duration_seconds, serialize to YAML then deserialize back
        and assert the new fields are identical."""
        mgr = ConfigManager()

        yaml_str = mgr.to_yaml(config)
        restored = mgr.from_yaml(yaml_str)

        assert len(restored.cameras) == len(config.cameras)
        for orig_cam, rest_cam in zip(config.cameras, restored.cameras):
            assert rest_cam.vlm_input_mode == orig_cam.vlm_input_mode, (
                f"vlm_input_mode mismatch: expected {orig_cam.vlm_input_mode!r}, "
                f"got {rest_cam.vlm_input_mode!r}"
            )
            assert rest_cam.vlm_video_duration_seconds == orig_cam.vlm_video_duration_seconds, (
                f"vlm_video_duration_seconds mismatch: expected "
                f"{orig_cam.vlm_video_duration_seconds}, "
                f"got {rest_cam.vlm_video_duration_seconds}"
            )
