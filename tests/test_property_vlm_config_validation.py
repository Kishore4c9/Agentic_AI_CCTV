"""Property-based test for Configuration Validation Rejects Invalid VLM Values.

# Feature: vlm-video-snippet, Property 2: Configuration Validation Rejects Invalid Values

**Validates: Requirements 1.3, 1.6**

For any string value not in {"image", "video"} set as vlm_input_mode, OR for any
numeric value outside [1, 60] set as vlm_video_duration_seconds,
ConfigManager.validate() SHALL return at least one ValidationError whose
field_path identifies the invalid field and whose message contains the allowed
values or range.
"""

from __future__ import annotations

import copy

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
# Hypothesis strategies — valid base config with at least one camera
# ---------------------------------------------------------------------------

_identifier_str = st.from_regex(r"[a-z][a-z0-9\-]{0,19}", fullmatch=True)

_valid_camera_strategy = st.builds(
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
        st.sampled_from(["person", "vehicle"]),
        min_size=1,
        max_size=2,
    ),
    inference_runtime=st.sampled_from(["pytorch", "tensorrt"]),
    model_path=st.just("./models/yolov8n.pt"),
    tracker_algorithm=st.sampled_from(["deepsort", "bytetrack"]),
    frame_skip=st.integers(min_value=1, max_value=10),
    vlm_input_mode=st.sampled_from(["image", "video"]),
    vlm_video_duration_seconds=st.integers(min_value=1, max_value=60),
)

_valid_system_config_strategy = st.builds(
    SystemConfig,
    deployment_profile=st.sampled_from(
        ["single-machine", "multi-machine", "edge-cloud-hybrid"]
    ),
    mqtt=st.builds(BrokerConfig, host=st.just("localhost"), port=st.just(1883)),
    vlm=st.builds(
        VLMConfig,
        backend=st.just("cosmos"),
        api_key=st.from_regex(r"nvapi-[a-z0-9]{10,20}", fullmatch=True),
    ),
    cameras=st.lists(_valid_camera_strategy, min_size=1, max_size=2),
    storage=st.builds(StorageConfig),
    alerts=st.builds(AlertConfig),
    security=st.builds(SecurityConfig),
)

# Strategy for invalid vlm_input_mode values
_invalid_vlm_input_mode = st.text(min_size=1, max_size=20).filter(
    lambda s: s not in {"image", "video"}
)

# Strategy for invalid vlm_video_duration_seconds values (outside [1, 60])
_invalid_vlm_duration = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=61, max_value=1000),
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestVLMConfigValidationRejectsInvalid:
    """Property 2: Configuration Validation Rejects Invalid Values.

    **Validates: Requirements 1.3, 1.6**
    """

    @given(
        config=_valid_system_config_strategy,
        invalid_mode=_invalid_vlm_input_mode,
        data=st.data(),
    )
    @settings(max_examples=20)
    def test_invalid_vlm_input_mode_rejected(
        self,
        config: SystemConfig,
        invalid_mode: str,
        data: st.DataObject,
    ) -> None:
        """For any string not in {"image", "video"} as vlm_input_mode,
        validate() returns at least one ValidationError with field_path
        containing 'vlm_input_mode'."""
        mutated = copy.deepcopy(config)
        idx = data.draw(
            st.integers(min_value=0, max_value=len(mutated.cameras) - 1)
        )
        mutated.cameras[idx].vlm_input_mode = invalid_mode

        mgr = ConfigManager()
        mgr._config = mutated
        errors = mgr.validate()

        field_paths = [e.field_path for e in errors]
        assert any("vlm_input_mode" in fp for fp in field_paths), (
            f"Expected ValidationError with 'vlm_input_mode' in field_path "
            f"for invalid mode {invalid_mode!r}, got: {field_paths}"
        )

    @given(
        config=_valid_system_config_strategy,
        invalid_duration=_invalid_vlm_duration,
        data=st.data(),
    )
    @settings(max_examples=20)
    def test_invalid_vlm_video_duration_rejected(
        self,
        config: SystemConfig,
        invalid_duration: int,
        data: st.DataObject,
    ) -> None:
        """For any numeric value outside [1, 60] as vlm_video_duration_seconds,
        validate() returns at least one ValidationError with field_path
        containing 'vlm_video_duration_seconds'."""
        mutated = copy.deepcopy(config)
        idx = data.draw(
            st.integers(min_value=0, max_value=len(mutated.cameras) - 1)
        )
        mutated.cameras[idx].vlm_video_duration_seconds = invalid_duration

        mgr = ConfigManager()
        mgr._config = mutated
        errors = mgr.validate()

        field_paths = [e.field_path for e in errors]
        assert any("vlm_video_duration_seconds" in fp for fp in field_paths), (
            f"Expected ValidationError with 'vlm_video_duration_seconds' in "
            f"field_path for invalid duration {invalid_duration}, got: {field_paths}"
        )
