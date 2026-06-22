"""Property-based test for Environment Template Validity.

# Feature: agentic-ai-cctv-monitoring, Property 13: Environment Template Validity

**Validates: Requirements 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.11**

For each pre-built environment template, the generated SystemConfig has all
required fields populated, a valid ``inference_runtime`` (``pytorch`` or
``tensorrt``), and a valid ``deployment_profile``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.config_manager import (
    ConfigManager,
    VALID_DEPLOYMENT_PROFILES,
    VALID_INFERENCE_RUNTIMES,
)
from agentic_cctv.environment_templates import (
    EnvironmentTemplate,
    list_templates,
)


# ---------------------------------------------------------------------------
# Strategy: sample from all pre-built templates (exhaustive)
# ---------------------------------------------------------------------------

_all_templates = list_templates()

template_strategy = st.sampled_from(_all_templates)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestEnvironmentTemplateValidity:
    """Property 13: Environment Template Validity.

    **Validates: Requirements 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.11**
    """

    @given(template=template_strategy)
    @settings(max_examples=20)
    def test_template_produces_valid_config(
        self, template: EnvironmentTemplate
    ) -> None:
        """For each pre-built template, the generated SystemConfig has all
        required fields, a valid inference_runtime, and a valid
        deployment_profile."""

        config = template.to_system_config()

        # -- deployment_profile is valid -----------------------------------
        assert config.deployment_profile in VALID_DEPLOYMENT_PROFILES, (
            f"Template '{template.name}' produced invalid deployment_profile: "
            f"'{config.deployment_profile}'"
        )

        # -- All cameras have valid inference_runtime ----------------------
        assert len(config.cameras) > 0, (
            f"Template '{template.name}' produced a config with no cameras"
        )
        for cam in config.cameras:
            assert cam.inference_runtime in VALID_INFERENCE_RUNTIMES, (
                f"Template '{template.name}' camera '{cam.camera_id}' has "
                f"invalid inference_runtime: '{cam.inference_runtime}'"
            )

        # -- Required string fields are non-empty --------------------------
        assert config.deployment_profile, "deployment_profile is empty"
        assert config.mqtt.host, "mqtt.host is empty"
        assert config.mqtt.port > 0, "mqtt.port must be positive"
        assert config.vlm.backend, "vlm.backend is empty"
        assert config.vlm.api_key, "vlm.api_key is empty"

        for cam in config.cameras:
            assert cam.camera_id, "camera_id is empty"
            assert cam.uri, "camera uri is empty"
            assert cam.tenant_id, "tenant_id is empty"
            assert cam.site_id, "site_id is empty"
            assert 0.0 <= cam.confidence_threshold <= 1.0, (
                f"confidence_threshold out of range: {cam.confidence_threshold}"
            )

        # -- Storage fields are populated ----------------------------------
        assert config.storage.timeseries_db, "timeseries_db is empty"
        assert config.storage.timeseries_path, "timeseries_path is empty"
        assert config.storage.vector_db, "vector_db is empty"
        assert config.storage.vector_path, "vector_path is empty"

        # -- Alert channels are populated ----------------------------------
        assert len(config.alerts.channels) > 0, "alert channels list is empty"

        # -- Validate via ConfigManager (except placeholder API key) -------
        mgr = ConfigManager()
        mgr._config = config
        errors = mgr.validate()
        # Filter out the expected API key placeholder error
        non_api_key_errors = [
            e for e in errors if "api_key" not in e.field_path
        ]
        assert len(non_api_key_errors) == 0, (
            f"Template '{template.name}' produced validation errors: "
            f"{[(e.field_path, e.message) for e in non_api_key_errors]}"
        )
