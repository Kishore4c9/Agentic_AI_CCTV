"""Unit tests for environment templates and deployment profiles.

Tests cover template loading, listing, error handling, overrides, and
hardware-specific settings for each pre-built template.
"""

from __future__ import annotations

import pytest

from agentic_cctv.config_manager import ConfigManager
from agentic_cctv.environment_templates import (
    FARM_TEMPLATE,
    FOREST_TEMPLATE,
    GPU_DESKTOP_TEMPLATE,
    HOME_TEMPLATE,
    MALL_TEMPLATE,
    PORT_TEMPLATE,
    EnvironmentTemplate,
    generate_config_from_template,
    get_template,
    list_templates,
)


# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    """Tests for list_templates()."""

    def test_returns_all_six_templates(self) -> None:
        templates = list_templates()
        assert len(templates) == 6

    def test_returns_sorted_by_name(self) -> None:
        templates = list_templates()
        names = [t.name for t in templates]
        assert names == sorted(names)

    def test_all_templates_are_environment_template_instances(self) -> None:
        for t in list_templates():
            assert isinstance(t, EnvironmentTemplate)

    def test_expected_template_names(self) -> None:
        names = {t.name for t in list_templates()}
        assert names == {"home", "farm", "forest", "mall", "port", "gpu_desktop"}


# ---------------------------------------------------------------------------
# get_template
# ---------------------------------------------------------------------------


class TestGetTemplate:
    """Tests for get_template()."""

    @pytest.mark.parametrize(
        "name",
        ["home", "farm", "forest", "mall", "port", "gpu_desktop"],
    )
    def test_each_template_can_be_loaded(self, name: str) -> None:
        template = get_template(name)
        assert template.name == name

    def test_raises_key_error_for_unknown_template(self) -> None:
        with pytest.raises(KeyError, match="Unknown environment template"):
            get_template("nonexistent")

    def test_error_message_lists_available_templates(self) -> None:
        with pytest.raises(KeyError, match="home"):
            get_template("bad_name")


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    """Tests for override support in generate_config_from_template."""

    def test_override_inference_runtime(self) -> None:
        config = generate_config_from_template(
            "home", inference_runtime="tensorrt"
        )
        assert config.cameras[0].inference_runtime == "tensorrt"

    def test_override_deployment_profile(self) -> None:
        config = generate_config_from_template(
            "home", deployment_profile="multi-machine"
        )
        assert config.deployment_profile == "multi-machine"

    def test_override_vlm_api_key(self) -> None:
        config = generate_config_from_template(
            "farm", vlm_api_key="my-secret-key"
        )
        assert config.vlm.api_key == "my-secret-key"

    def test_override_mqtt_host_and_port(self) -> None:
        config = generate_config_from_template(
            "mall", mqtt_host="broker.example.com", mqtt_port=8883
        )
        assert config.mqtt.host == "broker.example.com"
        assert config.mqtt.port == 8883

    def test_override_camera_id(self) -> None:
        config = generate_config_from_template(
            "port", camera_id="cam-dock-01"
        )
        assert config.cameras[0].camera_id == "cam-dock-01"

    def test_override_tenant_and_site(self) -> None:
        config = generate_config_from_template(
            "gpu_desktop", tenant_id="acme-corp", site_id="hq-floor-3"
        )
        assert config.cameras[0].tenant_id == "acme-corp"
        assert config.cameras[0].site_id == "hq-floor-3"


# ---------------------------------------------------------------------------
# Hardware-specific settings per template
# ---------------------------------------------------------------------------


class TestTemplateHardwareSettings:
    """Verify each template has correct hardware-specific defaults."""

    def test_home_template(self) -> None:
        t = HOME_TEMPLATE
        assert t.hardware_class == "jetson_nano"
        assert t.power_profile == "mains"
        assert t.inference_runtime == "pytorch"
        assert t.deployment_profile == "single-machine"

        config = t.to_system_config()
        assert config.cameras[0].inference_runtime == "pytorch"
        assert config.deployment_profile == "single-machine"

    def test_farm_template(self) -> None:
        t = FARM_TEMPLATE
        assert t.hardware_class == "jetson_nano"
        assert t.power_profile == "solar_battery"
        assert t.inference_runtime == "tensorrt"

        config = t.to_system_config()
        assert config.cameras[0].inference_runtime == "tensorrt"

    def test_forest_template(self) -> None:
        t = FOREST_TEMPLATE
        assert t.hardware_class == "coral_dev_board_micro"
        assert t.power_profile == "battery_lora"
        assert t.inference_runtime == "tensorrt"

        config = t.to_system_config()
        assert config.cameras[0].inference_runtime == "tensorrt"

    def test_mall_template(self) -> None:
        t = MALL_TEMPLATE
        assert t.hardware_class == "jetson_orin_nx"
        assert t.power_profile == "mains"
        assert t.inference_runtime == "tensorrt"

        config = t.to_system_config()
        assert config.cameras[0].inference_runtime == "tensorrt"

    def test_port_template(self) -> None:
        t = PORT_TEMPLATE
        assert t.hardware_class == "jetson_orin_nx"
        assert t.power_profile == "mains"
        assert t.inference_runtime == "tensorrt"

        config = t.to_system_config()
        assert config.cameras[0].inference_runtime == "tensorrt"

    def test_gpu_desktop_template(self) -> None:
        t = GPU_DESKTOP_TEMPLATE
        assert t.hardware_class == "nvidia_gpu_desktop"
        assert t.power_profile == "mains"
        assert t.inference_runtime == "pytorch"

        config = t.to_system_config()
        assert config.cameras[0].inference_runtime == "pytorch"


# ---------------------------------------------------------------------------
# ConfigManager.load_template integration
# ---------------------------------------------------------------------------


class TestConfigManagerLoadTemplate:
    """Tests for ConfigManager.load_template()."""

    def test_load_template_returns_system_config(self) -> None:
        mgr = ConfigManager()
        config = mgr.load_template("home")
        assert config.deployment_profile == "single-machine"
        assert len(config.cameras) == 1

    def test_load_template_with_overrides(self) -> None:
        mgr = ConfigManager()
        config = mgr.load_template(
            "farm", inference_runtime="pytorch", vlm_api_key="test-key"
        )
        assert config.cameras[0].inference_runtime == "pytorch"
        assert config.vlm.api_key == "test-key"

    def test_load_template_sets_internal_config(self) -> None:
        mgr = ConfigManager()
        config = mgr.load_template("mall")
        # After load_template, validate() should work
        errors = mgr.validate()
        # Only the placeholder API key error is expected
        non_api_errors = [e for e in errors if "api_key" not in e.field_path]
        assert len(non_api_errors) == 0

    def test_load_template_raises_for_unknown(self) -> None:
        mgr = ConfigManager()
        with pytest.raises(KeyError):
            mgr.load_template("nonexistent")


# ---------------------------------------------------------------------------
# Generated config structure
# ---------------------------------------------------------------------------


class TestGeneratedConfigStructure:
    """Verify generated configs have correct structure."""

    @pytest.mark.parametrize(
        "name",
        ["home", "farm", "forest", "mall", "port", "gpu_desktop"],
    )
    def test_config_has_one_camera(self, name: str) -> None:
        config = generate_config_from_template(name)
        assert len(config.cameras) == 1

    @pytest.mark.parametrize(
        "name",
        ["home", "farm", "forest", "mall", "port", "gpu_desktop"],
    )
    def test_config_uses_sqlite_and_chromadb(self, name: str) -> None:
        config = generate_config_from_template(name)
        assert config.storage.timeseries_db == "sqlite"
        assert config.storage.vector_db == "chromadb"

    @pytest.mark.parametrize(
        "name",
        ["home", "farm", "forest", "mall", "port", "gpu_desktop"],
    )
    def test_config_uses_localhost_mqtt(self, name: str) -> None:
        config = generate_config_from_template(name)
        assert config.mqtt.host == "localhost"
        assert config.mqtt.port == 1883

    @pytest.mark.parametrize(
        "name",
        ["home", "farm", "forest", "mall", "port", "gpu_desktop"],
    )
    def test_config_has_alert_channels(self, name: str) -> None:
        config = generate_config_from_template(name)
        assert "push" in config.alerts.channels
        assert "webhook" in config.alerts.channels
