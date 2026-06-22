"""Unit tests for the MQTT client module.

Tests cover:
- ``build_topic`` validation and formatting
- ``MQTTPublisher`` connect / publish / disconnect lifecycle
- ``MQTTSubscriber`` connect / subscribe / disconnect lifecycle
- QoS validation, retained messages, TLS configuration
- Protocol conformance (``MQTTPublisherProtocol``)

All tests mock ``paho.mqtt.client.Client`` — no real broker is needed.
"""

from __future__ import annotations

import asyncio
import ssl
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agentic_cctv.models import BrokerConfig
from agentic_cctv.mqtt_client import (
    MQTTPublisher,
    MQTTSubscriber,
    build_topic,
    _configure_tls,
    _configure_auth,
)
from agentic_cctv.event_encoder import MQTTPublisherProtocol


# ---------------------------------------------------------------------------
# build_topic tests
# ---------------------------------------------------------------------------


class TestBuildTopic:
    """Tests for the ``build_topic`` helper."""

    def test_valid_topic(self) -> None:
        result = build_topic("tenant-a", "site-1", "cam-01", "events")
        assert result == "tenant-a/site-1/cam-01/events"

    def test_valid_topic_alerts(self) -> None:
        result = build_topic("acme", "hq", "lobby", "alerts")
        assert result == "acme/hq/lobby/alerts"

    def test_valid_topic_health(self) -> None:
        result = build_topic("acme", "hq", "lobby", "health")
        assert result == "acme/hq/lobby/health"

    def test_empty_tenant_id_raises(self) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            build_topic("", "site-1", "cam-01", "events")

    def test_empty_site_id_raises(self) -> None:
        with pytest.raises(ValueError, match="site_id"):
            build_topic("tenant-a", "", "cam-01", "events")

    def test_empty_camera_id_raises(self) -> None:
        with pytest.raises(ValueError, match="camera_id"):
            build_topic("tenant-a", "site-1", "", "events")

    def test_empty_suffix_raises(self) -> None:
        with pytest.raises(ValueError, match="suffix"):
            build_topic("tenant-a", "site-1", "cam-01", "")

    def test_whitespace_only_tenant_raises(self) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            build_topic("   ", "site-1", "cam-01", "events")

    def test_whitespace_only_suffix_raises(self) -> None:
        with pytest.raises(ValueError, match="suffix"):
            build_topic("tenant-a", "site-1", "cam-01", "  ")

    def test_topic_has_four_segments(self) -> None:
        topic = build_topic("t", "s", "c", "events")
        parts = topic.split("/")
        assert len(parts) == 4
        assert all(p for p in parts)  # no empty segments


# ---------------------------------------------------------------------------
# MQTTPublisher tests
# ---------------------------------------------------------------------------


class TestMQTTPublisher:
    """Tests for ``MQTTPublisher``."""

    def _make_config(self, **overrides: object) -> BrokerConfig:
        defaults = dict(host="localhost", port=1883)
        defaults.update(overrides)
        return BrokerConfig(**defaults)  # type: ignore[arg-type]

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_connect_success(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        pub = MQTTPublisher(self._make_config())
        pub._client = mock_instance

        await pub.connect()

        mock_instance.connect.assert_called_once_with("localhost", 1883, keepalive=30)
        mock_instance.loop_start.assert_called_once()
        assert pub.is_connected is True

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        mock_instance.connect.side_effect = OSError("Connection refused")
        pub = MQTTPublisher(self._make_config())
        pub._client = mock_instance

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await pub.connect()

        assert pub.is_connected is False

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publish_qos_0(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        pub = MQTTPublisher(self._make_config())
        pub._client = mock_instance
        pub._connected = True

        await pub.publish("test/topic", b"hello", qos=0, retain=False)

        mock_instance.publish.assert_called_once_with(
            "test/topic", b"hello", qos=0, retain=False
        )
        # QoS 0 should NOT wait for publish
        mock_info.wait_for_publish.assert_not_called()

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publish_qos_1(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        pub = MQTTPublisher(self._make_config())
        pub._client = mock_instance
        pub._connected = True

        await pub.publish("test/topic", b"hello", qos=1, retain=False)

        mock_instance.publish.assert_called_once_with(
            "test/topic", b"hello", qos=1, retain=False
        )
        mock_info.wait_for_publish.assert_called_once()

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publish_qos_2(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        pub = MQTTPublisher(self._make_config())
        pub._client = mock_instance
        pub._connected = True

        await pub.publish("test/topic", b"alert-data", qos=2, retain=False)

        mock_instance.publish.assert_called_once_with(
            "test/topic", b"alert-data", qos=2, retain=False
        )
        mock_info.wait_for_publish.assert_called_once()

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publish_retained(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        pub = MQTTPublisher(self._make_config())
        pub._client = mock_instance
        pub._connected = True

        await pub.publish("test/health", b"heartbeat", qos=1, retain=True)

        mock_instance.publish.assert_called_once_with(
            "test/health", b"heartbeat", qos=1, retain=True
        )

    @pytest.mark.asyncio
    async def test_publish_invalid_qos_raises(self) -> None:
        pub = MQTTPublisher(self._make_config())
        pub._connected = True

        with pytest.raises(ValueError, match="QoS must be 0, 1, or 2"):
            await pub.publish("t", b"x", qos=3)

    @pytest.mark.asyncio
    async def test_publish_when_not_connected_raises(self) -> None:
        pub = MQTTPublisher(self._make_config())

        with pytest.raises(ConnectionError, match="not connected"):
            await pub.publish("t", b"x", qos=1)

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_disconnect(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        pub = MQTTPublisher(self._make_config())
        pub._client = mock_instance
        pub._connected = True

        await pub.disconnect()

        mock_instance.loop_stop.assert_called_once()
        mock_instance.disconnect.assert_called_once()
        assert pub.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_noop(self) -> None:
        pub = MQTTPublisher(self._make_config())
        # Should not raise
        await pub.disconnect()
        assert pub.is_connected is False

    def test_publisher_satisfies_protocol(self) -> None:
        """MQTTPublisher must be a structural subtype of MQTTPublisherProtocol."""
        pub = MQTTPublisher(self._make_config())
        assert isinstance(pub, MQTTPublisherProtocol)


# ---------------------------------------------------------------------------
# MQTTSubscriber tests
# ---------------------------------------------------------------------------


class TestMQTTSubscriber:
    """Tests for ``MQTTSubscriber``."""

    def _make_config(self, **overrides: object) -> BrokerConfig:
        defaults = dict(host="localhost", port=1883)
        defaults.update(overrides)
        return BrokerConfig(**defaults)  # type: ignore[arg-type]

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_connect_success(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        sub = MQTTSubscriber(self._make_config())
        sub._client = mock_instance

        await sub.connect()

        mock_instance.connect.assert_called_once_with("localhost", 1883, keepalive=30)
        mock_instance.loop_start.assert_called_once()
        assert sub.is_connected is True

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        mock_instance.connect.side_effect = OSError("Connection refused")
        sub = MQTTSubscriber(self._make_config())
        sub._client = mock_instance

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await sub.connect()

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_subscribe_registers_callback(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        sub = MQTTSubscriber(self._make_config())
        sub._client = mock_instance
        sub._connected = True

        callback = MagicMock()
        await sub.subscribe("test/+/events", qos=1, callback=callback)

        mock_instance.subscribe.assert_called_once_with("test/+/events", qos=1)
        assert "test/+/events" in sub._callbacks

    @pytest.mark.asyncio
    async def test_subscribe_invalid_qos_raises(self) -> None:
        sub = MQTTSubscriber(self._make_config())
        sub._connected = True

        with pytest.raises(ValueError, match="QoS must be 0, 1, or 2"):
            await sub.subscribe("t", qos=5)

    @pytest.mark.asyncio
    async def test_subscribe_when_not_connected_raises(self) -> None:
        sub = MQTTSubscriber(self._make_config())

        with pytest.raises(ConnectionError, match="not connected"):
            await sub.subscribe("t", qos=1)

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_disconnect(self, MockClient: MagicMock) -> None:
        mock_instance = MockClient.return_value
        sub = MQTTSubscriber(self._make_config())
        sub._client = mock_instance
        sub._connected = True
        sub._callbacks["test"] = MagicMock()

        await sub.disconnect()

        mock_instance.loop_stop.assert_called_once()
        mock_instance.disconnect.assert_called_once()
        assert sub.is_connected is False
        assert len(sub._callbacks) == 0

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_on_message_dispatches_to_callback(self, MockClient: MagicMock) -> None:
        """Verify that incoming messages are dispatched to the correct callback."""
        mock_instance = MockClient.return_value
        sub = MQTTSubscriber(self._make_config())
        sub._client = mock_instance

        callback = MagicMock()
        sub._callbacks["tenant/+/+/events"] = callback

        # Simulate an incoming message
        msg = MagicMock()
        msg.topic = "tenant/site1/cam1/events"
        msg.payload = b'{"event_id": "123"}'
        msg.qos = 1

        sub._on_message(mock_instance, None, msg)

        callback.assert_called_once_with(
            "tenant/site1/cam1/events", b'{"event_id": "123"}', 1
        )

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_on_message_callback_error_does_not_propagate(self, MockClient: MagicMock) -> None:
        """A failing callback should not crash the subscriber."""
        mock_instance = MockClient.return_value
        sub = MQTTSubscriber(self._make_config())
        sub._client = mock_instance

        callback = MagicMock(side_effect=RuntimeError("boom"))
        sub._callbacks["test/#"] = callback

        msg = MagicMock()
        msg.topic = "test/foo"
        msg.payload = b"data"
        msg.qos = 0

        # Should not raise
        sub._on_message(mock_instance, None, msg)


# ---------------------------------------------------------------------------
# TLS / Auth configuration tests
# ---------------------------------------------------------------------------


class TestTLSAndAuth:
    """Tests for TLS and authentication configuration helpers."""

    def test_configure_tls_when_disabled(self) -> None:
        client = MagicMock()
        config = BrokerConfig(use_tls=False)
        _configure_tls(client, config)
        client.tls_set_context.assert_not_called()

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_configure_tls_with_ca_cert(self, mock_ctx_factory: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(use_tls=True, ca_cert="/path/ca.pem")
        _configure_tls(client, config)

        mock_ctx_factory.assert_called_once_with(cafile="/path/ca.pem")
        # Verify TLS 1.3 minimum is enforced (Requirement 10.1)
        assert mock_ctx.minimum_version == ssl.TLSVersion.TLSv1_3
        client.tls_set_context.assert_called_once_with(mock_ctx)

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_configure_tls_with_client_certs(self, mock_ctx_factory: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(
            use_tls=True,
            ca_cert="/path/ca.pem",
            client_cert="/path/client.pem",
            client_key="/path/client.key",
        )
        _configure_tls(client, config)

        mock_ctx.load_cert_chain.assert_called_once_with(
            certfile="/path/client.pem", keyfile="/path/client.key"
        )

    def test_configure_auth_with_credentials(self) -> None:
        client = MagicMock()
        config = BrokerConfig(username="user", password="pass")
        _configure_auth(client, config)
        client.username_pw_set.assert_called_once_with("user", "pass")

    def test_configure_auth_without_credentials(self) -> None:
        client = MagicMock()
        config = BrokerConfig()
        _configure_auth(client, config)
        client.username_pw_set.assert_not_called()


# ---------------------------------------------------------------------------
# QoS mapping tests (from design document)
# ---------------------------------------------------------------------------


class TestQoSMapping:
    """Verify the QoS mapping table from the design document."""

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_events_use_qos_1_no_retain(self, MockClient: MagicMock) -> None:
        """Structured_Event → /events → QoS 1, not retained."""
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        pub = MQTTPublisher(BrokerConfig())
        pub._client = mock_instance
        pub._connected = True

        topic = build_topic("tenant", "site", "cam", "events")
        await pub.publish(topic, b"event-data", qos=1, retain=False)

        mock_instance.publish.assert_called_once_with(
            "tenant/site/cam/events", b"event-data", qos=1, retain=False
        )

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_alerts_use_qos_2_no_retain(self, MockClient: MagicMock) -> None:
        """Alert → /alerts → QoS 2, not retained."""
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        pub = MQTTPublisher(BrokerConfig())
        pub._client = mock_instance
        pub._connected = True

        topic = build_topic("tenant", "site", "cam", "alerts")
        await pub.publish(topic, b"alert-data", qos=2, retain=False)

        mock_instance.publish.assert_called_once_with(
            "tenant/site/cam/alerts", b"alert-data", qos=2, retain=False
        )

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_health_uses_qos_1_retained(self, MockClient: MagicMock) -> None:
        """Health/Heartbeat → /health → QoS 1, retained."""
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        pub = MQTTPublisher(BrokerConfig())
        pub._client = mock_instance
        pub._connected = True

        topic = build_topic("tenant", "site", "cam", "health")
        await pub.publish(topic, b"heartbeat", qos=1, retain=True)

        mock_instance.publish.assert_called_once_with(
            "tenant/site/cam/health", b"heartbeat", qos=1, retain=True
        )
