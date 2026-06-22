"""Integration tests for distributed MQTT broker support.

Tests cover:
- TLS 1.3 handshake configuration and ``tls_insecure`` mode
- Username/password authentication
- Remote broker publish/subscribe (non-localhost)
- Store-and-forward over network interruption with reconnection
- Connectivity validation (``validate_connectivity``)
- Connection timeout configuration

All tests mock ``paho.mqtt.client.Client`` — no real broker is needed.
"""

from __future__ import annotations

import ssl
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from agentic_cctv.models import BrokerConfig
from agentic_cctv.mqtt_client import (
    ConnectivityResult,
    MQTTPublisher,
    MQTTSubscriber,
    _configure_tls,
    _configure_auth,
)
from agentic_cctv.store_and_forward import StoreAndForwardQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _remote_config(**overrides: object) -> BrokerConfig:
    """Create a BrokerConfig for a remote broker (non-localhost)."""
    defaults = dict(
        host="mqtt.example.com",
        port=8883,
        use_tls=True,
        ca_cert="/path/ca.pem",
        username="admin",
        password="secret",
    )
    defaults.update(overrides)
    return BrokerConfig(**defaults)  # type: ignore[arg-type]


def _patch_ssl_and_paho():
    """Return stacked patches for both paho Client and ssl.create_default_context."""
    return (
        patch("agentic_cctv.mqtt_client.paho_mqtt.Client"),
        patch("agentic_cctv.mqtt_client.ssl.create_default_context"),
    )


# ---------------------------------------------------------------------------
# TLS handshake configuration tests
# ---------------------------------------------------------------------------


class TestTLSHandshakeConfiguration:
    """Verify TLS 1.3 minimum, CA cert, client certs, and tls_insecure."""

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_tls_13_minimum_version_enforced(self, mock_ctx_factory: MagicMock) -> None:
        """When use_tls=True, SSL context must have TLS 1.3 minimum."""
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(use_tls=True, ca_cert="/path/ca.pem")
        _configure_tls(client, config)

        assert mock_ctx.minimum_version == ssl.TLSVersion.TLSv1_3

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_ca_cert_loaded(self, mock_ctx_factory: MagicMock) -> None:
        """CA certificate is passed to ssl.create_default_context."""
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(use_tls=True, ca_cert="/certs/ca-bundle.pem")
        _configure_tls(client, config)

        mock_ctx_factory.assert_called_once_with(cafile="/certs/ca-bundle.pem")

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_client_certs_loaded_when_provided(self, mock_ctx_factory: MagicMock) -> None:
        """Client cert and key are loaded into the SSL context."""
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(
            use_tls=True,
            ca_cert="/certs/ca.pem",
            client_cert="/certs/client.pem",
            client_key="/certs/client.key",
        )
        _configure_tls(client, config)

        mock_ctx.load_cert_chain.assert_called_once_with(
            certfile="/certs/client.pem", keyfile="/certs/client.key"
        )

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_client_certs_not_loaded_when_absent(self, mock_ctx_factory: MagicMock) -> None:
        """Client certs are not loaded when not provided."""
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(use_tls=True, ca_cert="/certs/ca.pem")
        _configure_tls(client, config)

        mock_ctx.load_cert_chain.assert_not_called()

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_tls_insecure_disables_hostname_check(self, mock_ctx_factory: MagicMock) -> None:
        """When tls_insecure=True, hostname check is disabled and verify mode is CERT_NONE."""
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(
            use_tls=True,
            ca_cert="/certs/ca.pem",
            tls_insecure=True,
        )
        _configure_tls(client, config)

        assert mock_ctx.check_hostname is False
        assert mock_ctx.verify_mode == ssl.CERT_NONE

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_tls_secure_preserves_defaults(self, mock_ctx_factory: MagicMock) -> None:
        """When tls_insecure=False (default), hostname check and verify mode are not modified."""
        mock_ctx = MagicMock()
        # Simulate default ssl context attributes
        mock_ctx.check_hostname = True
        mock_ctx.verify_mode = ssl.CERT_REQUIRED
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(
            use_tls=True,
            ca_cert="/certs/ca.pem",
            tls_insecure=False,
        )
        _configure_tls(client, config)

        # check_hostname and verify_mode should remain at their defaults
        assert mock_ctx.check_hostname is True
        assert mock_ctx.verify_mode == ssl.CERT_REQUIRED

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    def test_tls_context_set_on_client(self, mock_ctx_factory: MagicMock) -> None:
        """The SSL context is applied to the paho-mqtt client."""
        mock_ctx = MagicMock()
        mock_ctx_factory.return_value = mock_ctx

        client = MagicMock()
        config = BrokerConfig(use_tls=True, ca_cert="/certs/ca.pem")
        _configure_tls(client, config)

        client.tls_set_context.assert_called_once_with(mock_ctx)


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Verify username/password authentication configuration."""

    def test_auth_credentials_set_on_client(self) -> None:
        """Username and password are set on the paho-mqtt client."""
        client = MagicMock()
        config = BrokerConfig(username="mqtt-user", password="mqtt-pass")
        _configure_auth(client, config)

        client.username_pw_set.assert_called_once_with("mqtt-user", "mqtt-pass")

    def test_auth_not_set_without_username(self) -> None:
        """No auth is set when username is None."""
        client = MagicMock()
        config = BrokerConfig(username=None, password=None)
        _configure_auth(client, config)

        client.username_pw_set.assert_not_called()

    def test_auth_with_username_only(self) -> None:
        """Username with None password is still set (some brokers allow this)."""
        client = MagicMock()
        config = BrokerConfig(username="user-only", password=None)
        _configure_auth(client, config)

        client.username_pw_set.assert_called_once_with("user-only", None)


# ---------------------------------------------------------------------------
# Remote broker publish/subscribe tests
# ---------------------------------------------------------------------------


class TestRemoteBrokerPublishSubscribe:
    """Test publisher and subscriber with remote (non-localhost) broker."""

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publisher_connects_to_remote_broker(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Publisher connects to a remote broker address."""
        mock_instance = MockClient.return_value
        config = _remote_config()
        pub = MQTTPublisher(config)
        pub._client = mock_instance

        await pub.connect()

        mock_instance.connect.assert_called_once_with(
            "mqtt.example.com", 8883, keepalive=30
        )
        assert pub.is_connected is True

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_subscriber_connects_to_remote_broker(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Subscriber connects to a remote broker address."""
        mock_instance = MockClient.return_value
        config = _remote_config()
        sub = MQTTSubscriber(config)
        sub._client = mock_instance

        await sub.connect()

        mock_instance.connect.assert_called_once_with(
            "mqtt.example.com", 8883, keepalive=30
        )
        assert sub.is_connected is True

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publish_to_remote_broker(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Messages can be published to a remote broker."""
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        config = _remote_config()
        pub = MQTTPublisher(config)
        pub._client = mock_instance
        pub._connected = True

        await pub.publish("tenant/site/cam/events", b"event-data", qos=1)

        mock_instance.publish.assert_called_once_with(
            "tenant/site/cam/events", b"event-data", qos=1, retain=False
        )
        mock_info.wait_for_publish.assert_called_once()

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_subscribe_on_remote_broker(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Subscriber can subscribe to topics on a remote broker."""
        mock_instance = MockClient.return_value
        config = _remote_config()
        sub = MQTTSubscriber(config)
        sub._client = mock_instance
        sub._connected = True

        callback = MagicMock()
        await sub.subscribe("+/+/+/events", qos=1, callback=callback)

        mock_instance.subscribe.assert_called_once_with("+/+/+/events", qos=1)
        assert "+/+/+/events" in sub._callbacks
        assert "+/+/+/events" in sub._subscriptions

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_subscriber_tracks_subscriptions_for_resubscribe(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Subscriptions are tracked for re-subscribe on reconnection."""
        mock_instance = MockClient.return_value
        config = _remote_config()
        sub = MQTTSubscriber(config)
        sub._client = mock_instance
        sub._connected = True

        await sub.subscribe("topic/a", qos=1, callback=MagicMock())
        await sub.subscribe("topic/b", qos=2, callback=MagicMock())

        assert sub._subscriptions == {"topic/a": 1, "topic/b": 2}


# ---------------------------------------------------------------------------
# Store-and-forward over network interruption tests
# ---------------------------------------------------------------------------


class TestStoreAndForwardReconnection:
    """Test reconnection flow with store-and-forward queue drain."""

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_on_disconnect_sets_connected_false(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Disconnect callback sets connected to False and triggers reconnection."""
        mock_instance = MockClient.return_value
        # Prevent reconnect from succeeding immediately
        mock_instance.reconnect.side_effect = OSError("still down")

        config = _remote_config()
        pub = MQTTPublisher(config)
        pub._client = mock_instance
        pub._connected = True

        # Patch time.sleep so the reconnect loop doesn't block
        # Use a counter to break the loop after first attempt
        call_count = [0]
        original_connected = [None]

        def fake_sleep(seconds):
            call_count[0] += 1
            # Capture connected state after first reconnect attempt
            original_connected[0] = pub._connected
            # Force connected to True to break the while loop
            pub._connected = True

        with patch("agentic_cctv.mqtt_client.time.sleep", side_effect=fake_sleep):
            pub._on_disconnect(mock_instance, None, MagicMock(), 1, None)
            # Wait for the background thread to finish
            import time
            time.sleep(0.2)

        # The disconnect callback set _connected to False before reconnect attempt
        assert original_connected[0] is False

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_publisher_reconnect_drains_queue(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """On reconnection, the store-and-forward queue is drained."""
        mock_instance = MockClient.return_value
        config = _remote_config()

        # Create a mock store-and-forward queue
        mock_queue = MagicMock()
        mock_queue.drain.return_value = 3

        pub = MQTTPublisher(config, store_and_forward_queue=mock_queue)
        pub._client = mock_instance
        pub._connected = True

        # Simulate successful reconnection
        pub._drain_queue_on_reconnect()

        mock_queue.drain.assert_called_once_with(pub)

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_store_and_forward_buffers_during_disconnect(self, MockClient: MagicMock) -> None:
        """Messages are buffered in the store-and-forward queue during disconnect."""
        queue = StoreAndForwardQueue(":memory:")

        # Enqueue messages while "disconnected"
        queue.enqueue("tenant/site/cam/events", b"event-1", qos=1)
        queue.enqueue("tenant/site/cam/events", b"event-2", qos=1)
        queue.enqueue("tenant/site/cam/alerts", b"alert-1", qos=2)

        assert queue.size() == 3

        queue.close()

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_store_and_forward_drain_after_reconnect(self, MockClient: MagicMock) -> None:
        """Queue is drained in FIFO order after reconnection."""
        mock_instance = MockClient.return_value
        mock_info = MagicMock()
        mock_instance.publish.return_value = mock_info

        config = BrokerConfig(host="localhost", port=1883)
        queue = StoreAndForwardQueue(":memory:")

        # Enqueue messages
        queue.enqueue("topic/a", b"msg-1", qos=1)
        queue.enqueue("topic/b", b"msg-2", qos=1)

        pub = MQTTPublisher(config, store_and_forward_queue=queue)
        pub._client = mock_instance
        pub._connected = True

        # Drain the queue (synchronous call, no running event loop)
        drained = queue.drain(pub)

        assert drained == 2
        assert queue.size() == 0

        queue.close()

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_reconnect_loop_retries_on_failure(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Reconnect loop retries with exponential backoff on failure."""
        mock_instance = MockClient.return_value
        # First call fails, second succeeds
        mock_instance.reconnect.side_effect = [OSError("refused"), None]

        config = _remote_config()
        pub = MQTTPublisher(config)
        pub._client = mock_instance
        pub._connected = False

        # Patch time.sleep to avoid actual delays
        with patch("agentic_cctv.mqtt_client.time.sleep") as mock_sleep:
            pub._reconnect_loop()

        assert pub._connected is True
        assert mock_instance.reconnect.call_count == 2
        # First retry should sleep for 1 second (initial delay)
        mock_sleep.assert_called_once_with(1)

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_subscriber_on_disconnect_sets_connected_false(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Subscriber disconnect callback sets connected to False and triggers reconnection."""
        mock_instance = MockClient.return_value
        # Prevent reconnect from succeeding immediately
        mock_instance.reconnect.side_effect = OSError("still down")

        config = _remote_config()
        sub = MQTTSubscriber(config)
        sub._client = mock_instance
        sub._connected = True

        # Patch time.sleep so the reconnect loop doesn't block
        call_count = [0]
        original_connected = [None]

        def fake_sleep(seconds):
            call_count[0] += 1
            original_connected[0] = sub._connected
            # Force connected to True to break the while loop
            sub._connected = True

        with patch("agentic_cctv.mqtt_client.time.sleep", side_effect=fake_sleep):
            sub._on_disconnect(mock_instance, None, MagicMock(), 1, None)
            import time
            time.sleep(0.2)

        # The disconnect callback set _connected to False before reconnect attempt
        assert original_connected[0] is False

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    def test_subscriber_reconnect_resubscribes(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Subscriber re-subscribes to all topics after reconnection."""
        mock_instance = MockClient.return_value

        config = _remote_config()
        sub = MQTTSubscriber(config)
        sub._client = mock_instance
        sub._connected = False

        # Set up tracked subscriptions
        sub._subscriptions = {"topic/a": 1, "topic/b": 2}
        sub._callbacks = {"topic/a": MagicMock(), "topic/b": MagicMock()}

        # Simulate successful reconnection
        sub._resubscribe()

        assert mock_instance.subscribe.call_count == 2
        mock_instance.subscribe.assert_any_call("topic/a", qos=1)
        mock_instance.subscribe.assert_any_call("topic/b", qos=2)


# ---------------------------------------------------------------------------
# Connectivity validation tests
# ---------------------------------------------------------------------------


class TestConnectivityValidation:
    """Test the validate_connectivity method."""

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publisher_validate_connectivity_success(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Successful connectivity validation returns success result."""
        # The first Client() call is for the publisher __init__
        # The second Client() call is for validate_connectivity's test_client
        mock_init_client = MagicMock()
        mock_test_client = MagicMock()
        MockClient.side_effect = [mock_init_client, mock_test_client]

        # Simulate CONNACK with rc=0 (success)
        def fake_connect(*args, **kwargs):
            pass

        def fake_loop_start():
            # Trigger on_connect callback immediately
            on_connect = mock_test_client.on_connect
            rc_mock = MagicMock()
            rc_mock.value = 0
            on_connect(mock_test_client, None, None, rc_mock, None)

        mock_test_client.connect = fake_connect
        mock_test_client.loop_start = fake_loop_start

        config = _remote_config()
        pub = MQTTPublisher(config)
        result = await pub.validate_connectivity()

        assert result.success is True
        assert "mqtt.example.com" in result.message
        assert result.latency_ms is not None

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publisher_validate_connectivity_connection_refused(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Failed connectivity validation returns failure result."""
        mock_init_client = MagicMock()
        mock_test_client = MagicMock()
        MockClient.side_effect = [mock_init_client, mock_test_client]
        mock_test_client.connect.side_effect = OSError("Connection refused")

        config = _remote_config()
        pub = MQTTPublisher(config)
        result = await pub.validate_connectivity()

        assert result.success is False
        assert "Failed to connect" in result.message

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publisher_validate_connectivity_bad_credentials(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Connectivity validation with bad credentials returns failure."""
        mock_init_client = MagicMock()
        mock_test_client = MagicMock()
        MockClient.side_effect = [mock_init_client, mock_test_client]

        def fake_connect(*args, **kwargs):
            pass

        def fake_loop_start():
            on_connect = mock_test_client.on_connect
            rc_mock = MagicMock()
            rc_mock.value = 5  # Not authorized
            on_connect(mock_test_client, None, None, rc_mock, None)

        mock_test_client.connect = fake_connect
        mock_test_client.loop_start = fake_loop_start

        config = _remote_config()
        pub = MQTTPublisher(config)
        result = await pub.validate_connectivity()

        assert result.success is False
        assert "refused" in result.message.lower()

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_subscriber_validate_connectivity_success(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Subscriber connectivity validation works the same as publisher."""
        mock_init_client = MagicMock()
        mock_test_client = MagicMock()
        MockClient.side_effect = [mock_init_client, mock_test_client]

        def fake_connect(*args, **kwargs):
            pass

        def fake_loop_start():
            on_connect = mock_test_client.on_connect
            rc_mock = MagicMock()
            rc_mock.value = 0
            on_connect(mock_test_client, None, None, rc_mock, None)

        mock_test_client.connect = fake_connect
        mock_test_client.loop_start = fake_loop_start

        config = _remote_config()
        sub = MQTTSubscriber(config)
        result = await sub.validate_connectivity()

        assert result.success is True
        assert result.latency_ms is not None

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_subscriber_validate_connectivity_failure(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """Subscriber connectivity validation reports failure."""
        mock_init_client = MagicMock()
        mock_test_client = MagicMock()
        MockClient.side_effect = [mock_init_client, mock_test_client]
        mock_test_client.connect.side_effect = OSError("Network unreachable")

        config = _remote_config()
        sub = MQTTSubscriber(config)
        result = await sub.validate_connectivity()

        assert result.success is False
        assert "Failed to connect" in result.message


# ---------------------------------------------------------------------------
# Connection timeout tests
# ---------------------------------------------------------------------------


class TestConnectionTimeout:
    """Test that connection timeout is properly configured."""

    def test_default_connect_timeout(self) -> None:
        """Default connect_timeout is 30 seconds."""
        config = BrokerConfig()
        assert config.connect_timeout == 30

    def test_custom_connect_timeout(self) -> None:
        """Custom connect_timeout is respected."""
        config = BrokerConfig(connect_timeout=60)
        assert config.connect_timeout == 60

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_publisher_passes_timeout_to_connect(self, MockClient: MagicMock) -> None:
        """Publisher passes connect_timeout as keepalive to paho-mqtt connect."""
        mock_instance = MockClient.return_value
        config = BrokerConfig(host="broker.example.com", port=8883, connect_timeout=45)
        pub = MQTTPublisher(config)
        pub._client = mock_instance

        await pub.connect()

        mock_instance.connect.assert_called_once_with(
            "broker.example.com", 8883, keepalive=45
        )

    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_subscriber_passes_timeout_to_connect(self, MockClient: MagicMock) -> None:
        """Subscriber passes connect_timeout as keepalive to paho-mqtt connect."""
        mock_instance = MockClient.return_value
        config = BrokerConfig(host="broker.example.com", port=8883, connect_timeout=60)
        sub = MQTTSubscriber(config)
        sub._client = mock_instance

        await sub.connect()

        mock_instance.connect.assert_called_once_with(
            "broker.example.com", 8883, keepalive=60
        )

    @patch("agentic_cctv.mqtt_client.ssl.create_default_context")
    @patch("agentic_cctv.mqtt_client.paho_mqtt.Client")
    @pytest.mark.asyncio
    async def test_validate_connectivity_uses_timeout(
        self, MockClient: MagicMock, mock_ssl: MagicMock
    ) -> None:
        """validate_connectivity uses connect_timeout for CONNACK wait."""
        mock_init_client = MagicMock()
        mock_test_client = MagicMock()
        MockClient.side_effect = [mock_init_client, mock_test_client]

        # Simulate timeout — on_connect never fires
        def fake_connect(*args, **kwargs):
            pass

        def fake_loop_start():
            pass  # Don't trigger on_connect

        mock_test_client.connect = fake_connect
        mock_test_client.loop_start = fake_loop_start

        config = BrokerConfig(
            host="slow-broker.example.com",
            port=8883,
            use_tls=True,
            ca_cert="/path/ca.pem",
            connect_timeout=1,  # 1 second timeout for fast test
        )
        pub = MQTTPublisher(config)
        result = await pub.validate_connectivity()

        assert result.success is False
        assert "timed out" in result.message.lower()
