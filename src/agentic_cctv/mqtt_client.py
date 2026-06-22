"""MQTT Publisher and Subscriber for the Agentic AI CCTV Monitoring Framework.

Wraps ``paho-mqtt`` v2 with async connect/publish/subscribe/disconnect,
QoS 0/1/2 support, retained messages, TLS 1.3 configuration, automatic
reconnection with exponential backoff, and connectivity validation.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import paho.mqtt.client as paho_mqtt

from agentic_cctv.models import BrokerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connectivity validation result
# ---------------------------------------------------------------------------


@dataclass
class ConnectivityResult:
    """Result of a broker connectivity validation attempt."""

    success: bool
    message: str
    latency_ms: Optional[float] = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Topic builder
# ---------------------------------------------------------------------------


def build_topic(tenant_id: str, site_id: str, camera_id: str, suffix: str) -> str:
    """Build an MQTT topic enforcing the ``{tenant_id}/{site_id}/{camera_id}/{suffix}`` hierarchy.

    Parameters
    ----------
    tenant_id:
        Tenant identifier.  Must not be empty.
    site_id:
        Site identifier.  Must not be empty.
    camera_id:
        Camera identifier.  Must not be empty.
    suffix:
        Topic suffix (e.g. ``"events"``, ``"alerts"``, ``"health"``).
        Must not be empty.

    Returns
    -------
    str
        The fully-qualified MQTT topic string.

    Raises
    ------
    ValueError
        If any segment is empty or contains only whitespace.
    """
    segments = {
        "tenant_id": tenant_id,
        "site_id": site_id,
        "camera_id": camera_id,
        "suffix": suffix,
    }
    for name, value in segments.items():
        if not value or not value.strip():
            raise ValueError(
                f"MQTT topic segment '{name}' must not be empty, got {value!r}"
            )
    return f"{tenant_id}/{site_id}/{camera_id}/{suffix}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_tls(client: paho_mqtt.Client, config: BrokerConfig) -> None:
    """Apply TLS settings to a paho-mqtt *client* when ``config.use_tls`` is True.

    Enforces TLS 1.3 as the minimum protocol version per Requirement 10.1.
    Supports ``tls_insecure`` mode for testing with self-signed certificates.
    """
    if not config.use_tls:
        return

    ssl_context = ssl.create_default_context(cafile=config.ca_cert)

    # Enforce TLS 1.3 minimum (Requirement 10.1)
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_3

    if config.client_cert and config.client_key:
        ssl_context.load_cert_chain(
            certfile=config.client_cert,
            keyfile=config.client_key,
        )

    # Support tls_insecure mode for self-signed certs in testing
    if config.tls_insecure:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    client.tls_set_context(ssl_context)


def _configure_auth(client: paho_mqtt.Client, config: BrokerConfig) -> None:
    """Apply username/password authentication to a paho-mqtt *client*."""
    if config.username is not None:
        client.username_pw_set(config.username, config.password)


# ---------------------------------------------------------------------------
# MQTTPublisher
# ---------------------------------------------------------------------------


class MQTTPublisher:
    """Async MQTT publisher wrapping ``paho-mqtt`` v2.

    Satisfies :class:`~agentic_cctv.event_encoder.MQTTPublisherProtocol`.

    Features:
    - TLS 1.3 with optional client certificates
    - Username/password authentication
    - Automatic reconnection with exponential backoff
    - Store-and-forward queue drain on reconnection
    - Connectivity validation

    Parameters
    ----------
    broker_config:
        Connection parameters for the MQTT broker.
    store_and_forward_queue:
        Optional store-and-forward queue for draining on reconnection.
    """

    def __init__(
        self,
        broker_config: BrokerConfig,
        store_and_forward_queue: object | None = None,
    ) -> None:
        self._config = broker_config
        self._client: paho_mqtt.Client = paho_mqtt.Client(
            callback_api_version=paho_mqtt.CallbackAPIVersion.VERSION2,
        )
        _configure_tls(self._client, self._config)
        _configure_auth(self._client, self._config)
        self._connected = False
        self._store_and_forward_queue = store_and_forward_queue
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()

        # Configure reconnection backoff (1s min, 30s max)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        # Wire up disconnect callback for automatic reconnection
        self._client.on_disconnect = self._on_disconnect

    # -- callbacks ----------------------------------------------------------

    def _on_disconnect(
        self,
        client: paho_mqtt.Client,
        userdata: object,
        disconnect_flags: object,
        rc: object,
        properties: object = None,
    ) -> None:
        """Handle unexpected disconnections with automatic reconnection."""
        self._connected = False
        logger.warning(
            "MQTTPublisher disconnected from %s:%s (rc=%s)",
            self._config.host,
            self._config.port,
            rc,
        )

        # Start reconnection in a background thread
        with self._reconnect_lock:
            if not self._reconnecting:
                self._reconnecting = True
                thread = threading.Thread(
                    target=self._reconnect_loop, daemon=True
                )
                thread.start()

    def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        delay = 1
        max_delay = 30
        try:
            while not self._connected:
                try:
                    logger.info(
                        "Attempting reconnection to %s:%s (delay=%ds)",
                        self._config.host,
                        self._config.port,
                        delay,
                    )
                    self._client.reconnect()
                    self._connected = True
                    logger.info(
                        "MQTTPublisher reconnected to %s:%s",
                        self._config.host,
                        self._config.port,
                    )
                    # Drain store-and-forward queue on reconnection
                    self._drain_queue_on_reconnect()
                    break
                except Exception:
                    logger.warning(
                        "Reconnection failed, retrying in %ds", delay,
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
        finally:
            with self._reconnect_lock:
                self._reconnecting = False

    def _drain_queue_on_reconnect(self) -> None:
        """Drain the store-and-forward queue after reconnection."""
        if self._store_and_forward_queue is not None:
            try:
                drained = self._store_and_forward_queue.drain(self)
                if drained > 0:
                    logger.info(
                        "Drained %d messages from store-and-forward queue",
                        drained,
                    )
            except Exception:
                logger.exception(
                    "Error draining store-and-forward queue on reconnect"
                )

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the MQTT broker asynchronously.

        Uses the ``connect_timeout`` from the broker configuration.

        Raises
        ------
        ConnectionError
            If the broker is unreachable or the connection is refused.
        """
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.connect(
                    self._config.host,
                    self._config.port,
                    keepalive=self._config.connect_timeout,
                ),
            )
            self._client.loop_start()
            self._connected = True
            logger.info(
                "MQTTPublisher connected to %s:%s",
                self._config.host,
                self._config.port,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to MQTT broker at "
                f"{self._config.host}:{self._config.port}: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker gracefully."""
        if self._connected:
            loop = asyncio.get_running_loop()
            try:
                self._client.loop_stop()
                await loop.run_in_executor(None, self._client.disconnect)
            except Exception:
                logger.exception("Error during MQTTPublisher disconnect")
            finally:
                self._connected = False
                logger.info("MQTTPublisher disconnected")

    async def validate_connectivity(self) -> ConnectivityResult:
        """Validate connectivity to the MQTT broker.

        Attempts to connect, verifies the connection is established,
        and disconnects. Useful for validating that a remote broker
        (EMQX or Mosquitto cluster) is reachable and credentials are valid.

        Returns
        -------
        ConnectivityResult
            Result indicating success/failure with details.
        """
        test_client = paho_mqtt.Client(
            callback_api_version=paho_mqtt.CallbackAPIVersion.VERSION2,
        )
        _configure_tls(test_client, self._config)
        _configure_auth(test_client, self._config)

        connack_received = threading.Event()
        connack_rc: list = []

        def on_connect(
            client: paho_mqtt.Client,
            userdata: object,
            flags: object,
            rc: object,
            properties: object = None,
        ) -> None:
            connack_rc.append(rc)
            connack_received.set()

        test_client.on_connect = on_connect

        start_time = time.monotonic()
        try:
            test_client.connect(
                self._config.host,
                self._config.port,
                keepalive=self._config.connect_timeout,
            )
            test_client.loop_start()

            # Wait for CONNACK with timeout
            if not connack_received.wait(timeout=self._config.connect_timeout):
                test_client.loop_stop()
                test_client.disconnect()
                return ConnectivityResult(
                    success=False,
                    message=f"Connection to {self._config.host}:{self._config.port} timed out waiting for CONNACK",
                    details={"timeout_seconds": self._config.connect_timeout},
                )

            latency = (time.monotonic() - start_time) * 1000
            rc = connack_rc[0] if connack_rc else None

            test_client.loop_stop()
            test_client.disconnect()

            # paho-mqtt v2 uses ReasonCode objects
            rc_value = getattr(rc, "value", rc)
            if rc_value == 0:
                return ConnectivityResult(
                    success=True,
                    message=f"Successfully connected to {self._config.host}:{self._config.port}",
                    latency_ms=round(latency, 2),
                    details={"host": self._config.host, "port": self._config.port},
                )
            else:
                return ConnectivityResult(
                    success=False,
                    message=f"Connection refused by {self._config.host}:{self._config.port} (rc={rc})",
                    latency_ms=round(latency, 2),
                    details={"reason_code": str(rc)},
                )

        except Exception as exc:
            return ConnectivityResult(
                success=False,
                message=f"Failed to connect to {self._config.host}:{self._config.port}: {exc}",
                details={"error": str(exc)},
            )

    # -- publishing ---------------------------------------------------------

    async def publish(
        self,
        topic: str,
        payload: bytes,
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        """Publish a message to the MQTT broker.

        Parameters
        ----------
        topic:
            The MQTT topic to publish to.
        payload:
            The message payload as bytes.
        qos:
            Quality of Service level (0, 1, or 2).
        retain:
            Whether the broker should retain this message.

        Raises
        ------
        ValueError
            If *qos* is not 0, 1, or 2.
        ConnectionError
            If the publisher is not connected.
        """
        if qos not in (0, 1, 2):
            raise ValueError(f"QoS must be 0, 1, or 2, got {qos}")
        if not self._connected:
            raise ConnectionError("MQTTPublisher is not connected")

        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(
            None,
            lambda: self._client.publish(topic, payload, qos=qos, retain=retain),
        )
        # For QoS > 0, wait for the broker acknowledgement
        if qos > 0:
            await loop.run_in_executor(None, info.wait_for_publish)

        logger.debug("Published to %s (qos=%d, retain=%s)", topic, qos, retain)

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the publisher is currently connected."""
        return self._connected


# ---------------------------------------------------------------------------
# MQTTSubscriber
# ---------------------------------------------------------------------------


class MQTTSubscriber:
    """Async MQTT subscriber wrapping ``paho-mqtt`` v2.

    Features:
    - TLS 1.3 with optional client certificates
    - Username/password authentication
    - Automatic reconnection with exponential backoff
    - Re-subscription on reconnection
    - Connectivity validation

    Parameters
    ----------
    broker_config:
        Connection parameters for the MQTT broker.
    """

    def __init__(self, broker_config: BrokerConfig) -> None:
        self._config = broker_config
        self._client: paho_mqtt.Client = paho_mqtt.Client(
            callback_api_version=paho_mqtt.CallbackAPIVersion.VERSION2,
        )
        _configure_tls(self._client, self._config)
        _configure_auth(self._client, self._config)
        self._connected = False
        self._callbacks: dict[str, Callable] = {}
        self._subscriptions: dict[str, int] = {}  # topic -> qos for re-subscribe
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()

        # Configure reconnection backoff (1s min, 30s max)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        # Wire up the paho callbacks
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

    # -- internal dispatch --------------------------------------------------

    def _on_message(
        self,
        client: paho_mqtt.Client,
        userdata: object,
        message: paho_mqtt.MQTTMessage,
    ) -> None:
        """Dispatch incoming messages to registered per-topic callbacks."""
        topic = message.topic
        for pattern, callback in self._callbacks.items():
            if paho_mqtt.topic_matches_sub(pattern, topic):
                try:
                    callback(topic, message.payload, message.qos)
                except Exception:
                    logger.exception(
                        "Error in subscriber callback for topic %s", topic
                    )

    # -- callbacks ----------------------------------------------------------

    def _on_disconnect(
        self,
        client: paho_mqtt.Client,
        userdata: object,
        disconnect_flags: object,
        rc: object,
        properties: object = None,
    ) -> None:
        """Handle unexpected disconnections with automatic reconnection."""
        self._connected = False
        logger.warning(
            "MQTTSubscriber disconnected from %s:%s (rc=%s)",
            self._config.host,
            self._config.port,
            rc,
        )

        # Start reconnection in a background thread
        with self._reconnect_lock:
            if not self._reconnecting:
                self._reconnecting = True
                thread = threading.Thread(
                    target=self._reconnect_loop, daemon=True
                )
                thread.start()

    def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff and re-subscribe."""
        delay = 1
        max_delay = 30
        try:
            while not self._connected:
                try:
                    logger.info(
                        "Attempting reconnection to %s:%s (delay=%ds)",
                        self._config.host,
                        self._config.port,
                        delay,
                    )
                    self._client.reconnect()
                    self._connected = True
                    logger.info(
                        "MQTTSubscriber reconnected to %s:%s",
                        self._config.host,
                        self._config.port,
                    )
                    # Re-subscribe to all previously subscribed topics
                    self._resubscribe()
                    break
                except Exception:
                    logger.warning(
                        "Reconnection failed, retrying in %ds", delay,
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
        finally:
            with self._reconnect_lock:
                self._reconnecting = False

    def _resubscribe(self) -> None:
        """Re-subscribe to all topics after reconnection."""
        for topic, qos in self._subscriptions.items():
            try:
                self._client.subscribe(topic, qos=qos)
                logger.info("Re-subscribed to %s (qos=%d)", topic, qos)
            except Exception:
                logger.exception("Failed to re-subscribe to %s", topic)

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the MQTT broker asynchronously.

        Uses the ``connect_timeout`` from the broker configuration.

        Raises
        ------
        ConnectionError
            If the broker is unreachable or the connection is refused.
        """
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.connect(
                    self._config.host,
                    self._config.port,
                    keepalive=self._config.connect_timeout,
                ),
            )
            self._client.loop_start()
            self._connected = True
            logger.info(
                "MQTTSubscriber connected to %s:%s",
                self._config.host,
                self._config.port,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to MQTT broker at "
                f"{self._config.host}:{self._config.port}: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker gracefully."""
        if self._connected:
            loop = asyncio.get_running_loop()
            try:
                self._client.loop_stop()
                await loop.run_in_executor(None, self._client.disconnect)
            except Exception:
                logger.exception("Error during MQTTSubscriber disconnect")
            finally:
                self._connected = False
                self._callbacks.clear()
                self._subscriptions.clear()
                logger.info("MQTTSubscriber disconnected")

    async def validate_connectivity(self) -> ConnectivityResult:
        """Validate connectivity to the MQTT broker.

        Attempts to connect, verifies the connection is established,
        and disconnects. Useful for validating that a remote broker
        (EMQX or Mosquitto cluster) is reachable and credentials are valid.

        Returns
        -------
        ConnectivityResult
            Result indicating success/failure with details.
        """
        test_client = paho_mqtt.Client(
            callback_api_version=paho_mqtt.CallbackAPIVersion.VERSION2,
        )
        _configure_tls(test_client, self._config)
        _configure_auth(test_client, self._config)

        connack_received = threading.Event()
        connack_rc: list = []

        def on_connect(
            client: paho_mqtt.Client,
            userdata: object,
            flags: object,
            rc: object,
            properties: object = None,
        ) -> None:
            connack_rc.append(rc)
            connack_received.set()

        test_client.on_connect = on_connect

        start_time = time.monotonic()
        try:
            test_client.connect(
                self._config.host,
                self._config.port,
                keepalive=self._config.connect_timeout,
            )
            test_client.loop_start()

            # Wait for CONNACK with timeout
            if not connack_received.wait(timeout=self._config.connect_timeout):
                test_client.loop_stop()
                test_client.disconnect()
                return ConnectivityResult(
                    success=False,
                    message=f"Connection to {self._config.host}:{self._config.port} timed out waiting for CONNACK",
                    details={"timeout_seconds": self._config.connect_timeout},
                )

            latency = (time.monotonic() - start_time) * 1000
            rc = connack_rc[0] if connack_rc else None

            test_client.loop_stop()
            test_client.disconnect()

            # paho-mqtt v2 uses ReasonCode objects
            rc_value = getattr(rc, "value", rc)
            if rc_value == 0:
                return ConnectivityResult(
                    success=True,
                    message=f"Successfully connected to {self._config.host}:{self._config.port}",
                    latency_ms=round(latency, 2),
                    details={"host": self._config.host, "port": self._config.port},
                )
            else:
                return ConnectivityResult(
                    success=False,
                    message=f"Connection refused by {self._config.host}:{self._config.port} (rc={rc})",
                    latency_ms=round(latency, 2),
                    details={"reason_code": str(rc)},
                )

        except Exception as exc:
            return ConnectivityResult(
                success=False,
                message=f"Failed to connect to {self._config.host}:{self._config.port}: {exc}",
                details={"error": str(exc)},
            )

    # -- subscribing --------------------------------------------------------

    async def subscribe(
        self,
        topic: str,
        qos: int = 1,
        callback: Optional[Callable] = None,
    ) -> None:
        """Subscribe to an MQTT topic.

        Parameters
        ----------
        topic:
            The MQTT topic or wildcard pattern to subscribe to.
        qos:
            Quality of Service level (0, 1, or 2).
        callback:
            A callable ``(topic: str, payload: bytes, qos: int) -> None``
            invoked for each message matching *topic*.

        Raises
        ------
        ValueError
            If *qos* is not 0, 1, or 2.
        ConnectionError
            If the subscriber is not connected.
        """
        if qos not in (0, 1, 2):
            raise ValueError(f"QoS must be 0, 1, or 2, got {qos}")
        if not self._connected:
            raise ConnectionError("MQTTSubscriber is not connected")

        if callback is not None:
            self._callbacks[topic] = callback

        # Track subscription for re-subscribe on reconnection
        self._subscriptions[topic] = qos

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.subscribe(topic, qos=qos),
        )
        logger.info("Subscribed to %s (qos=%d)", topic, qos)

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the subscriber is currently connected."""
        return self._connected
