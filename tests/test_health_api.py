"""Tests for the health API endpoint (REST dashboard).

Validates Requirements 9.3 and 17.1: real-time device health dashboard
exposing Watchdog.get_device_status for all cameras, returning current
status, last heartbeat, and device metrics as JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from agentic_cctv.health_api import (
    _serialize_datetime,
    _serialize_device_health,
    _serialize_heartbeat,
    _watchdog_key,
    create_health_app,
)
from agentic_cctv.models import DeviceHealth, HeartbeatMessage
from agentic_cctv.watchdog import Watchdog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_heartbeat(
    camera_id: str = "cam-01",
    cpu: float = 45.2,
    mem: float = 62.1,
    temp: Optional[float] = 68.5,
    latency: float = 35.2,
    gpu: Optional[float] = 78.0,
) -> HeartbeatMessage:
    return HeartbeatMessage(
        camera_id=camera_id,
        tenant_id="tenant-a",
        site_id="site-1",
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        cpu_percent=cpu,
        memory_percent=mem,
        temperature_celsius=temp,
        inference_latency_ms=latency,
        gpu_utilization_percent=gpu,
    )


def _make_device_health(
    camera_id: str = "cam-01",
    status: str = "online",
    with_metrics: bool = True,
) -> DeviceHealth:
    hb = _make_heartbeat(camera_id=camera_id) if with_metrics else None
    last_hb = (
        datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        if with_metrics
        else None
    )
    return DeviceHealth(
        camera_id=camera_id,
        status=status,
        last_heartbeat=last_hb,
        metrics=hb,
    )


def _make_mock_watchdog(
    devices: Optional[List[DeviceHealth]] = None,
) -> MagicMock:
    """Create a mock Watchdog with configurable device status responses."""
    mock = MagicMock(spec=Watchdog)
    if devices is None:
        devices = []

    device_map = {d.camera_id: d for d in devices}

    def get_all() -> List[DeviceHealth]:
        return devices

    def get_one(camera_id: str) -> DeviceHealth:
        if camera_id in device_map:
            return device_map[camera_id]
        return DeviceHealth(
            camera_id=camera_id,
            status="offline",
            last_heartbeat=None,
            metrics=None,
        )

    mock.get_all_device_status.side_effect = get_all
    mock.get_device_status.side_effect = get_one
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_watchdog() -> MagicMock:
    return _make_mock_watchdog()


@pytest.fixture
def app_with_devices() -> web.Application:
    """App with two cameras: cam-01 online, cam-02 offline."""
    devices = [
        _make_device_health("cam-01", "online", with_metrics=True),
        _make_device_health("cam-02", "offline", with_metrics=False),
    ]
    mock_wd = _make_mock_watchdog(devices)
    return create_health_app(mock_wd)


@pytest.fixture
def app_empty() -> web.Application:
    """App with no tracked cameras."""
    mock_wd = _make_mock_watchdog([])
    return create_health_app(mock_wd)


# ---------------------------------------------------------------------------
# Tests: serialisation helpers
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_serialize_datetime_with_value(self) -> None:
        dt = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = _serialize_datetime(dt)
        assert result == "2025-01-15T14:30:00+00:00"

    def test_serialize_datetime_none(self) -> None:
        assert _serialize_datetime(None) is None

    def test_serialize_heartbeat_with_value(self) -> None:
        hb = _make_heartbeat()
        result = _serialize_heartbeat(hb)
        assert result is not None
        assert result["camera_id"] == "cam-01"
        assert result["cpu_percent"] == 45.2
        assert result["memory_percent"] == 62.1
        assert result["temperature_celsius"] == 68.5
        assert result["inference_latency_ms"] == 35.2
        assert result["gpu_utilization_percent"] == 78.0
        assert result["tenant_id"] == "tenant-a"
        assert result["site_id"] == "site-1"
        assert result["timestamp"] is not None

    def test_serialize_heartbeat_none(self) -> None:
        assert _serialize_heartbeat(None) is None

    def test_serialize_heartbeat_with_none_optional_fields(self) -> None:
        hb = _make_heartbeat(temp=None, gpu=None)
        result = _serialize_heartbeat(hb)
        assert result is not None
        assert result["temperature_celsius"] is None
        assert result["gpu_utilization_percent"] is None

    def test_serialize_device_health_online(self) -> None:
        dh = _make_device_health("cam-01", "online", with_metrics=True)
        result = _serialize_device_health(dh)
        assert result["camera_id"] == "cam-01"
        assert result["status"] == "online"
        assert result["last_heartbeat"] is not None
        assert result["metrics"] is not None
        assert result["metrics"]["cpu_percent"] == 45.2

    def test_serialize_device_health_offline_no_metrics(self) -> None:
        dh = _make_device_health("cam-02", "offline", with_metrics=False)
        result = _serialize_device_health(dh)
        assert result["camera_id"] == "cam-02"
        assert result["status"] == "offline"
        assert result["last_heartbeat"] is None
        assert result["metrics"] is None


# ---------------------------------------------------------------------------
# Tests: GET /api/health/devices (all devices)
# ---------------------------------------------------------------------------


class TestGetAllDevices:
    @pytest.mark.asyncio
    async def test_returns_all_devices(self, app_with_devices: web.Application) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices")
            assert resp.status == 200
            data = await resp.json()
            assert "devices" in data
            assert len(data["devices"]) == 2

            camera_ids = {d["camera_id"] for d in data["devices"]}
            assert camera_ids == {"cam-01", "cam-02"}

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_devices(
        self, app_empty: web.Application
    ) -> None:
        async with TestClient(TestServer(app_empty)) as client:
            resp = await client.get("/api/health/devices")
            assert resp.status == 200
            data = await resp.json()
            assert data["devices"] == []

    @pytest.mark.asyncio
    async def test_response_content_type_is_json(
        self, app_with_devices: web.Application
    ) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices")
            assert resp.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_online_device_has_metrics(
        self, app_with_devices: web.Application
    ) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices")
            data = await resp.json()
            online_device = next(
                d for d in data["devices"] if d["camera_id"] == "cam-01"
            )
            assert online_device["status"] == "online"
            assert online_device["last_heartbeat"] is not None
            assert online_device["metrics"] is not None
            assert "cpu_percent" in online_device["metrics"]
            assert "memory_percent" in online_device["metrics"]
            assert "inference_latency_ms" in online_device["metrics"]

    @pytest.mark.asyncio
    async def test_offline_device_has_null_metrics(
        self, app_with_devices: web.Application
    ) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices")
            data = await resp.json()
            offline_device = next(
                d for d in data["devices"] if d["camera_id"] == "cam-02"
            )
            assert offline_device["status"] == "offline"
            assert offline_device["last_heartbeat"] is None
            assert offline_device["metrics"] is None


# ---------------------------------------------------------------------------
# Tests: GET /api/health/devices/{camera_id} (single device)
# ---------------------------------------------------------------------------


class TestGetSingleDevice:
    @pytest.mark.asyncio
    async def test_returns_known_device(
        self, app_with_devices: web.Application
    ) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices/cam-01")
            assert resp.status == 200
            data = await resp.json()
            assert data["camera_id"] == "cam-01"
            assert data["status"] == "online"
            assert data["last_heartbeat"] is not None
            assert data["metrics"] is not None

    @pytest.mark.asyncio
    async def test_returns_offline_for_unknown_device(
        self, app_with_devices: web.Application
    ) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices/unknown-cam")
            assert resp.status == 200
            data = await resp.json()
            assert data["camera_id"] == "unknown-cam"
            assert data["status"] == "offline"
            assert data["last_heartbeat"] is None
            assert data["metrics"] is None

    @pytest.mark.asyncio
    async def test_response_content_type_is_json(
        self, app_with_devices: web.Application
    ) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices/cam-01")
            assert resp.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_offline_device_by_id(
        self, app_with_devices: web.Application
    ) -> None:
        async with TestClient(TestServer(app_with_devices)) as client:
            resp = await client.get("/api/health/devices/cam-02")
            assert resp.status == 200
            data = await resp.json()
            assert data["camera_id"] == "cam-02"
            assert data["status"] == "offline"


# ---------------------------------------------------------------------------
# Tests: create_health_app
# ---------------------------------------------------------------------------


class TestCreateHealthApp:
    def test_app_has_watchdog_in_state(self) -> None:
        mock_wd = _make_mock_watchdog()
        app = create_health_app(mock_wd)
        assert app[_watchdog_key] is mock_wd

    def test_app_has_routes(self) -> None:
        mock_wd = _make_mock_watchdog()
        app = create_health_app(mock_wd)
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, 'resource') and r.resource]
        assert "/api/health/devices" in routes
        assert "/api/health/devices/{camera_id}" in routes
