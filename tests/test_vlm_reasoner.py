"""Unit tests for VLMReasoner, VLM backends, and schema validation.

Tests retry/fallback behaviour with mocked backend failures, schema validation
edge cases, VectorDB embedding storage, and failure logging to TimeSeriesDB.

Requirements: 5.5, 5.6, 5.7
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_cctv.models import (
    BoundingBox,
    IdentifiedObject,
    SceneUnderstanding,
    StructuredEvent,
    VLMConfig,
)
from agentic_cctv.vlm_backends import (
    CosmosBackend,
    GPT4oBackend,
    Claude3Backend,
    Gemini15Backend,
)
from agentic_cctv.vlm_reasoner import (
    VLMReasoner,
    validate_vlm_response,
    VALID_THREAT_LEVELS,
    VALID_ACTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event(
    event_id: str = "evt-001",
    confidence: float = 0.85,
    object_type: str = "person",
) -> StructuredEvent:
    """Create a test StructuredEvent."""
    return StructuredEvent(
        event_id=event_id,
        camera_id="cam-01",
        tenant_id="tenant-01",
        site_id="site-01",
        timestamp=datetime(2025, 1, 15, 14, 30, 0),
        object_type=object_type,
        track_id="trk-001",
        bounding_box=BoundingBox(x=100, y=100, width=200, height=300),
        confidence=confidence,
        frame_crop="dGVzdA==",  # base64 "test"
    )


def _valid_vlm_response() -> dict:
    """Return a valid VLM response dict."""
    return {
        "scene_description": "A person walking near the entrance.",
        "threat_level": "low",
        "objects_identified": [
            {"type": "person", "action": "walking", "location": "entrance"},
        ],
        "recommended_action": "log",
        "confidence": 0.85,
    }


def _valid_vlm_response_with_embedding() -> dict:
    """Return a valid VLM response dict with an embedding."""
    resp = _valid_vlm_response()
    resp["embedding"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    return resp


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------


class TestValidateVLMResponse:
    """Test the validate_vlm_response function."""

    def test_valid_response_passes(self) -> None:
        """A complete valid response passes validation."""
        assert validate_vlm_response(_valid_vlm_response()) is True

    def test_missing_scene_description_fails(self) -> None:
        """Missing scene_description fails validation."""
        resp = _valid_vlm_response()
        del resp["scene_description"]
        assert validate_vlm_response(resp) is False

    def test_missing_threat_level_fails(self) -> None:
        """Missing threat_level fails validation."""
        resp = _valid_vlm_response()
        del resp["threat_level"]
        assert validate_vlm_response(resp) is False

    def test_missing_objects_identified_fails(self) -> None:
        """Missing objects_identified fails validation."""
        resp = _valid_vlm_response()
        del resp["objects_identified"]
        assert validate_vlm_response(resp) is False

    def test_missing_recommended_action_fails(self) -> None:
        """Missing recommended_action fails validation."""
        resp = _valid_vlm_response()
        del resp["recommended_action"]
        assert validate_vlm_response(resp) is False

    def test_missing_confidence_fails(self) -> None:
        """Missing confidence fails validation."""
        resp = _valid_vlm_response()
        del resp["confidence"]
        assert validate_vlm_response(resp) is False

    def test_wrong_type_scene_description_fails(self) -> None:
        """Non-string scene_description fails validation."""
        resp = _valid_vlm_response()
        resp["scene_description"] = 42
        assert validate_vlm_response(resp) is False

    def test_wrong_type_objects_identified_fails(self) -> None:
        """Non-list objects_identified fails validation."""
        resp = _valid_vlm_response()
        resp["objects_identified"] = "not a list"
        assert validate_vlm_response(resp) is False

    def test_confidence_out_of_range_high_fails(self) -> None:
        """Confidence > 1 fails validation."""
        resp = _valid_vlm_response()
        resp["confidence"] = 1.5
        assert validate_vlm_response(resp) is False

    def test_confidence_out_of_range_low_fails(self) -> None:
        """Confidence < 0 fails validation."""
        resp = _valid_vlm_response()
        resp["confidence"] = -0.1
        assert validate_vlm_response(resp) is False

    def test_invalid_threat_level_fails(self) -> None:
        """Invalid threat_level string fails validation."""
        resp = _valid_vlm_response()
        resp["threat_level"] = "extreme"
        assert validate_vlm_response(resp) is False

    def test_invalid_recommended_action_fails(self) -> None:
        """Invalid recommended_action string fails validation."""
        resp = _valid_vlm_response()
        resp["recommended_action"] = "ignore"
        assert validate_vlm_response(resp) is False

    def test_confidence_as_int_passes(self) -> None:
        """Integer confidence (0 or 1) passes validation."""
        resp = _valid_vlm_response()
        resp["confidence"] = 1
        assert validate_vlm_response(resp) is True
        resp["confidence"] = 0
        assert validate_vlm_response(resp) is True

    def test_confidence_as_bool_fails(self) -> None:
        """Boolean confidence fails validation (bool is subclass of int)."""
        resp = _valid_vlm_response()
        resp["confidence"] = True
        assert validate_vlm_response(resp) is False

    def test_empty_dict_fails(self) -> None:
        """Empty dict fails validation."""
        assert validate_vlm_response({}) is False

    def test_non_dict_fails(self) -> None:
        """Non-dict input fails validation."""
        assert validate_vlm_response("not a dict") is False  # type: ignore[arg-type]
        assert validate_vlm_response([]) is False  # type: ignore[arg-type]
        assert validate_vlm_response(None) is False  # type: ignore[arg-type]

    def test_all_valid_threat_levels(self) -> None:
        """All valid threat levels pass validation."""
        for level in VALID_THREAT_LEVELS:
            resp = _valid_vlm_response()
            resp["threat_level"] = level
            assert validate_vlm_response(resp) is True

    def test_all_valid_actions(self) -> None:
        """All valid recommended actions pass validation."""
        for action in VALID_ACTIONS:
            resp = _valid_vlm_response()
            resp["recommended_action"] = action
            assert validate_vlm_response(resp) is True

    def test_confidence_boundary_zero(self) -> None:
        """Confidence exactly 0.0 passes."""
        resp = _valid_vlm_response()
        resp["confidence"] = 0.0
        assert validate_vlm_response(resp) is True

    def test_confidence_boundary_one(self) -> None:
        """Confidence exactly 1.0 passes."""
        resp = _valid_vlm_response()
        resp["confidence"] = 1.0
        assert validate_vlm_response(resp) is True

    def test_empty_objects_list_passes(self) -> None:
        """Empty objects_identified list passes validation."""
        resp = _valid_vlm_response()
        resp["objects_identified"] = []
        assert validate_vlm_response(resp) is True


# ---------------------------------------------------------------------------
# VLMReasoner Tests — Retry/Fallback Behaviour
# ---------------------------------------------------------------------------


class TestVLMReasonerRetryFallback:
    """Test VLMReasoner retry and fallback logic."""

    @pytest.mark.asyncio
    async def test_successful_vlm_call(self) -> None:
        """Successful VLM call returns valid SceneUnderstanding."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response()

        reasoner = VLMReasoner(backend=backend)
        event = _make_event()
        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        assert result.event_id == event.event_id
        assert result.scene_description == "A person walking near the entrance."
        assert result.threat_level == "low"
        assert result.recommended_action == "log"
        assert result.confidence == 0.85
        backend.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_failure_retry_success(self) -> None:
        """First failure triggers retry; success on retry returns valid result."""
        backend = AsyncMock()
        backend.analyze.side_effect = [
            RuntimeError("API timeout"),
            _valid_vlm_response(),
        ]

        reasoner = VLMReasoner(backend=backend)
        event = _make_event()
        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        assert result.event_id == event.event_id
        assert result.threat_level == "low"
        assert backend.analyze.call_count == 2

    @pytest.mark.asyncio
    async def test_two_failures_fallback(self) -> None:
        """Two failures trigger fallback to rule-based classification."""
        backend = AsyncMock()
        backend.analyze.side_effect = RuntimeError("API down")

        reasoner = VLMReasoner(backend=backend)
        event = _make_event(confidence=0.85)
        result = await reasoner.reason(event, matched_rules=["rule-001"])

        assert isinstance(result, SceneUnderstanding)
        assert result.event_id == event.event_id
        assert result.recommended_action == "log"
        assert result.raw_response.get("fallback") is True
        assert result.raw_response.get("matched_rules") == ["rule-001"]
        assert backend.analyze.call_count == 2

    @pytest.mark.asyncio
    async def test_schema_validation_rejects_invalid_triggers_retry(self) -> None:
        """Invalid schema on first call triggers retry."""
        invalid_response = {"scene_description": "test"}  # missing fields
        valid_response = _valid_vlm_response()

        backend = AsyncMock()
        backend.analyze.side_effect = [invalid_response, valid_response]

        reasoner = VLMReasoner(backend=backend)
        event = _make_event()
        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        assert result.threat_level == "low"
        assert backend.analyze.call_count == 2

    @pytest.mark.asyncio
    async def test_schema_validation_both_invalid_triggers_fallback(self) -> None:
        """Invalid schema on both calls triggers fallback."""
        invalid_response = {"bad": "data"}

        backend = AsyncMock()
        backend.analyze.return_value = invalid_response

        reasoner = VLMReasoner(backend=backend)
        event = _make_event(confidence=0.95)
        result = await reasoner.reason(event)

        assert result.raw_response.get("fallback") is True
        assert result.threat_level == "high"  # confidence 0.95 → high
        assert backend.analyze.call_count == 2

    @pytest.mark.asyncio
    async def test_vlm_embedding_storage_in_vector_db(self) -> None:
        """VLM embedding is stored in VectorDB when available."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response_with_embedding()

        vector_db = MagicMock()
        reasoner = VLMReasoner(backend=backend, vector_db=vector_db)
        event = _make_event()
        result = await reasoner.reason(event)

        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        vector_db.store_embedding.assert_called_once()
        call_args = vector_db.store_embedding.call_args
        assert call_args.kwargs["event_id"] == event.event_id
        assert call_args.kwargs["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_vlm_reasoner_works_without_vector_db(self) -> None:
        """VLMReasoner works correctly when VectorDB is None."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response_with_embedding()

        reasoner = VLMReasoner(backend=backend, vector_db=None)
        event = _make_event()
        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_failure_logging_to_timeseries_db(self) -> None:
        """VLM failures are logged to TimeSeriesDB."""
        backend = AsyncMock()
        backend.analyze.side_effect = RuntimeError("API error")

        timeseries_db = MagicMock()
        reasoner = VLMReasoner(backend=backend, timeseries_db=timeseries_db)
        event = _make_event()
        await reasoner.reason(event)

        # Two failures → two log calls
        assert timeseries_db.insert_event.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_threat_level_high_confidence(self) -> None:
        """Fallback with confidence >= 0.9 produces 'high' threat level."""
        backend = AsyncMock()
        backend.analyze.side_effect = RuntimeError("fail")

        reasoner = VLMReasoner(backend=backend)
        event = _make_event(confidence=0.95)
        result = await reasoner.reason(event)

        assert result.threat_level == "high"

    @pytest.mark.asyncio
    async def test_fallback_threat_level_medium_confidence(self) -> None:
        """Fallback with confidence >= 0.7 produces 'medium' threat level."""
        backend = AsyncMock()
        backend.analyze.side_effect = RuntimeError("fail")

        reasoner = VLMReasoner(backend=backend)
        event = _make_event(confidence=0.75)
        result = await reasoner.reason(event)

        assert result.threat_level == "medium"

    @pytest.mark.asyncio
    async def test_fallback_threat_level_low_confidence(self) -> None:
        """Fallback with confidence >= 0.5 produces 'low' threat level."""
        backend = AsyncMock()
        backend.analyze.side_effect = RuntimeError("fail")

        reasoner = VLMReasoner(backend=backend)
        event = _make_event(confidence=0.55)
        result = await reasoner.reason(event)

        assert result.threat_level == "low"

    @pytest.mark.asyncio
    async def test_fallback_threat_level_none_confidence(self) -> None:
        """Fallback with confidence < 0.5 produces 'none' threat level."""
        backend = AsyncMock()
        backend.analyze.side_effect = RuntimeError("fail")

        reasoner = VLMReasoner(backend=backend)
        event = _make_event(confidence=0.3)
        result = await reasoner.reason(event)

        assert result.threat_level == "none"

    @pytest.mark.asyncio
    async def test_fallback_without_matched_rules(self) -> None:
        """Fallback without matched_rules still works."""
        backend = AsyncMock()
        backend.analyze.side_effect = RuntimeError("fail")

        reasoner = VLMReasoner(backend=backend)
        event = _make_event()
        result = await reasoner.reason(event)

        assert result.raw_response.get("matched_rules") == []

    @pytest.mark.asyncio
    async def test_vector_db_error_does_not_break_reasoning(self) -> None:
        """VectorDB storage error does not prevent SceneUnderstanding return."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response_with_embedding()

        vector_db = MagicMock()
        vector_db.store_embedding.side_effect = RuntimeError("ChromaDB error")

        reasoner = VLMReasoner(backend=backend, vector_db=vector_db)
        event = _make_event()
        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        assert result.threat_level == "low"


# ---------------------------------------------------------------------------
# Stub Backend Tests
# ---------------------------------------------------------------------------


class TestStubBackends:
    """Test that stub backends raise NotImplementedError."""

    @pytest.mark.asyncio
    async def test_gpt4o_backend_raises(self) -> None:
        """GPT4oBackend raises NotImplementedError."""
        backend = GPT4oBackend()
        with pytest.raises(NotImplementedError, match="GPT-4o"):
            await backend.analyze("img", {})

    @pytest.mark.asyncio
    async def test_claude3_backend_raises(self) -> None:
        """Claude3Backend raises NotImplementedError."""
        backend = Claude3Backend()
        with pytest.raises(NotImplementedError, match="Claude 3"):
            await backend.analyze("img", {})

    @pytest.mark.asyncio
    async def test_gemini15_backend_raises(self) -> None:
        """Gemini15Backend raises NotImplementedError."""
        backend = Gemini15Backend()
        with pytest.raises(NotImplementedError, match="Gemini 1.5"):
            await backend.analyze("img", {})


# ---------------------------------------------------------------------------
# CosmosBackend Tests
# ---------------------------------------------------------------------------


class TestCosmosBackend:
    """Test CosmosBackend configuration."""

    def test_default_endpoint(self) -> None:
        """CosmosBackend uses default endpoint when none specified."""
        config = VLMConfig(api_key="test-key")
        backend = CosmosBackend(config)
        assert backend._endpoint == "https://integrate.api.nvidia.com/v1"

    def test_custom_endpoint(self) -> None:
        """CosmosBackend uses custom endpoint when specified."""
        config = VLMConfig(api_key="test-key", endpoint="https://custom.api.com/v1")
        backend = CosmosBackend(config)
        assert backend._endpoint == "https://custom.api.com/v1"

    def test_timeout_from_config(self) -> None:
        """CosmosBackend uses timeout from config."""
        config = VLMConfig(api_key="test-key", timeout_seconds=60)
        backend = CosmosBackend(config)
        assert backend._timeout == 60


# ---------------------------------------------------------------------------
# VLMReasoner Video Dispatch Tests
# ---------------------------------------------------------------------------

from agentic_cctv.models import CameraConfig


def _make_camera_config(
    camera_id: str = "cam-01",
    vlm_input_mode: str = "image",
) -> CameraConfig:
    return CameraConfig(
        camera_id=camera_id,
        uri="rtsp://test:554/stream",
        tenant_id="tenant-01",
        site_id="site-01",
        confidence_threshold=0.7,
        vlm_input_mode=vlm_input_mode,
        vlm_video_duration_seconds=10,
    )


class TestVLMReasonerVideoDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_video_snippet_when_video_mode(self) -> None:
        """VLMReasoner dispatches video snippet when vlm_input_mode='video' and video_snippet present."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response()

        camera_configs = {"cam-01": _make_camera_config(vlm_input_mode="video")}
        reasoner = VLMReasoner(backend=backend, camera_configs=camera_configs)

        event = _make_event()
        event.video_snippet = "dmlkZW9fZGF0YQ=="  # base64 "video_data"

        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        # Verify the backend was called with video_snippet, not frame_crop
        call_args = backend.analyze.call_args
        assert call_args[0][0] == "dmlkZW9fZGF0YQ=="
        assert call_args[1]["media_type"] == "video"

    @pytest.mark.asyncio
    async def test_falls_back_to_frame_crop_when_video_snippet_none(self) -> None:
        """VLMReasoner falls back to frame_crop when video_snippet is None."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response()

        camera_configs = {"cam-01": _make_camera_config(vlm_input_mode="video")}
        reasoner = VLMReasoner(backend=backend, camera_configs=camera_configs)

        event = _make_event()
        # video_snippet is None by default

        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        call_args = backend.analyze.call_args
        assert call_args[0][0] == event.frame_crop
        assert call_args[1]["media_type"] == "image"

    @pytest.mark.asyncio
    async def test_falls_back_to_frame_crop_when_camera_config_not_found(self) -> None:
        """VLMReasoner falls back to frame_crop when camera config not found."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response()

        # No camera configs provided
        reasoner = VLMReasoner(backend=backend, camera_configs={})

        event = _make_event()
        event.video_snippet = "dmlkZW9fZGF0YQ=="

        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        call_args = backend.analyze.call_args
        assert call_args[0][0] == event.frame_crop
        assert call_args[1]["media_type"] == "image"

    @pytest.mark.asyncio
    async def test_image_mode_uses_frame_crop(self) -> None:
        """VLMReasoner uses frame_crop when vlm_input_mode='image'."""
        backend = AsyncMock()
        backend.analyze.return_value = _valid_vlm_response()

        camera_configs = {"cam-01": _make_camera_config(vlm_input_mode="image")}
        reasoner = VLMReasoner(backend=backend, camera_configs=camera_configs)

        event = _make_event()
        result = await reasoner.reason(event)

        assert isinstance(result, SceneUnderstanding)
        call_args = backend.analyze.call_args
        assert call_args[0][0] == event.frame_crop
        assert call_args[1]["media_type"] == "image"


# ---------------------------------------------------------------------------
# CosmosBackend Payload Format Tests
# ---------------------------------------------------------------------------

import json
from unittest.mock import patch


class TestCosmosBackendPayloadFormat:
    @pytest.mark.asyncio
    async def test_image_payload_format(self) -> None:
        """CosmosBackend image payload uses image_url structure."""
        config = VLMConfig(api_key="test-key", endpoint="https://test.api.com/v1")
        backend = CosmosBackend(config)

        captured = []

        def mock_urlopen(req, timeout=None):
            captured.append(json.loads(req.data.decode("utf-8")))
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": json.dumps(_valid_vlm_response())}}]
            }).encode("utf-8")
            return mock_resp

        with patch("agentic_cctv.vlm_backends.urllib.request.urlopen", side_effect=mock_urlopen):
            await backend.analyze("aW1hZ2VfZGF0YQ==", {"event_id": "test"}, media_type="image")

        payload = captured[0]
        media_block = payload["messages"][1]["content"][1]
        assert media_block["type"] == "image_url"
        assert "data:image/jpeg;base64," in media_block["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_video_payload_format(self) -> None:
        """CosmosBackend video payload uses video_url structure."""
        config = VLMConfig(api_key="test-key", endpoint="https://test.api.com/v1")
        backend = CosmosBackend(config)

        captured = []

        def mock_urlopen(req, timeout=None):
            captured.append(json.loads(req.data.decode("utf-8")))
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": json.dumps(_valid_vlm_response())}}]
            }).encode("utf-8")
            return mock_resp

        with patch("agentic_cctv.vlm_backends.urllib.request.urlopen", side_effect=mock_urlopen):
            await backend.analyze("dmlkZW9fZGF0YQ==", {"event_id": "test"}, media_type="video")

        payload = captured[0]
        media_block = payload["messages"][1]["content"][1]
        assert media_block["type"] == "video_url"
        assert "data:video/mp4;base64," in media_block["video_url"]["url"]
