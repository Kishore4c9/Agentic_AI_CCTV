"""Property-based tests for VLM Backend payload format.

Feature: vlm-video-snippet, Property 7: VLM Backend Payload Format Matches Media Type
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, strategies as st

from agentic_cctv.models import VLMConfig
from agentic_cctv.vlm_backends import CosmosBackend


# ---------------------------------------------------------------------------
# Property 7: Backend Payload Format Matches Media Type
# ---------------------------------------------------------------------------


@settings(max_examples=20)
@given(
    media_b64=st.text(
        alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="),
        min_size=4,
        max_size=50,
    ),
    media_type=st.sampled_from(["image", "video"]),
)
def test_backend_payload_format_matches_media_type(
    media_b64: str, media_type: str
) -> None:
    """**Validates: Requirements 5.1, 5.2, 4.3**

    For any base64 string and media_type in {"image", "video"}, the CosmosBackend
    constructs a payload where: media_type="image" → content type is "image_url"
    with data:image/jpeg;base64,...; media_type="video" → content type is "video_url"
    with data:video/mp4;base64,...
    """
    config = VLMConfig(api_key="test-key", endpoint="https://test.api.com/v1")
    backend = CosmosBackend(config)

    # We need to capture the payload that would be sent.
    # We'll mock urllib.request.Request to capture the payload.
    captured_payloads = []

    def mock_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        # Return a mock response
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "scene_description": "test",
                        "threat_level": "none",
                        "objects_identified": [],
                        "recommended_action": "log",
                        "confidence": 0.5,
                    })
                }
            }]
        }).encode("utf-8")
        return mock_resp

    import asyncio

    with patch("agentic_cctv.vlm_backends.urllib.request.urlopen", side_effect=mock_urlopen):
        result = asyncio.get_event_loop().run_until_complete(
            backend.analyze(media_b64, {"event_id": "test"}, media_type=media_type)
        )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]

    # Find the media content block in the user message
    user_msg = payload["messages"][1]
    content_blocks = user_msg["content"]

    # The second content block should be the media block
    media_block = content_blocks[1]

    if media_type == "image":
        assert media_block["type"] == "image_url"
        assert media_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert media_b64 in media_block["image_url"]["url"]
    else:
        assert media_block["type"] == "video_url"
        assert media_block["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert media_b64 in media_block["video_url"]["url"]
