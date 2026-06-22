"""VLM Backend implementations for the Agentic AI CCTV Monitoring Framework.

Provides a pluggable ``VLMBackend`` protocol and concrete implementations for
NVIDIA Cosmos (default v1), GPT-4o, Claude 3, and Gemini 1.5.

The ``CosmosBackend`` calls the NVIDIA Cosmos NIM VLM API using an
OpenAI-compatible chat completions endpoint.  Stub backends raise
``NotImplementedError`` for v1.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Protocol, runtime_checkable

from agentic_cctv.models import VLMConfig

logger = logging.getLogger(__name__)

# Default NVIDIA NIM endpoint
_DEFAULT_COSMOS_ENDPOINT = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# VLMBackend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VLMBackend(Protocol):
    """Protocol for VLM backend implementations.

    All backends accept a base64-encoded media (image or video) and an event
    context dict, and return a structured dict with scene understanding fields.
    """

    async def analyze(
        self, media_b64: str, event_context: dict, media_type: str = "image"
    ) -> dict:
        """Analyse media with event context and return structured results.

        Parameters
        ----------
        media_b64:
            Base64-encoded media (JPEG for image, MP4 for video).
        event_context:
            Dictionary containing event metadata (event_id, camera_id,
            object_type, confidence, etc.).
        media_type:
            "image" or "video". Determines payload format.

        Returns
        -------
        dict
            Structured response with keys: ``scene_description``,
            ``threat_level``, ``objects_identified``, ``recommended_action``,
            ``confidence``, and optionally ``embedding``.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# CosmosBackend — NVIDIA Cosmos NIM VLM API
# ---------------------------------------------------------------------------


class CosmosBackend:
    """NVIDIA Cosmos NIM VLM backend using the OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    config:
        VLM configuration containing ``api_key``, ``endpoint``, and
        ``timeout_seconds``.
    """

    def __init__(self, config: VLMConfig) -> None:
        self._api_key = config.api_key
        self._endpoint = config.endpoint or _DEFAULT_COSMOS_ENDPOINT
        self._timeout = config.timeout_seconds

    async def analyze(
        self, media_b64: str, event_context: dict, media_type: str = "image"
    ) -> dict:
        """Call the NVIDIA Cosmos NIM VLM API and return structured results.

        Sends a chat completions request with the event context as a system
        message and the base64 media as a user content message.

        Parameters
        ----------
        media_b64:
            Base64-encoded media (JPEG for image, MP4 for video).
        event_context:
            Dictionary containing event metadata.
        media_type:
            "image" or "video". Determines payload format.

        Returns
        -------
        dict
            Parsed JSON response from the VLM.

        Raises
        ------
        RuntimeError
            If the API call fails or the response cannot be parsed.
        """
        url = f"{self._endpoint.rstrip('/')}/chat/completions"

        system_prompt = (
            "You are a CCTV scene analysis AI. Analyse the provided image and event context. "
            "Return a JSON object with these exact fields:\n"
            '- "scene_description": string describing the scene\n'
            '- "threat_level": one of "none", "low", "medium", "high", "critical"\n'
            '- "objects_identified": list of objects, each with "type", "action", "location"\n'
            '- "recommended_action": one of "alert", "log", "summarise", "escalate"\n'
            '- "confidence": float between 0 and 1\n'
            "Return ONLY valid JSON, no markdown or extra text."
        )

        # Build content block based on media_type
        if media_type == "video":
            media_content = {
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{media_b64}",
                },
            }
        else:
            media_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{media_b64}",
                },
            }

        payload: dict[str, Any] = {
            "model": "nvidia/cosmos",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Event context: {json.dumps(event_context)}",
                        },
                        media_content,
                    ],
                },
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }

        request_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        req = urllib.request.Request(url, data=request_body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            logger.error("Cosmos API call failed: %s", exc)
            raise RuntimeError(f"Cosmos API call failed: {exc}") from exc

        # Parse the assistant's message content as JSON
        try:
            content = response_data["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse Cosmos response: %s", exc)
            raise RuntimeError(f"Failed to parse Cosmos response: {exc}") from exc

        return result


# ---------------------------------------------------------------------------
# Stub Backends (v1 — not implemented)
# ---------------------------------------------------------------------------


class GPT4oBackend:
    """GPT-4o VLM backend stub.  Not implemented in v1."""

    async def analyze(
        self, media_b64: str, event_context: dict, media_type: str = "image"
    ) -> dict:
        """Raise NotImplementedError — GPT-4o backend is not available in v1."""
        raise NotImplementedError("GPT-4o backend not implemented in v1")


class Claude3Backend:
    """Claude 3 VLM backend stub.  Not implemented in v1."""

    async def analyze(
        self, media_b64: str, event_context: dict, media_type: str = "image"
    ) -> dict:
        """Raise NotImplementedError — Claude 3 backend is not available in v1."""
        raise NotImplementedError("Claude 3 backend not implemented in v1")


class Gemini15Backend:
    """Gemini 1.5 VLM backend stub.  Not implemented in v1."""

    async def analyze(
        self, media_b64: str, event_context: dict, media_type: str = "image"
    ) -> dict:
        """Raise NotImplementedError — Gemini 1.5 backend is not available in v1."""
        raise NotImplementedError("Gemini 1.5 backend not implemented in v1")
