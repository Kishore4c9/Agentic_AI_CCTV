"""MCP Context Server for the Agentic AI CCTV Monitoring Framework.

Implements a lightweight, in-process MCP context server for v1 single-machine
mode.  The server maintains a thread-safe, in-memory shared context store
keyed by ``(camera_id, event_id)`` and exposes read/write/list/clear
operations that the ``MCPContextTool`` uses for cross-camera context sharing.

The interface is designed to be compatible with a future ``langchain-mcp-adapters``
integration for distributed multi-machine deployments — the local server can
be swapped for a remote MCP client without changing the tool layer.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """A single context entry stored in the MCP context server.

    Attributes
    ----------
    camera_id:
        The camera that produced this context.
    event_id:
        The event that this context relates to.
    context_data:
        Arbitrary context payload (scene description, threat level, etc.).
    timestamp:
        When this entry was written.
    """

    camera_id: str
    event_id: str
    context_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MCPContextServer:
    """In-process MCP context server for cross-camera state sharing.

    Maintains a thread-safe in-memory store keyed by ``(camera_id, event_id)``.
    For v1 single-machine mode this runs locally; the interface is designed
    so it can be replaced by a remote MCP client in future phases.

    All public methods are thread-safe.
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], ContextEntry] = {}
        self._lock = threading.Lock()
        logger.info("MCPContextServer initialised (in-process mode)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_context(
        self,
        camera_id: str,
        event_id: str,
        context_data: Dict[str, Any],
    ) -> None:
        """Write or update context for a ``(camera_id, event_id)`` key.

        Parameters
        ----------
        camera_id:
            The camera identifier.
        event_id:
            The event identifier.
        context_data:
            Arbitrary context payload to store.
        """
        entry = ContextEntry(
            camera_id=camera_id,
            event_id=event_id,
            context_data=context_data,
        )
        with self._lock:
            self._store[(camera_id, event_id)] = entry
        logger.debug(
            "MCPContextServer: wrote context for camera=%s event=%s",
            camera_id,
            event_id,
        )

    def read_context(
        self,
        camera_id: str,
        event_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read context for a specific ``(camera_id, event_id)`` key.

        Parameters
        ----------
        camera_id:
            The camera identifier.
        event_id:
            The event identifier.

        Returns
        -------
        Optional[Dict[str, Any]]
            The stored context data, or ``None`` if no entry exists.
        """
        with self._lock:
            entry = self._store.get((camera_id, event_id))
        if entry is None:
            return None
        return entry.context_data

    def list_contexts(
        self,
        camera_id: str,
    ) -> List[ContextEntry]:
        """List all context entries for a given camera.

        Parameters
        ----------
        camera_id:
            The camera identifier to filter by.

        Returns
        -------
        List[ContextEntry]
            All context entries belonging to the specified camera,
            ordered by timestamp (oldest first).
        """
        with self._lock:
            entries = [
                entry
                for entry in self._store.values()
                if entry.camera_id == camera_id
            ]
        entries.sort(key=lambda e: e.timestamp)
        return entries

    def list_cross_camera_contexts(
        self,
        exclude_camera_id: str,
    ) -> List[ContextEntry]:
        """List context entries from all cameras *except* the specified one.

        This is the primary method used by ``MCPContextTool`` to retrieve
        cross-camera context for multi-camera coordination.

        Parameters
        ----------
        exclude_camera_id:
            The camera to exclude (typically the current camera).

        Returns
        -------
        List[ContextEntry]
            Context entries from other cameras, ordered by timestamp
            (most recent first).
        """
        with self._lock:
            entries = [
                entry
                for entry in self._store.values()
                if entry.camera_id != exclude_camera_id
            ]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def clear_context(self, camera_id: str) -> int:
        """Remove all context entries for a given camera.

        Parameters
        ----------
        camera_id:
            The camera identifier whose entries should be removed.

        Returns
        -------
        int
            The number of entries removed.
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._store if key[0] == camera_id
            ]
            for key in keys_to_remove:
                del self._store[key]
        removed = len(keys_to_remove)
        logger.debug(
            "MCPContextServer: cleared %d entries for camera=%s",
            removed,
            camera_id,
        )
        return removed

    def size(self) -> int:
        """Return the total number of context entries in the store.

        Returns
        -------
        int
            Total entry count.
        """
        with self._lock:
            return len(self._store)
