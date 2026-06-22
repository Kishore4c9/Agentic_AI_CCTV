"""Vector DB wrapper for the Agentic AI CCTV Monitoring Framework.

Provides a ``VectorDB`` class wrapping ChromaDB for persistent vector storage
of VLM embeddings keyed by event ID.

ChromaDB is an optional dependency — if not installed, the class raises
``ImportError`` on instantiation with a helpful message.

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VectorDB:
    """ChromaDB-backed vector database for VLM embedding storage.

    Parameters
    ----------
    path:
        Path to the persistent storage directory for ChromaDB.
    """

    def __init__(self, path: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for VectorDB. "
                "Install it with: pip install chromadb==1.0.7"
            ) from exc

        self._path = path
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name="vlm_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorDB initialised at %s", path)

    def store_embedding(
        self,
        event_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> None:
        """Store a VLM embedding with associated metadata.

        Parameters
        ----------
        event_id:
            Unique event identifier used as the document ID.
        embedding:
            The embedding vector as a list of floats.
        metadata:
            Additional metadata to store alongside the embedding.
        tenant_id:
            Optional tenant identifier.  If provided, it is added to the
            metadata so that tenant-scoped searches can filter by it.
            If ``tenant_id`` is already present in *metadata*, this
            parameter is ignored.
        """
        # Ensure tenant_id is always in metadata for tenant isolation
        if tenant_id is not None and "tenant_id" not in metadata:
            metadata = {**metadata, "tenant_id": tenant_id}

        # ChromaDB requires metadata values to be str, int, float, or bool
        safe_metadata = {
            k: v
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
        self._collection.upsert(
            ids=[event_id],
            embeddings=[embedding],
            metadatas=[safe_metadata],
        )
        logger.debug("Stored embedding for event %s", event_id)

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        tenant_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar embeddings, optionally scoped by tenant.

        Parameters
        ----------
        query_embedding:
            The query embedding vector.
        n_results:
            Maximum number of results to return.
        tenant_id:
            If provided, only return embeddings belonging to this tenant.
            Uses ChromaDB's ``where`` clause for server-side filtering.

        Returns
        -------
        list[dict]
            List of result dicts with ``id``, ``distance``, and ``metadata``.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if tenant_id is not None:
            kwargs["where"] = {"tenant_id": tenant_id}

        results = self._collection.query(**kwargs)

        output: list[dict[str, Any]] = []
        if results and results["ids"]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [None] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            for i, doc_id in enumerate(ids):
                output.append(
                    {
                        "id": doc_id,
                        "distance": distances[i],
                        "metadata": metadatas[i],
                    }
                )
        return output

    def purge_orphaned_embeddings(self, valid_event_ids: set[str]) -> int:
        """Delete embeddings whose event IDs are no longer in the TimeSeriesDB.

        This implements the lifecycle-link between VectorDB embeddings and
        TimeSeriesDB records: when raw events are purged, their corresponding
        embeddings should also be removed.

        Parameters
        ----------
        valid_event_ids:
            Set of event IDs that still exist in the TimeSeriesDB.

        Returns
        -------
        int
            Number of orphaned embeddings deleted.
        """
        # Get all embedding IDs currently stored
        all_ids = self._collection.get()["ids"]
        if not all_ids:
            return 0

        orphaned_ids = [eid for eid in all_ids if eid not in valid_event_ids]
        if not orphaned_ids:
            return 0

        self._collection.delete(ids=orphaned_ids)
        logger.info("Purged %d orphaned embeddings from VectorDB.", len(orphaned_ids))
        return len(orphaned_ids)

    def get_all_ids(self) -> list[str]:
        """Return all embedding IDs currently stored.

        Returns
        -------
        list[str]
            List of all embedding document IDs.
        """
        return self._collection.get()["ids"]
