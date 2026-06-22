"""Frame Crop Store for the Agentic AI CCTV Monitoring Framework.

Stores JPEG crops encrypted with AES-256-GCM at rest, auto-deletes after
72 hours, and provides access via time-limited pre-signed URLs (local
file-based for v1).

Uses ``from __future__ import annotations`` for Python 3.9 compatibility.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import parse_qs, urlparse

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NONCE_SIZE = 12  # 96-bit nonce for AES-GCM
_KEY_SIZE = 32  # 256-bit key for AES-256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_key(key_hex: str) -> bytes:
    """Validate and decode a hex-encoded AES-256 key.

    Parameters
    ----------
    key_hex:
        Hex-encoded 32-byte key string.

    Returns
    -------
    bytes
        The raw 32-byte key.

    Raises
    ------
    ValueError
        If the key is empty, not valid hex, or not 32 bytes.
    """
    if not key_hex:
        raise ValueError(
            "Encryption key is missing. "
            "Set 'storage.frame_crop_encryption_key' in config.yaml. "
            "Do not store unencrypted crops."
        )
    try:
        raw = bytes.fromhex(key_hex)
    except ValueError:
        raise ValueError(
            "Encryption key is not valid hex. "
            "Provide a 64-character hex string (32 bytes / 256 bits)."
        )
    if len(raw) != _KEY_SIZE:
        raise ValueError(
            f"Encryption key must be exactly {_KEY_SIZE} bytes "
            f"({_KEY_SIZE * 2} hex chars), got {len(raw)} bytes."
        )
    return raw


# ---------------------------------------------------------------------------
# Encryption / Decryption
# ---------------------------------------------------------------------------


def encrypt_crop(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt *plaintext* with AES-256-GCM.

    Returns ``nonce || ciphertext`` (nonce is prepended).

    Parameters
    ----------
    plaintext:
        The raw bytes to encrypt.
    key:
        A 32-byte AES-256 key.

    Returns
    -------
    bytes
        ``nonce (12 bytes) || ciphertext+tag``.
    """
    nonce = secrets.token_bytes(_NONCE_SIZE)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ct


def decrypt_crop(blob: bytes, key: bytes) -> bytes:
    """Decrypt a blob produced by :func:`encrypt_crop`.

    Parameters
    ----------
    blob:
        ``nonce (12 bytes) || ciphertext+tag`` as returned by
        :func:`encrypt_crop`.
    key:
        The same 32-byte AES-256 key used for encryption.

    Returns
    -------
    bytes
        The original plaintext.

    Raises
    ------
    ValueError
        If the blob is too short or decryption fails (wrong key / tampered).
    """
    if len(blob) <= _NONCE_SIZE:
        raise ValueError("Encrypted blob is too short to contain a nonce.")
    nonce = blob[:_NONCE_SIZE]
    ct = blob[_NONCE_SIZE:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None)


# ---------------------------------------------------------------------------
# Pre-signed URL helpers
# ---------------------------------------------------------------------------


@dataclass
class PreSignedURL:
    """A time-limited pre-signed URL for accessing a stored crop."""

    url: str
    expires_at: datetime


def generate_presigned_url(
    file_path: str,
    key: bytes,
    expires_seconds: int = 3600,
) -> PreSignedURL:
    """Generate a local ``file://`` pre-signed URL with HMAC token.

    Parameters
    ----------
    file_path:
        Absolute path to the encrypted crop file.
    key:
        The AES-256 key (reused as HMAC key for simplicity in v1).
    expires_seconds:
        Lifetime of the URL in seconds (default 1 hour).

    Returns
    -------
    PreSignedURL
        The URL string and its expiry datetime.
    """
    expires_ts = int(time.time()) + expires_seconds
    # Normalise to forward slashes for consistent HMAC
    norm_path = file_path.replace("\\", "/")
    message = f"{norm_path}:{expires_ts}".encode()
    token = hmac.new(key, message, hashlib.sha256).hexdigest()
    # Encode the normalised path into the URL
    url = f"file://{norm_path}?token={token}&expires={expires_ts}"
    return PreSignedURL(url=url, expires_at=datetime.utcfromtimestamp(expires_ts))


def validate_presigned_url(url: str, key: bytes) -> Optional[str]:
    """Validate a pre-signed URL and return the file path if valid.

    Parameters
    ----------
    url:
        The pre-signed URL to validate.
    key:
        The AES-256 key used to generate the HMAC.

    Returns
    -------
    Optional[str]
        The file path if the URL is valid and not expired, else ``None``.
    """
    try:
        # Split on '?' to separate path from query string
        if "?" not in url:
            return None

        base, query_str = url.split("?", 1)

        # Extract the file path from the URL (strip 'file://' prefix)
        if base.startswith("file://"):
            raw_path = base[len("file://"):]
        else:
            return None

        qs = parse_qs(query_str)
        token = qs.get("token", [None])[0]
        expires_str = qs.get("expires", [None])[0]

        if token is None or expires_str is None:
            return None

        expires_ts = int(expires_str)
        if int(time.time()) > expires_ts:
            return None

        message = f"{raw_path}:{expires_ts}".encode()
        expected = hmac.new(key, message, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(token, expected):
            return None

        return raw_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FrameCropStore
# ---------------------------------------------------------------------------


class FrameCropStore:
    """Encrypted frame crop storage with retention and pre-signed URL access.

    Parameters
    ----------
    crop_path:
        Directory where encrypted crop files are stored.
    encryption_key_hex:
        Hex-encoded 32-byte AES-256 key.
    retention_hours:
        Number of hours to retain crops before auto-deletion (default 72).
    url_expiry_seconds:
        Lifetime of pre-signed URLs in seconds (default 3600 = 1 hour).

    Raises
    ------
    ValueError
        If the encryption key is missing or invalid.
    """

    def __init__(
        self,
        crop_path: str = "./data/crops",
        encryption_key_hex: str = "",
        retention_hours: int = 72,
        url_expiry_seconds: int = 3600,
    ) -> None:
        self._key = _validate_key(encryption_key_hex)
        self._crop_path = os.path.abspath(crop_path)
        self._retention_hours = retention_hours
        self._url_expiry_seconds = url_expiry_seconds
        self._lock = threading.Lock()

        # Ensure crop directory exists
        os.makedirs(self._crop_path, exist_ok=True)

        # SQLite metadata database
        db_path = os.path.join(self._crop_path, "crops_meta.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the metadata table if it does not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crop_metadata (
                event_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                created_at REAL NOT NULL,
                file_size INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_created "
            "ON crop_metadata(created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_tenant "
            "ON crop_metadata(tenant_id)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_crop(
        self,
        event_id: str,
        tenant_id: str,
        camera_id: str,
        crop_bytes: bytes,
    ) -> PreSignedURL:
        """Encrypt and store a JPEG crop, returning a pre-signed URL.

        Parameters
        ----------
        event_id:
            Unique event identifier (used as file name stem).
        tenant_id:
            Tenant identifier for metadata.
        camera_id:
            Camera identifier for metadata.
        crop_bytes:
            Raw JPEG bytes of the frame crop.

        Returns
        -------
        PreSignedURL
            A time-limited URL for accessing the crop.

        Raises
        ------
        OSError
            If the disk is full or the file cannot be written.
        """
        encrypted = encrypt_crop(crop_bytes, self._key)
        file_name = f"{event_id}.enc"
        file_path = os.path.join(self._crop_path, file_name)

        with self._lock:
            try:
                with open(file_path, "wb") as f:
                    f.write(encrypted)
            except OSError as exc:
                logger.critical(
                    "Frame_Crop_Store disk full or write error: %s. "
                    "Stopping new crop storage but alert pipeline continues.",
                    exc,
                )
                raise

            self._conn.execute(
                """
                INSERT OR REPLACE INTO crop_metadata
                    (event_id, tenant_id, camera_id, file_name, created_at, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    tenant_id,
                    camera_id,
                    file_name,
                    time.time(),
                    len(encrypted),
                ),
            )
            self._conn.commit()

        logger.debug(
            "Stored encrypted crop for event %s (%d bytes encrypted).",
            event_id,
            len(encrypted),
        )

        return generate_presigned_url(
            file_path, self._key, self._url_expiry_seconds
        )

    def get_crop(self, url: str) -> Optional[bytes]:
        """Retrieve and decrypt a crop using a pre-signed URL.

        Parameters
        ----------
        url:
            A pre-signed URL previously returned by :meth:`store_crop`.

        Returns
        -------
        Optional[bytes]
            The decrypted JPEG bytes, or ``None`` if the URL is invalid,
            expired, or the file does not exist.
        """
        file_path = validate_presigned_url(url, self._key)
        if file_path is None:
            logger.warning("Invalid or expired pre-signed URL.")
            return None

        try:
            with open(file_path, "rb") as f:
                blob = f.read()
            return decrypt_crop(blob, self._key)
        except FileNotFoundError:
            logger.warning("Crop file not found: %s", file_path)
            return None
        except Exception as exc:
            logger.error("Failed to decrypt crop: %s", exc)
            return None

    def get_crop_by_event_id(
        self, event_id: str, tenant_id: Optional[str] = None
    ) -> Optional[bytes]:
        """Retrieve and decrypt a crop by event ID (no URL validation).

        Parameters
        ----------
        event_id:
            The event identifier.
        tenant_id:
            If provided, validates that the crop belongs to this tenant
            before returning data.  Returns ``None`` on tenant mismatch.

        Returns
        -------
        Optional[bytes]
            The decrypted JPEG bytes, or ``None`` if not found or tenant
            mismatch.
        """
        if tenant_id is not None:
            row = self._conn.execute(
                "SELECT tenant_id FROM crop_metadata WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            if row[0] != tenant_id:
                logger.warning(
                    "Tenant mismatch for crop %s: expected %s, got %s",
                    event_id,
                    tenant_id,
                    row[0],
                )
                return None

        file_name = f"{event_id}.enc"
        file_path = os.path.join(self._crop_path, file_name)
        try:
            with open(file_path, "rb") as f:
                blob = f.read()
            return decrypt_crop(blob, self._key)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.error("Failed to decrypt crop for event %s: %s", event_id, exc)
            return None

    def generate_url(
        self, event_id: str, tenant_id: Optional[str] = None
    ) -> Optional[PreSignedURL]:
        """Generate a new pre-signed URL for an existing crop.

        Parameters
        ----------
        event_id:
            The event identifier.
        tenant_id:
            If provided, validates that the crop belongs to this tenant
            before generating a URL.  Returns ``None`` on tenant mismatch.

        Returns
        -------
        Optional[PreSignedURL]
            A new pre-signed URL, or ``None`` if the crop does not exist
            or tenant mismatch.
        """
        if tenant_id is not None:
            row = self._conn.execute(
                "SELECT tenant_id FROM crop_metadata WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            if row[0] != tenant_id:
                logger.warning(
                    "Tenant mismatch for URL generation on crop %s: "
                    "expected %s, got %s",
                    event_id,
                    tenant_id,
                    row[0],
                )
                return None

        file_name = f"{event_id}.enc"
        file_path = os.path.join(self._crop_path, file_name)
        if not os.path.exists(file_path):
            return None
        return generate_presigned_url(
            file_path, self._key, self._url_expiry_seconds
        )

    def purge_expired(self, tenant_id: Optional[str] = None) -> int:
        """Delete crops older than the retention period.

        Parameters
        ----------
        tenant_id:
            If provided, only purge expired crops belonging to this tenant.

        Returns
        -------
        int
            Number of crops purged.
        """
        cutoff = time.time() - (self._retention_hours * 3600)
        purged = 0

        with self._lock:
            if tenant_id is not None:
                cursor = self._conn.execute(
                    "SELECT event_id, file_name FROM crop_metadata "
                    "WHERE created_at < ? AND tenant_id = ?",
                    (cutoff, tenant_id),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT event_id, file_name FROM crop_metadata WHERE created_at < ?",
                    (cutoff,),
                )
            rows = cursor.fetchall()

            for event_id, file_name in rows:
                file_path = os.path.join(self._crop_path, file_name)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    purged += 1
                except OSError as exc:
                    logger.error(
                        "Failed to delete expired crop %s: %s", file_name, exc
                    )

            if rows:
                if tenant_id is not None:
                    self._conn.execute(
                        "DELETE FROM crop_metadata "
                        "WHERE created_at < ? AND tenant_id = ?",
                        (cutoff, tenant_id),
                    )
                else:
                    self._conn.execute(
                        "DELETE FROM crop_metadata WHERE created_at < ?",
                        (cutoff,),
                    )
                self._conn.commit()

        if purged > 0:
            logger.info("Purged %d expired frame crops.", purged)

        return purged

    def count(self) -> int:
        """Return the number of stored crops."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM crop_metadata")
        return cursor.fetchone()[0]

    def get_crops_by_tenant(self, tenant_id: str) -> list[dict]:
        """Return crop metadata for all crops belonging to a tenant.

        Parameters
        ----------
        tenant_id:
            The tenant identifier to scope the query.

        Returns
        -------
        list[dict]
            List of crop metadata dicts with keys ``event_id``,
            ``tenant_id``, ``camera_id``, ``file_name``, ``created_at``,
            and ``file_size``.
        """
        cursor = self._conn.execute(
            "SELECT event_id, tenant_id, camera_id, file_name, created_at, file_size "
            "FROM crop_metadata WHERE tenant_id = ? ORDER BY created_at DESC",
            (tenant_id,),
        )
        return [
            {
                "event_id": row[0],
                "tenant_id": row[1],
                "camera_id": row[2],
                "file_name": row[3],
                "created_at": row[4],
                "file_size": row[5],
            }
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        """Close the metadata database connection."""
        self._conn.close()

    @property
    def crop_path(self) -> str:
        """Return the absolute path to the crop storage directory."""
        return self._crop_path

    @property
    def retention_hours(self) -> int:
        """Return the retention period in hours."""
        return self._retention_hours


# ---------------------------------------------------------------------------
# FrameCropStoreTool — for OrchestrationAgent integration
# ---------------------------------------------------------------------------


class FrameCropStoreTool:
    """Orchestration tool that stores frame crops and returns pre-signed URLs.

    Integrates with the :class:`OrchestrationAgent` tool chain so that
    when an alert action is taken, the frame crop from the event is stored
    encrypted and a pre-signed URL is generated for the alert payload.

    Parameters
    ----------
    store:
        The :class:`FrameCropStore` instance.
    """

    def __init__(self, store: FrameCropStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "FrameCropStoreTool"

    def invoke(
        self,
        scene: object,
        event: object,
    ) -> object:
        """Store the frame crop from the event and return a pre-signed URL.

        Parameters
        ----------
        scene:
            The :class:`SceneUnderstanding` (used for metadata only).
        event:
            The :class:`StructuredEvent` containing the frame_crop.

        Returns
        -------
        ToolResult
            Contains ``frame_crop_url`` on success.
        """
        # Import here to avoid circular imports
        from agentic_cctv.orchestration_agent import ToolResult

        try:
            import base64

            # Decode the base64 frame_crop from the event
            crop_bytes = base64.b64decode(event.frame_crop)  # type: ignore[attr-defined]

            presigned = self._store.store_crop(
                event_id=event.event_id,  # type: ignore[attr-defined]
                tenant_id=event.tenant_id,  # type: ignore[attr-defined]
                camera_id=event.camera_id,  # type: ignore[attr-defined]
                crop_bytes=crop_bytes,
            )

            logger.info(
                "FrameCropStoreTool: stored crop for event %s, URL expires %s",
                event.event_id,  # type: ignore[attr-defined]
                presigned.expires_at.isoformat(),
            )

            return ToolResult(
                success=True,
                data={
                    "frame_crop_url": presigned.url,
                    "expires_at": presigned.expires_at.isoformat(),
                },
            )
        except Exception as exc:
            logger.error("FrameCropStoreTool failed: %s", exc, exc_info=True)
            return ToolResult(success=False, error=str(exc))
