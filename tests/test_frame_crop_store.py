"""Unit tests for FrameCropStore.

Tests encryption/decryption, 72-hour retention purge, pre-signed URL
generation and expiry.

Requirements: 10.3, 10.4, 10.5
"""

from __future__ import annotations

import os
import secrets
import time
from unittest.mock import patch

import pytest

from agentic_cctv.frame_crop_store import (
    FrameCropStore,
    FrameCropStoreTool,
    PreSignedURL,
    decrypt_crop,
    encrypt_crop,
    generate_presigned_url,
    validate_presigned_url,
    _validate_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A valid 32-byte hex key for testing
_TEST_KEY_HEX = secrets.token_hex(32)
_TEST_KEY = bytes.fromhex(_TEST_KEY_HEX)


@pytest.fixture
def crop_dir(tmp_path):
    """Return a temporary directory for crop storage."""
    return str(tmp_path / "crops")


@pytest.fixture
def store(crop_dir):
    """Create a FrameCropStore with a temporary directory."""
    s = FrameCropStore(
        crop_path=crop_dir,
        encryption_key_hex=_TEST_KEY_HEX,
        retention_hours=72,
        url_expiry_seconds=3600,
    )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Key validation tests
# ---------------------------------------------------------------------------


class TestKeyValidation:
    """Test encryption key validation."""

    def test_empty_key_raises(self):
        """Encryption key missing → fail startup with descriptive error."""
        with pytest.raises(ValueError, match="missing"):
            _validate_key("")

    def test_invalid_hex_raises(self):
        """Non-hex key → fail with descriptive error."""
        with pytest.raises(ValueError, match="not valid hex"):
            _validate_key("not-a-hex-string")

    def test_wrong_length_raises(self):
        """Key with wrong length → fail with descriptive error."""
        short_key = secrets.token_hex(16)  # 16 bytes, not 32
        with pytest.raises(ValueError, match="32 bytes"):
            _validate_key(short_key)

    def test_valid_key_returns_bytes(self):
        """Valid 32-byte hex key → returns raw bytes."""
        raw = _validate_key(_TEST_KEY_HEX)
        assert len(raw) == 32
        assert isinstance(raw, bytes)


# ---------------------------------------------------------------------------
# Encryption / Decryption tests
# ---------------------------------------------------------------------------


class TestEncryptionDecryption:
    """Test AES-256-GCM encryption and decryption."""

    def test_round_trip_small(self):
        """Encrypt and decrypt a small payload."""
        data = b"hello world"
        encrypted = encrypt_crop(data, _TEST_KEY)
        decrypted = decrypt_crop(encrypted, _TEST_KEY)
        assert decrypted == data

    def test_round_trip_empty_like(self):
        """Encrypt and decrypt a single byte."""
        data = b"\x00"
        encrypted = encrypt_crop(data, _TEST_KEY)
        decrypted = decrypt_crop(encrypted, _TEST_KEY)
        assert decrypted == data

    def test_round_trip_large(self):
        """Encrypt and decrypt a larger payload (64 KB)."""
        data = secrets.token_bytes(65536)
        encrypted = encrypt_crop(data, _TEST_KEY)
        decrypted = decrypt_crop(encrypted, _TEST_KEY)
        assert decrypted == data

    def test_encrypted_differs_from_plaintext(self):
        """Encrypted output must differ from plaintext."""
        data = b"test data for encryption"
        encrypted = encrypt_crop(data, _TEST_KEY)
        assert encrypted != data

    def test_wrong_key_fails(self):
        """Decrypting with a different key must fail."""
        data = b"secret data"
        encrypted = encrypt_crop(data, _TEST_KEY)
        wrong_key = secrets.token_bytes(32)
        with pytest.raises(Exception):
            decrypt_crop(encrypted, wrong_key)

    def test_tampered_ciphertext_fails(self):
        """Tampered ciphertext must fail authentication."""
        data = b"important data"
        encrypted = encrypt_crop(data, _TEST_KEY)
        # Flip a byte in the ciphertext (after the 12-byte nonce)
        tampered = bytearray(encrypted)
        tampered[15] ^= 0xFF
        with pytest.raises(Exception):
            decrypt_crop(bytes(tampered), _TEST_KEY)

    def test_too_short_blob_raises(self):
        """Blob shorter than nonce size must raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            decrypt_crop(b"short", _TEST_KEY)


# ---------------------------------------------------------------------------
# FrameCropStore tests
# ---------------------------------------------------------------------------


class TestFrameCropStore:
    """Test the FrameCropStore class."""

    def test_store_missing_key_raises(self, crop_dir):
        """FrameCropStore with empty key → ValueError at init."""
        with pytest.raises(ValueError, match="missing"):
            FrameCropStore(crop_path=crop_dir, encryption_key_hex="")

    def test_store_and_retrieve(self, store):
        """Store a crop and retrieve it via pre-signed URL."""
        data = b"JPEG crop data here"
        presigned = store.store_crop(
            event_id="evt-001",
            tenant_id="tenant-a",
            camera_id="cam-01",
            crop_bytes=data,
        )

        assert isinstance(presigned, PreSignedURL)
        assert "token=" in presigned.url
        assert "expires=" in presigned.url

        # Retrieve via URL
        retrieved = store.get_crop(presigned.url)
        assert retrieved == data

    def test_store_and_retrieve_by_event_id(self, store):
        """Store a crop and retrieve it by event ID."""
        data = b"another crop"
        store.store_crop(
            event_id="evt-002",
            tenant_id="tenant-a",
            camera_id="cam-01",
            crop_bytes=data,
        )

        retrieved = store.get_crop_by_event_id("evt-002")
        assert retrieved == data

    def test_nonexistent_event_returns_none(self, store):
        """Retrieving a non-existent event returns None."""
        assert store.get_crop_by_event_id("evt-nonexistent") is None

    def test_count(self, store):
        """Count returns the number of stored crops."""
        assert store.count() == 0
        store.store_crop("evt-1", "t", "c", b"data1")
        assert store.count() == 1
        store.store_crop("evt-2", "t", "c", b"data2")
        assert store.count() == 2

    def test_generate_url_for_existing(self, store):
        """Generate a new pre-signed URL for an existing crop."""
        store.store_crop("evt-url", "t", "c", b"data")
        url = store.generate_url("evt-url")
        assert url is not None
        assert "token=" in url.url

    def test_generate_url_for_nonexistent(self, store):
        """Generate URL for non-existent crop returns None."""
        assert store.generate_url("evt-nope") is None

    def test_encrypted_file_on_disk(self, store):
        """The file on disk must be encrypted (not plaintext)."""
        data = b"plaintext crop data"
        store.store_crop("evt-disk", "t", "c", data)
        file_path = os.path.join(store.crop_path, "evt-disk.enc")
        assert os.path.exists(file_path)
        with open(file_path, "rb") as f:
            on_disk = f.read()
        assert on_disk != data  # Must be encrypted


# ---------------------------------------------------------------------------
# Retention / purge tests
# ---------------------------------------------------------------------------


class TestRetentionPurge:
    """Test 72-hour retention and auto-purge."""

    def test_purge_expired_crops(self, store):
        """Crops older than retention period are purged."""
        # Store a crop
        store.store_crop("evt-old", "t", "c", b"old data")
        assert store.count() == 1

        # Simulate time passing beyond retention (72 hours + 1 second)
        past_time = time.time() - (72 * 3600 + 1)
        store._conn.execute(
            "UPDATE crop_metadata SET created_at = ? WHERE event_id = ?",
            (past_time, "evt-old"),
        )
        store._conn.commit()

        purged = store.purge_expired()
        assert purged == 1
        assert store.count() == 0
        assert store.get_crop_by_event_id("evt-old") is None

    def test_purge_keeps_recent_crops(self, store):
        """Crops within retention period are not purged."""
        store.store_crop("evt-recent", "t", "c", b"recent data")
        purged = store.purge_expired()
        assert purged == 0
        assert store.count() == 1

    def test_purge_mixed(self, store):
        """Only expired crops are purged; recent ones remain."""
        store.store_crop("evt-keep", "t", "c", b"keep")
        store.store_crop("evt-delete", "t", "c", b"delete")

        # Make one old
        past_time = time.time() - (72 * 3600 + 1)
        store._conn.execute(
            "UPDATE crop_metadata SET created_at = ? WHERE event_id = ?",
            (past_time, "evt-delete"),
        )
        store._conn.commit()

        purged = store.purge_expired()
        assert purged == 1
        assert store.count() == 1
        assert store.get_crop_by_event_id("evt-keep") == b"keep"
        assert store.get_crop_by_event_id("evt-delete") is None


# ---------------------------------------------------------------------------
# Pre-signed URL tests
# ---------------------------------------------------------------------------


class TestPreSignedURL:
    """Test pre-signed URL generation and validation."""

    def test_valid_url(self):
        """A freshly generated URL is valid."""
        path = "/tmp/test/crop.enc"
        presigned = generate_presigned_url(path, _TEST_KEY, expires_seconds=3600)
        result = validate_presigned_url(presigned.url, _TEST_KEY)
        assert result is not None

    def test_expired_url(self):
        """An expired URL is rejected."""
        path = "/tmp/test/crop.enc"
        # Generate with -1 seconds expiry (already expired)
        presigned = generate_presigned_url(path, _TEST_KEY, expires_seconds=-1)
        result = validate_presigned_url(presigned.url, _TEST_KEY)
        assert result is None

    def test_wrong_key_url(self):
        """A URL validated with a different key is rejected."""
        path = "/tmp/test/crop.enc"
        presigned = generate_presigned_url(path, _TEST_KEY, expires_seconds=3600)
        wrong_key = secrets.token_bytes(32)
        result = validate_presigned_url(presigned.url, wrong_key)
        assert result is None

    def test_tampered_url(self):
        """A URL with a tampered token is rejected."""
        path = "/tmp/test/crop.enc"
        presigned = generate_presigned_url(path, _TEST_KEY, expires_seconds=3600)
        tampered = presigned.url.replace("token=", "token=bad")
        result = validate_presigned_url(tampered, _TEST_KEY)
        assert result is None

    def test_missing_token(self):
        """A URL without a token is rejected."""
        result = validate_presigned_url("file:///tmp/crop.enc?expires=9999999999", _TEST_KEY)
        assert result is None

    def test_missing_expires(self):
        """A URL without an expires parameter is rejected."""
        result = validate_presigned_url("file:///tmp/crop.enc?token=abc", _TEST_KEY)
        assert result is None

    def test_get_crop_with_expired_url(self, store):
        """get_crop with an expired URL returns None."""
        data = b"crop data"
        store.store_crop("evt-exp", "t", "c", data)

        # Generate a URL that is already expired
        file_path = os.path.join(store.crop_path, "evt-exp.enc")
        presigned = generate_presigned_url(file_path, _TEST_KEY, expires_seconds=-1)

        result = store.get_crop(presigned.url)
        assert result is None


# ---------------------------------------------------------------------------
# FrameCropStoreTool tests
# ---------------------------------------------------------------------------


class TestFrameCropStoreTool:
    """Test the FrameCropStoreTool for OrchestrationAgent integration."""

    def test_tool_name(self, store):
        """Tool name is 'FrameCropStoreTool'."""
        tool = FrameCropStoreTool(store)
        assert tool.name == "FrameCropStoreTool"

    def test_tool_stores_crop(self, store):
        """Tool stores a crop and returns a pre-signed URL."""
        import base64
        from unittest.mock import MagicMock

        tool = FrameCropStoreTool(store)

        # Create mock event with base64 frame_crop
        crop_data = b"JPEG image data"
        event = MagicMock()
        event.event_id = "evt-tool-001"
        event.tenant_id = "tenant-test"
        event.camera_id = "cam-test"
        event.frame_crop = base64.b64encode(crop_data).decode()

        scene = MagicMock()

        result = tool.invoke(scene, event)
        assert result.success is True
        assert "frame_crop_url" in result.data
        assert "expires_at" in result.data

        # Verify the crop was stored
        retrieved = store.get_crop_by_event_id("evt-tool-001")
        assert retrieved == crop_data

    def test_tool_handles_error(self, store):
        """Tool returns failure on error."""
        from unittest.mock import MagicMock

        tool = FrameCropStoreTool(store)

        # Create mock event with invalid base64
        event = MagicMock()
        event.event_id = "evt-bad"
        event.tenant_id = "t"
        event.camera_id = "c"
        event.frame_crop = "not-valid-base64!!!"

        scene = MagicMock()

        result = tool.invoke(scene, event)
        assert result.success is False
        assert result.error is not None
