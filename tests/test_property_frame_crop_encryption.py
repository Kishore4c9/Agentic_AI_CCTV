"""Property-based test for Frame Crop Encryption Round-Trip.

# Feature: agentic-ai-cctv-monitoring, Property 11: Frame Crop Encryption Round-Trip

**Validates: Requirements 10.3**

For random byte arrays (1B to 1MB), encrypting with AES-256 and decrypting
with the same key produces the original bytes.
"""

from __future__ import annotations

import secrets

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_cctv.frame_crop_store import encrypt_crop, decrypt_crop

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Random byte arrays from 1 byte to 1 MB
_crop_bytes_strategy = st.binary(min_size=1, max_size=1_000_000)

# Valid AES-256 key (32 bytes)
_key_strategy = st.binary(min_size=32, max_size=32)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestFrameCropEncryptionRoundTrip:
    """Property 11: Frame Crop Encryption Round-Trip.

    **Validates: Requirements 10.3**
    """

    @given(
        plaintext=_crop_bytes_strategy,
        key=_key_strategy,
    )
    @settings(max_examples=20)
    def test_encrypt_decrypt_round_trip(
        self,
        plaintext: bytes,
        key: bytes,
    ) -> None:
        """For any random byte array (1B to 1MB), encrypting with AES-256
        and decrypting with the same key produces the original bytes.
        """
        encrypted = encrypt_crop(plaintext, key)

        # Encrypted output must differ from plaintext (nonce + ciphertext)
        assert encrypted != plaintext, (
            "Encrypted output should differ from plaintext"
        )

        # Encrypted output must be longer than plaintext (nonce + auth tag)
        assert len(encrypted) > len(plaintext), (
            "Encrypted output should be longer than plaintext "
            "(includes nonce and authentication tag)"
        )

        # Decrypting with the same key must produce the original plaintext
        decrypted = decrypt_crop(encrypted, key)
        assert decrypted == plaintext, (
            f"Decrypted bytes should equal original plaintext. "
            f"Original length={len(plaintext)}, decrypted length={len(decrypted)}"
        )
