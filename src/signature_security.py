from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


@dataclass
class SignatureResult:
    """Stores the result of a signature verification operation."""

    valid: bool
    message_digest: str
    message_size: int
    signature_size: int


class SignatureSecurity:
    """
    Digital signature security component.

    Uses:
        Elliptic Curve Digital Signature Algorithm (ECDSA)
        Curve: SECP256R1
        Hash: SHA-256
    """

    def __init__(self) -> None:
        # Generate a fresh private key.
        self.private_key = ec.generate_private_key(
            ec.SECP256R1()
        )

        # Derive the corresponding public key.
        self.public_key = self.private_key.public_key()

    # ------------------------------------------------------------------
    # SIGNATURE OPERATIONS
    # ------------------------------------------------------------------

    def sign(self, message: bytes) -> bytes:
        """
        Create an ECDSA digital signature for a message.
        """

        if not isinstance(message, bytes):
            raise TypeError("message must be bytes")

        return self.private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )

    def verify(
        self,
        message: bytes,
        signature: bytes
    ) -> bool:
        """
        Verify an ECDSA signature.

        Returns:
            True  -> signature is valid
            False -> signature is invalid
        """

        if not isinstance(message, bytes):
            raise TypeError("message must be bytes")

        if not isinstance(signature, bytes):
            raise TypeError("signature must be bytes")

        try:
            self.public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )

            return True

        except InvalidSignature:
            return False

    # ------------------------------------------------------------------
    # SECURITY METADATA
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_digest(message: bytes) -> str:
        """
        Calculate the SHA-256 digest of a message.

        Returns:
            Hexadecimal SHA-256 digest.
        """

        if not isinstance(message, bytes):
            raise TypeError("message must be bytes")

        return hashlib.sha256(message).hexdigest()

    @staticmethod
    def get_message_size(message: bytes) -> int:
        """Return message size in bytes."""

        if not isinstance(message, bytes):
            raise TypeError("message must be bytes")

        return len(message)

    @staticmethod
    def get_signature_size(signature: bytes) -> int:
        """Return signature size in bytes."""

        if not isinstance(signature, bytes):
            raise TypeError("signature must be bytes")

        return len(signature)

    # ------------------------------------------------------------------
    # KEY INFORMATION
    # ------------------------------------------------------------------

    def get_public_key_bytes(self) -> bytes:
        """
        Export the public key in DER format.
        """

        return self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def get_public_key_fingerprint(self) -> str:
        """
        Generate a SHA-256 fingerprint of the public key.

        This is useful for identifying the signing key
        without exposing the private key.
        """

        public_key_bytes = self.get_public_key_bytes()

        return hashlib.sha256(
            public_key_bytes
        ).hexdigest()

    # ------------------------------------------------------------------
    # ANALYSIS HELPER
    # ------------------------------------------------------------------

    def analyze_signature(
        self,
        message: bytes,
        signature: bytes
    ) -> SignatureResult:
        """
        Perform signature verification and return security metadata.
        """

        valid = self.verify(
            message,
            signature
        )

        return SignatureResult(
            valid=valid,
            message_digest=self.calculate_digest(message),
            message_size=self.get_message_size(message),
            signature_size=self.get_signature_size(signature)
        )

    # ------------------------------------------------------------------
    # SERIALIZATION HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def encode_signature(signature: bytes) -> str:
        """
        Encode a binary signature as Base64.

        Useful when storing signatures in CSV/JSON datasets.
        """

        if not isinstance(signature, bytes):
            raise TypeError("signature must be bytes")

        return base64.b64encode(signature).decode("utf-8")

    @staticmethod
    def decode_signature(encoded_signature: str) -> bytes:
        """
        Decode a Base64 signature back into bytes.
        """

        if not isinstance(encoded_signature, str):
            raise TypeError(
                "encoded_signature must be a string"
            )

        return base64.b64decode(
            encoded_signature.encode("utf-8")
        )