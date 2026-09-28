from __future__ import annotations

import random
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from .signature_security import SignatureSecurity


class SecurityEventGenerator:
    """
    Generates synthetic digital-signature security events.

    Supported event types:

        NORMAL
        TAMPERING
        REPLAY
        VERIFICATION_BURST

    The generated events are designed for evaluating
    statistical, machine-learning, and quantum-inspired
    threat detection techniques.
    """

    def __init__(self, seed: Optional[int] = None) -> None:

        if seed is not None:
            random.seed(seed)

        self.security = SignatureSecurity()
        self.events = []

    # ================================================================
    # COMMON EVENT CREATION
    # ================================================================

    def _base_event(
        self,
        message: bytes,
        signature: bytes,
        verification_success: bool,
        attack_type: str,
        timestamp: float,
        replay_count: int = 0,
        verification_attempts: int = 1,
    ) -> dict:
        """
        Create the common representation of a security event.
        """

        digest = self.security.calculate_digest(message)

        return {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": timestamp,

            # Message information
            "message": message.decode(
                errors="replace"
            ),
            "message_size": len(message),

            # Signature information
            "signature_size": len(signature),
            "signature_valid": int(
                verification_success
            ),

            # Verification information
            "verification_success": int(
                verification_success
            ),
            "verification_failure": int(
                not verification_success
            ),
            "verification_attempts": verification_attempts,

            # Replay information
            "replay_count": replay_count,

            # Cryptographic metadata
            "message_digest": digest,

            # Ground-truth event class
            "attack_type": attack_type,
        }

    # ================================================================
    # NORMAL EVENT
    # ================================================================

    def generate_normal_event(
        self,
        timestamp: float
    ) -> dict:
        """
        Generate a legitimate signed transaction.
        """

        amount = random.randint(
            100,
            5000
        )

        message = (
            f"Transfer request amount={amount}"
        ).encode()

        signature = self.security.sign(
            message
        )

        valid = self.security.verify(
            message,
            signature
        )

        return self._base_event(
            message=message,
            signature=signature,
            verification_success=valid,
            attack_type="NORMAL",
            timestamp=timestamp,
            replay_count=0,
            verification_attempts=1,
        )

    # ================================================================
    # TAMPERING EVENT
    # ================================================================

    def generate_tampered_event(
        self,
        timestamp: float
    ) -> dict:
        """
        Generate an event where the message is modified
        after the original message was signed.

        Expected result:
            Signature verification fails.
        """

        original_amount = random.randint(
            100,
            5000
        )

        modified_amount = (
            original_amount
            * random.randint(5, 20)
        )

        original = (
            f"Transfer request amount={original_amount}"
        ).encode()

        modified = (
            f"Transfer request amount={modified_amount}"
        ).encode()

        # Sign the original message.
        signature = self.security.sign(
            original
        )

        # Verify the modified message using
        # the original signature.
        valid = self.security.verify(
            modified,
            signature
        )

        return self._base_event(
            message=modified,
            signature=signature,
            verification_success=valid,
            attack_type="TAMPERING",
            timestamp=timestamp,
            replay_count=0,
            verification_attempts=1,
        )

    # ================================================================
    # REPLAY EVENT
    # ================================================================

    def generate_replay_event(
        self,
        message: bytes,
        signature: bytes,
        timestamp: float,
        replay_count: int,
    ) -> dict:
        """
        Generate a replay event.

        The same valid message/signature pair is
        submitted repeatedly.
        """

        valid = self.security.verify(
            message,
            signature
        )

        return self._base_event(
            message=message,
            signature=signature,
            verification_success=valid,
            attack_type="REPLAY",
            timestamp=timestamp,
            replay_count=replay_count,
            verification_attempts=1,
        )

    # ================================================================
    # VERIFICATION BURST EVENT
    # ================================================================

    def generate_verification_burst_event(
        self,
        timestamp: float,
        attempt_number: int,
    ) -> dict:
        """
        Generate a rapid sequence of invalid
        signature verification attempts.

        This models suspicious repeated verification
        activity over a short time interval.
        """

        original = (
            b"Transfer request amount=1000"
        )

        modified = (
            b"Transfer request amount=999999"
        )

        signature = self.security.sign(
            original
        )

        valid = self.security.verify(
            modified,
            signature
        )

        return self._base_event(
            message=modified,
            signature=signature,
            verification_success=valid,
            attack_type="VERIFICATION_BURST",
            timestamp=timestamp,
            replay_count=0,
            verification_attempts=attempt_number,
        )

    # ================================================================
    # DATASET GENERATION
    # ================================================================

    def generate_dataset(
        self,
        normal_count: int = 100,
        tampering_count: int = 30,
        replay_count: int = 30,
        burst_count: int = 20,
    ) -> pd.DataFrame:
        """
        Generate a complete security-event dataset.

        Default distribution:

            NORMAL              = 100
            TAMPERING           = 30
            REPLAY              = 30
            VERIFICATION_BURST  = 20

            TOTAL               = 180
        """

        events = []

        current_time = time.time()

        # ============================================================
        # NORMAL EVENTS
        # ============================================================

        for _ in range(normal_count):

            current_time += random.uniform(
                2.0,
                15.0
            )

            events.append(
                self.generate_normal_event(
                    current_time
                )
            )

        # ============================================================
        # TAMPERING EVENTS
        # ============================================================

        for _ in range(tampering_count):

            current_time += random.uniform(
                1.0,
                5.0
            )

            events.append(
                self.generate_tampered_event(
                    current_time
                )
            )

        # ============================================================
        # REPLAY EVENTS
        # ============================================================

        replay_message = (
            b"Transfer request amount=1500"
        )

        replay_signature = (
            self.security.sign(
                replay_message
            )
        )

        for replay_number in range(
            1,
            replay_count + 1
        ):

            # Very short interval between replay attempts.
            current_time += random.uniform(
                0.05,
                0.5
            )

            events.append(
                self.generate_replay_event(
                    message=replay_message,
                    signature=replay_signature,
                    timestamp=current_time,
                    replay_count=replay_number,
                )
            )

        # ============================================================
        # VERIFICATION BURST
        # ============================================================

        for attempt_number in range(
            1,
            burst_count + 1
        ):

            # Extremely short intervals represent
            # high-frequency verification attempts.
            current_time += random.uniform(
                0.01,
                0.2
            )

            events.append(
                self.generate_verification_burst_event(
                    timestamp=current_time,
                    attempt_number=attempt_number,
                )
            )

        # ============================================================
        # MAINTAIN CHRONOLOGICAL ORDER
        # ============================================================

        events.sort(
            key=lambda event: event["timestamp"]
        )

        self.events = events

        return pd.DataFrame(events)

    # ================================================================
    # DATASET SAVING
    # ================================================================

    def save_dataset(
        self,
        path: str = "data/raw/security_events.csv"
    ) -> pd.DataFrame:
        """
        Generate and save the dataset as CSV.
        """

        df = self.generate_dataset()

        output_path = Path(path)

        # Create the parent directory if necessary.
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        return df