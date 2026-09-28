from datetime import datetime


class SecurityAlertEngine:
    """
    Generates explainable security alerts from
    hybrid detection results and risk classification.
    """

    def generate_alert(
        self,
        event,
        detection_result,
        risk,
        threshold
    ):
        hybrid_score = float(
            detection_result.get("hybrid_score", 0.0)
        )

        statistical_score = float(
            detection_result.get("statistical_score", 0.0)
        )

        ml_score = float(
            detection_result.get("ml_score", 0.0)
        )

        quantum_score = float(
            detection_result.get("quantum_score", 0.0)
        )

        alert_required = risk in {
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        }

        timestamp = datetime.now().isoformat()

        # --------------------------------------------------
        # NO ALERT
        # --------------------------------------------------

        if not alert_required:
            return {
                "alert_required": False,
                "severity": "LOW",
                "title": "No Security Alert",
                "message": (
                    "The event remains within "
                    "the adaptive security baseline."
                ),
                "event_id": event.get("event_id"),
                "attack_type": event.get(
                    "attack_type",
                    "UNKNOWN"
                ),
                "hybrid_score": hybrid_score,
                "threshold": float(threshold),
                "statistical_score": statistical_score,
                "ml_score": ml_score,
                "quantum_score": quantum_score,
                "reasons": [],
                "timestamp": timestamp
            }

        attack_type = event.get(
            "attack_type",
            "UNKNOWN"
        )

        reasons = []

        # --------------------------------------------------
        # TAMPERING
        # --------------------------------------------------

        if attack_type == "TAMPERING":

            reasons.append(
                "Digital signature verification failed."
            )

            reasons.append(
                "The signed message differs from "
                "the observed message."
            )

        # --------------------------------------------------
        # REPLAY
        # --------------------------------------------------

        elif attack_type == "REPLAY":

            replay_count = int(
                event.get("replay_count", 0)
            )

            reasons.append(
                f"Signature replay detected "
                f"with replay count {replay_count}."
            )

            reasons.append(
                "A previously valid signature "
                "is being reused."
            )

        # --------------------------------------------------
        # VERIFICATION BURST
        # --------------------------------------------------

        elif attack_type == "VERIFICATION_BURST":

            attempts = int(
                event.get(
                    "verification_attempts",
                    0
                )
            )

            reasons.append(
                "High-frequency signature "
                "verification failures detected."
            )

            reasons.append(
                f"Verification attempt count "
                f"reached {attempts}."
            )

        # --------------------------------------------------
        # NORMAL / UNKNOWN ANOMALY
        # --------------------------------------------------

        else:

            reasons.append(
                "Behavior deviates from the "
                "learned security baseline."
            )

        # --------------------------------------------------
        # DETECTOR EVIDENCE
        # --------------------------------------------------

        detector_evidence = {
            "statistical": statistical_score,
            "machine_learning": ml_score,
            "quantum_inspired": quantum_score,
            "hybrid": hybrid_score
        }

        return {
            "alert_required": True,
            "severity": risk,
            "title": f"{risk} SECURITY ALERT",
            "message": (
                "The event exceeded the adaptive "
                "security risk criteria."
            ),
            "event_id": event.get("event_id"),
            "attack_type": attack_type,
            "hybrid_score": hybrid_score,
            "threshold": float(threshold),
            "statistical_score": statistical_score,
            "ml_score": ml_score,
            "quantum_score": quantum_score,
            "detector_evidence": detector_evidence,
            "reasons": reasons,
            "timestamp": timestamp
        }