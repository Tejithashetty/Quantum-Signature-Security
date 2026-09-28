from __future__ import annotations

import numpy as np


class QuantumInspiredEncoder:
    """
    Quantum-inspired feature encoder.

    This module does NOT require a physical quantum computer.
    It represents a classical feature vector as a normalized
    amplitude-like state and derives statistical quantities
    inspired by quantum-state analysis.
    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon

    def encode(self, features) -> dict:
        """
        Convert a numerical security feature vector into
        quantum-inspired statistical features.
        """

        x = np.asarray(
            features,
            dtype=float
        ).flatten()

        if x.size == 0:
            raise ValueError(
                "Feature vector cannot be empty."
            )

        # --------------------------------------------------------
        # Clean invalid values
        # --------------------------------------------------------

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # --------------------------------------------------------
        # Use magnitude for probability construction.
        # This prevents negative security features from
        # producing invalid probability values.
        # --------------------------------------------------------

        x = np.abs(x)

        norm = np.linalg.norm(x)

        # --------------------------------------------------------
        # Zero-vector protection
        # --------------------------------------------------------

        if norm <= self.epsilon:

            amplitudes = (
                np.ones(x.size) /
                np.sqrt(x.size)
            )

        else:

            amplitudes = x / norm

        # --------------------------------------------------------
        # Probability distribution
        # --------------------------------------------------------

        probabilities = amplitudes ** 2

        probabilities = np.clip(
            probabilities,
            self.epsilon,
            1.0
        )

        # Re-normalize after clipping.
        probabilities = (
            probabilities /
            probabilities.sum()
        )

        # ========================================================
        # QUANTUM-INSPIRED MEASURES
        # ========================================================

        # --------------------------------------------------------
        # 1. Shannon entropy
        # --------------------------------------------------------

        entropy = -np.sum(
            probabilities *
            np.log2(probabilities)
        )

        # --------------------------------------------------------
        # 2. State purity
        # --------------------------------------------------------

        purity = np.sum(
            probabilities ** 2
        )

        # --------------------------------------------------------
        # 3. Probability concentration
        # --------------------------------------------------------

        concentration = np.max(
            probabilities
        )

        # --------------------------------------------------------
        # 4. Effective dimension
        # --------------------------------------------------------

        effective_dimension = (
            1.0 /
            max(purity, self.epsilon)
        )

        # --------------------------------------------------------
        # 5. Probability variance
        # --------------------------------------------------------

        probability_variance = np.var(
            probabilities
        )

        # --------------------------------------------------------
        # 6. Dominant-state ratio
        # --------------------------------------------------------

        sorted_probabilities = np.sort(
            probabilities
        )[::-1]

        if len(sorted_probabilities) >= 2:

            dominant_ratio = (
                sorted_probabilities[0] /
                max(
                    sorted_probabilities[1],
                    self.epsilon
                )
            )

        else:

            dominant_ratio = 1.0

        # --------------------------------------------------------
        # 7. Tail concentration
        # --------------------------------------------------------

        top_k = max(
            1,
            int(np.ceil(
                len(probabilities) * 0.25
            ))
        )

        tail_concentration = np.sum(
            sorted_probabilities[:top_k]
        )

        # --------------------------------------------------------
        # Normalize entropy to approximately 0-1.
        # --------------------------------------------------------

        if len(probabilities) > 1:

            normalized_entropy = (
                entropy /
                np.log2(len(probabilities))
            )

        else:

            normalized_entropy = 0.0

        # --------------------------------------------------------
        # Quantum-inspired anomaly score
        #
        # Higher concentration + higher purity +
        # lower normalized entropy indicate a more
        # concentrated feature state.
        # --------------------------------------------------------

        quantum_anomaly_score = (
            0.35 * concentration
            + 0.30 * purity
            + 0.20 * (1.0 - normalized_entropy)
            + 0.15 * tail_concentration
        )

        quantum_anomaly_score = float(
            np.clip(
                quantum_anomaly_score,
                0.0,
                1.0
            )
        )

        return {
            # Existing project outputs
            "quantum_entropy": float(
                entropy
            ),

            "state_purity": float(
                purity
            ),

            "amplitude_concentration": float(
                concentration
            ),

            "effective_dimension": float(
                effective_dimension
            ),

            # Additional quantum-inspired measurements
            "normalized_quantum_entropy": float(
                normalized_entropy
            ),

            "probability_variance": float(
                probability_variance
            ),

            "dominant_state_ratio": float(
                dominant_ratio
            ),

            "tail_concentration": float(
                tail_concentration
            ),

            "quantum_anomaly_score": (
                quantum_anomaly_score
            ),
        }