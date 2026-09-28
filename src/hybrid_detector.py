from __future__ import annotations

import numpy as np

from .ml_detector import MLDetector
from .quantum_features import QuantumInspiredEncoder
from .statistical_detector import StatisticalDetector


class HybridThreatDetector:
    """
    Hybrid cybersecurity threat detector.

    Combines three independent signals:

        1. Statistical anomaly detection
        2. Machine-learning anomaly detection
        3. Quantum-inspired feature analysis

    The final score is a transparent weighted fusion
    of the three normalized signals.
    """

    def __init__(
        self,
        statistical_weight: float = 0.35,
        ml_weight: float = 0.40,
        quantum_weight: float = 0.25,
    ) -> None:

        weights = np.array(
            [
                statistical_weight,
                ml_weight,
                quantum_weight,
            ],
            dtype=float,
        )

        if np.any(weights < 0):
            raise ValueError(
                "Detector weights cannot be negative."
            )

        if np.sum(weights) <= 0:
            raise ValueError(
                "At least one detector weight must be positive."
            )

        # Normalize weights so that they sum to exactly 1.
        weights = weights / np.sum(weights)

        self.statistical_weight = float(
            weights[0]
        )

        self.ml_weight = float(
            weights[1]
        )

        self.quantum_weight = float(
            weights[2]
        )

        self.statistical = StatisticalDetector()

        self.ml = MLDetector()

        self.quantum = QuantumInspiredEncoder()

        self.feature_count = 0

        self.is_fitted = False

    # ============================================================
    # FIT
    # ============================================================

    def fit(self, X):
        """
        Train the statistical and ML components on NORMAL data.

        The quantum-inspired encoder does not require training.
        """

        X = np.asarray(
            X,
            dtype=float
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "Cannot fit hybrid detector on empty data."
            )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.feature_count = X.shape[1]

        self.statistical.fit(X)

        self.ml.fit(X)

        self.is_fitted = True

        return self

    # ============================================================
    # VALIDATION
    # ============================================================

    def _check_fitted(self):

        if not self.is_fitted:
            raise RuntimeError(
                "HybridThreatDetector must be fitted before prediction."
            )

    def _prepare_input(self, x):

        self._check_fitted()

        x = np.asarray(
            x,
            dtype=float
        ).flatten()

        if x.size != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, "
                f"received {x.size}."
            )

        return np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict(self, x) -> dict:
        """
        Calculate all detector scores and the final
        hybrid threat score.
        """

        x = self._prepare_input(x)

        # --------------------------------------------------------
        # Statistical detector
        # --------------------------------------------------------

        statistical_score = (
            self.statistical.score(x)
        )

        # --------------------------------------------------------
        # ML detector
        # --------------------------------------------------------

        ml_score = (
            self.ml.score(x)
        )

        # --------------------------------------------------------
        # Quantum-inspired analysis
        # --------------------------------------------------------

        quantum_result = (
            self.quantum.encode(x)
        )

        quantum_score = (
            quantum_result[
                "quantum_anomaly_score"
            ]
        )

        # --------------------------------------------------------
        # Weighted fusion
        # --------------------------------------------------------

        statistical_contribution = (
            self.statistical_weight *
            statistical_score
        )

        ml_contribution = (
            self.ml_weight *
            ml_score
        )

        quantum_contribution = (
            self.quantum_weight *
            quantum_score
        )

        hybrid_score = (
            statistical_contribution
            + ml_contribution
            + quantum_contribution
        )

        hybrid_score = float(
            np.clip(
                hybrid_score,
                0.0,
                1.0
            )
        )

        return {
            # Individual detector outputs
            "statistical_score": float(
                statistical_score
            ),

            "ml_score": float(
                ml_score
            ),

            "quantum_score": float(
                quantum_score
            ),

            # Final hybrid score
            "hybrid_score": hybrid_score,

            # Detector weights
            "statistical_weight": (
                self.statistical_weight
            ),

            "ml_weight": (
                self.ml_weight
            ),

            "quantum_weight": (
                self.quantum_weight
            ),

            # Contribution of each detector
            "statistical_contribution": (
                float(statistical_contribution)
            ),

            "ml_contribution": (
                float(ml_contribution)
            ),

            "quantum_contribution": (
                float(quantum_contribution)
            ),

            # Quantum-inspired details
            "quantum_features": quantum_result,
        }

    # ============================================================
    # BATCH PREDICTION
    # ============================================================

    def predict_batch(self, X):
        """
        Score multiple events.
        """

        X = np.asarray(
            X,
            dtype=float
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        return [
            self.predict(row)
            for row in X
        ]

    # ============================================================
    # DETECTOR SUMMARY
    # ============================================================

    def detector_summary(self) -> dict:
        """
        Return the configuration of the hybrid detector.
        """

        return {
            "statistical_weight": (
                self.statistical_weight
            ),

            "ml_weight": (
                self.ml_weight
            ),

            "quantum_weight": (
                self.quantum_weight
            ),

            "total_weight": (
                self.statistical_weight
                + self.ml_weight
                + self.quantum_weight
            ),
        }