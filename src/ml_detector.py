from __future__ import annotations

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class MLDetector:
    """
    Machine-learning anomaly detector based on Isolation Forest.

    The model is trained using NORMAL security events only.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:

        self.scaler = StandardScaler()

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_count = 0
        self.is_fitted = False

    # ============================================================
    # FIT
    # ============================================================

    def fit(self, X):
        """
        Train the Isolation Forest on NORMAL-event features.
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
                "Cannot train ML detector on empty data."
            )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.feature_count = X.shape[1]

        X_scaled = (
            self.scaler.fit_transform(X)
        )

        self.model.fit(
            X_scaled
        )

        self.is_fitted = True

        return self

    # ============================================================
    # VALIDATION
    # ============================================================

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "MLDetector must be fitted before scoring."
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
    # RAW ISOLATION FOREST SCORE
    # ============================================================

    def raw_score(self, x) -> float:
        """
        Return the raw Isolation Forest score.

        More negative values generally indicate stronger
        anomalous behavior.
        """

        x = self._prepare_input(x)

        x_scaled = (
            self.scaler.transform(
                x.reshape(1, -1)
            )
        )

        return float(
            self.model.score_samples(
                x_scaled
            )[0]
        )

    # ============================================================
    # DECISION FUNCTION
    # ============================================================

    def decision_score(self, x) -> float:
        """
        Return the Isolation Forest decision function.

        Positive values generally correspond to inlier
        behavior and negative values to outlier behavior.
        """

        x = self._prepare_input(x)

        x_scaled = (
            self.scaler.transform(
                x.reshape(1, -1)
            )
        )

        return float(
            self.model.decision_function(
                x_scaled
            )[0]
        )

    # ============================================================
    # NORMALIZED ANOMALY SCORE
    # ============================================================

    def score(self, x) -> float:
        """
        Convert the Isolation Forest output into a
        0–1 anomaly score.

        Higher score = more anomalous.
        """

        raw = self.raw_score(x)

        # Isolation Forest normally produces values
        # around a relatively small negative/positive range.
        #
        # A sigmoid provides a smooth 0–1 representation.
        normalized = 1.0 / (
            1.0 +
            np.exp(
                np.clip(
                    5.0 * raw,
                    -50,
                    50
                )
            )
        )

        return float(
            np.clip(
                normalized,
                0.0,
                1.0
            )
        )

    # ============================================================
    # BATCH SCORING
    # ============================================================

    def score_samples(self, X) -> np.ndarray:
        """
        Calculate normalized anomaly scores for multiple events.
        """

        self._check_fitted()

        X = np.asarray(
            X,
            dtype=float
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if X.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, "
                f"received {X.shape[1]}."
            )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        X_scaled = (
            self.scaler.transform(X)
        )

        raw_scores = (
            self.model.score_samples(
                X_scaled
            )
        )

        normalized = 1.0 / (
            1.0 +
            np.exp(
                np.clip(
                    5.0 * raw_scores,
                    -50,
                    50
                )
            )
        )

        return np.clip(
            normalized,
            0.0,
            1.0
        )

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(self, x) -> dict:
        """
        Return detailed ML anomaly information.
        """

        return {
            "ml_score": self.score(x),
            "raw_isolation_score": self.raw_score(x),
            "decision_score": self.decision_score(x),
        }