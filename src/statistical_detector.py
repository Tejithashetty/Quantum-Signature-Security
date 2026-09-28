from __future__ import annotations

import numpy as np


class StatisticalDetector:
    """
    Statistical anomaly detector based on standardized
    deviation from a NORMAL-event baseline.

    The detector is trained using NORMAL events only.
    """

    def __init__(
        self,
        score_scale: float = 3.0
    ) -> None:

        self.mean = None
        self.std = None
        self.score_scale = score_scale
        self.feature_count = 0

    # ============================================================
    # FIT
    # ============================================================

    def fit(self, X):
        """
        Learn the statistical baseline from NORMAL events.

        Parameters
        ----------
        X:
            Numerical matrix containing NORMAL-event features.
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
                "Cannot fit detector on an empty dataset."
            )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.mean = np.mean(
            X,
            axis=0
        )

        self.std = np.std(
            X,
            axis=0
        )

        # Prevent division by zero for
        # constant features.
        self.std = np.where(
            self.std < 1e-8,
            1.0,
            self.std
        )

        self.feature_count = X.shape[1]

        return self

    # ============================================================
    # VALIDATION
    # ============================================================

    def _check_fitted(self):
        """Ensure the detector has been fitted."""

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "StatisticalDetector must be fitted before scoring."
            )

    def _prepare_input(self, x):
        """Convert input to a safe numerical vector."""

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
    # Z-SCORE
    # ============================================================

    def z_scores(self, x):
        """
        Calculate absolute standardized deviations
        from the NORMAL baseline.
        """

        x = self._prepare_input(x)

        z_scores = np.abs(
            (x - self.mean) /
            self.std
        )

        return z_scores

    # ============================================================
    # RAW ANOMALY SCORE
    # ============================================================

    def raw_score(self, x) -> float:
        """
        Return the mean absolute standardized deviation.
        """

        z_scores = self.z_scores(x)

        return float(
            np.mean(z_scores)
        )

    # ============================================================
    # NORMALIZED SCORE
    # ============================================================

    def score(self, x) -> float:
        """
        Convert statistical deviation into a 0–1
        anomaly score.

        Higher score = more anomalous.
        """

        raw = self.raw_score(x)

        normalized = (
            1.0 -
            np.exp(
                -raw /
                self.score_scale
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
    # FEATURE CONTRIBUTIONS
    # ============================================================

    def feature_contributions(
        self,
        x,
        feature_names=None
    ) -> dict:
        """
        Return per-feature standardized anomaly contributions.

        Larger values indicate stronger deviation from
        the NORMAL baseline.
        """

        z_scores = self.z_scores(x)

        if feature_names is None:

            feature_names = [
                f"feature_{i}"
                for i in range(
                    self.feature_count
                )
            ]

        if len(feature_names) != self.feature_count:
            raise ValueError(
                "Number of feature names must match "
                "the number of features."
            )

        return {
            name: float(score)
            for name, score
            in zip(
                feature_names,
                z_scores
            )
        }

    # ============================================================
    # TOP ANOMALOUS FEATURES
    # ============================================================

    def top_anomalies(
        self,
        x,
        feature_names=None,
        top_k: int = 5
    ):
        """
        Return the features contributing most strongly
        to the statistical anomaly score.
        """

        contributions = (
            self.feature_contributions(
                x,
                feature_names
            )
        )

        ranked = sorted(
            contributions.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return ranked[:top_k]

    # ============================================================
    # DETAILED ANALYSIS
    # ============================================================

    def analyze(
        self,
        x,
        feature_names=None
    ) -> dict:
        """
        Return a complete statistical analysis.
        """

        score = self.score(x)

        contributions = (
            self.feature_contributions(
                x,
                feature_names
            )
        )

        top_features = sorted(
            contributions.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return {
            "statistical_score": score,
            "raw_z_score": self.raw_score(x),
            "feature_contributions": contributions,
            "top_anomalous_features": top_features[:5],
        }