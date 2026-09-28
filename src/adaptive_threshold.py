from __future__ import annotations

import numpy as np


class AdaptiveThreshold:
    """
    Adaptive threshold for hybrid threat scores.

    The threshold is learned from baseline/low-risk observations
    and updated only when the caller explicitly allows adaptation.
    """

    def __init__(
        self,
        k: float = 2.5,
        minimum_threshold: float = 0.50,
        maximum_threshold: float = 0.90,
        history_size: int = 100,
        default_threshold: float = 0.65,
    ) -> None:

        if k < 0:
            raise ValueError(
                "k must be non-negative."
            )

        if minimum_threshold >= maximum_threshold:
            raise ValueError(
                "minimum_threshold must be smaller "
                "than maximum_threshold."
            )

        if history_size < 5:
            raise ValueError(
                "history_size must be at least 5."
            )

        self.k = float(k)

        self.minimum_threshold = float(
            minimum_threshold
        )

        self.maximum_threshold = float(
            maximum_threshold
        )

        self.history_size = int(
            history_size
        )

        self.default_threshold = float(
            np.clip(
                default_threshold,
                self.minimum_threshold,
                self.maximum_threshold,
            )
        )

        self.history: list[float] = []

        self.threshold = (
            self.default_threshold
        )

    # ============================================================
    # INITIAL BASELINE
    # ============================================================

    def fit(self, baseline_scores):
        """
        Initialize the adaptive threshold using known
        NORMAL-event scores.
        """

        scores = np.asarray(
            baseline_scores,
            dtype=float,
        ).flatten()

        scores = scores[
            np.isfinite(scores)
        ]

        if scores.size == 0:
            self.history = []
            self.threshold = (
                self.default_threshold
            )
            return self.threshold

        self.history = list(
            scores[-self.history_size:]
        )

        self._recalculate()

        return self.threshold

    # ============================================================
    # UPDATE
    # ============================================================

    def update(
        self,
        score: float,
        allow_adaptation: bool = True,
    ) -> float:
        """
        Update the baseline with a new observation.

        IMPORTANT:
        Suspicious events should normally be passed with
        allow_adaptation=False so they do not contaminate
        the normal baseline.
        """

        score = float(score)

        if not np.isfinite(score):
            return self.threshold

        if not allow_adaptation:
            return self.threshold

        self.history.append(
            score
        )

        if len(self.history) > self.history_size:
            self.history.pop(0)

        self._recalculate()

        return self.threshold

    # ============================================================
    # CALCULATE THRESHOLD
    # ============================================================

    def _recalculate(self) -> None:
        """
        Recalculate threshold using:

            mean + k * standard deviation

        The result is bounded by the configured limits.
        """

        if len(self.history) < 5:
            self.threshold = (
                self.default_threshold
            )
            return

        scores = np.asarray(
            self.history,
            dtype=float,
        )

        mean = np.mean(
            scores
        )

        std = np.std(
            scores
        )

        adaptive_threshold = (
            mean +
            self.k * std
        )

        self.threshold = float(
            np.clip(
                adaptive_threshold,
                self.minimum_threshold,
                self.maximum_threshold,
            )
        )

    # ============================================================
    # THRESHOLD CHECK
    # ============================================================

    def is_anomalous(
        self,
        score: float
    ) -> bool:
        """
        Return True when the score exceeds the adaptive threshold.
        """

        return float(score) >= self.threshold

    # ============================================================
    # CLASSIFICATION
    # ============================================================

    def classify(
        self,
        score: float
    ) -> str:
        """
        Classify a threat score.

        LOW:
            Below adaptive threshold.

        MEDIUM:
            Above threshold but below 0.70.

        HIGH:
            0.70 to below 0.85.

        CRITICAL:
            0.85 or higher.
        """

        score = float(score)

        if not np.isfinite(score):
            return "LOW"

        if score < self.threshold:
            return "LOW"

        if score >= 0.85:
            return "CRITICAL"

        if score >= 0.70:
            return "HIGH"

        return "MEDIUM"

    # ============================================================
    # BASELINE STATISTICS
    # ============================================================

    def statistics(self) -> dict:
        """
        Return current baseline statistics.
        """

        if not self.history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "threshold": self.threshold,
                "samples": 0,
                "minimum": 0.0,
                "maximum": 0.0,
            }

        scores = np.asarray(
            self.history,
            dtype=float,
        )

        return {
            "mean": float(
                np.mean(scores)
            ),
            "std": float(
                np.std(scores)
            ),
            "threshold": float(
                self.threshold
            ),
            "samples": int(
                len(scores)
            ),
            "minimum": float(
                np.min(scores)
            ),
            "maximum": float(
                np.max(scores)
            ),
        }

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        """
        Clear the adaptive history and return to the
        default threshold.
        """

        self.history = []

        self.threshold = (
            self.default_threshold
        )