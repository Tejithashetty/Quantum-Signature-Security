"""
Evaluation framework for the Quantum-Inspired Digital Signature
Security Threat Detection System.

Evaluation protocol:

1. Split the dataset into NORMAL training events and a held-out
   test set containing NORMAL + attack events.
2. Fit Statistical and ML detectors using NORMAL training data.
3. Generate detector scores for training and test events.
4. Calibrate detector-specific thresholds using NORMAL training scores.
5. Evaluate binary anomaly detection:

       NORMAL = 0
       ATTACK = 1

6. Calculate:
       - Accuracy
       - Precision
       - Recall
       - F1-score
       - False Positive Rate
       - ROC-AUC
       - PR-AUC
       - Confusion matrix
       - Attack-wise detection rate
7. Save evaluation results as CSV and JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .feature_engineering import create_detection_features
from .hybrid_detector import HybridThreatDetector


class SecurityEvaluation:
    """
    Complete evaluation pipeline for the hybrid security detector.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        threshold_quantile: float = 0.95,
    ):
        if not 0.5 <= train_ratio < 1.0:
            raise ValueError(
                "train_ratio must be between 0.5 and 1.0."
            )

        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError(
                "threshold_quantile must be between 0.5 and 1.0."
            )

        self.train_ratio = train_ratio
        self.threshold_quantile = threshold_quantile

        self.detector = HybridThreatDetector()

        self.feature_columns = []
        self.thresholds = {}

        self.train_df = None
        self.test_df = None

        self.train_features = None
        self.test_features = None

        self.train_scores = {}
        self.test_scores = {}

    # =========================================================
    # DATA PREPARATION
    # =========================================================

    def load_dataset(
        self,
        path: str | Path,
    ) -> pd.DataFrame:
        """
        Load and validate the security event dataset.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        df = pd.read_csv(path)

        required_columns = {
            "event_id",
            "timestamp",
            "attack_type",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Dataset is missing columns: {sorted(missing)}"
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = (
            df.dropna(
                subset=["timestamp"]
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return df

    # =========================================================
    # TRAIN / TEST SPLIT
    # =========================================================

    def split_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create an evaluation split where:

        Training:
            NORMAL events only.

        Testing:
            Held-out NORMAL events +
            all attack categories.

        This prevents the chronological ordering of the
        synthetic dataset from causing the test set to
        contain attacks only.
        """

        normal_df = df[
            df["attack_type"] == "NORMAL"
        ].copy()

        attack_df = df[
            df["attack_type"] != "NORMAL"
        ].copy()

        if normal_df.empty:
            raise ValueError(
                "Dataset contains no NORMAL events."
            )

        if attack_df.empty:
            raise ValueError(
                "Dataset contains no attack events."
            )

        # Shuffle NORMAL events before splitting.
        normal_df = normal_df.sample(
            frac=1.0,
            random_state=42,
        ).reset_index(drop=True)

        # Calculate number of NORMAL training events.
        normal_train_count = int(
            len(normal_df) * self.train_ratio
        )

        if normal_train_count <= 0:
            raise ValueError(
                "Not enough NORMAL events for training."
            )

        # NORMAL training baseline.
        train_df = normal_df[
            :normal_train_count
        ].copy()

        # Held-out NORMAL events.
        test_normal_df = normal_df[
            normal_train_count:
        ].copy()

        # All attack events are included in the test set.
        test_df = pd.concat(
            [
                test_normal_df,
                attack_df,
            ],
            ignore_index=True,
        )

        # Shuffle test events.
        test_df = test_df.sample(
            frac=1.0,
            random_state=42,
        ).reset_index(drop=True)

        # Keep training data chronological.
        train_df = (
            train_df.sort_values("timestamp")
            .reset_index(drop=True)
        )

        return train_df, test_df

    # =========================================================
    # FIT DETECTORS
    # =========================================================

    def fit(
        self,
        train_features: pd.DataFrame,
        feature_columns: list[str],
        train_labels: pd.Series,
    ):
        """
        Fit the statistical and ML detectors using NORMAL
        training events only.
        """

        self.feature_columns = feature_columns

        normal_mask = (
            train_labels == "NORMAL"
        )

        normal_features = train_features.loc[
            normal_mask,
            feature_columns,
        ].copy()

        if normal_features.empty:
            raise ValueError(
                "No NORMAL events available for training."
            )

        self.detector.fit(
            normal_features
        )

    # =========================================================
    # SCORE EVENTS
    # =========================================================

    def _score_dataframe(
        self,
        features: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Generate scores from all detector components.
        """

        X = features[
            self.feature_columns
        ].to_numpy(
            dtype=float
        )

        statistical_scores = []
        ml_scores = []
        quantum_scores = []
        hybrid_scores = []

        for row in X:

            result = self.detector.predict(
                row
            )

            statistical_scores.append(
                result["statistical_score"]
            )

            ml_scores.append(
                result["ml_score"]
            )

            quantum_scores.append(
                result["quantum_score"]
            )

            hybrid_scores.append(
                result["hybrid_score"]
            )

        return {
            "Statistical": np.asarray(
                statistical_scores
            ),
            "ML": np.asarray(
                ml_scores
            ),
            "Quantum-Inspired": np.asarray(
                quantum_scores
            ),
            "Hybrid": np.asarray(
                hybrid_scores
            ),
        }

    # =========================================================
    # THRESHOLD CALIBRATION
    # =========================================================

    def calibrate_thresholds(
        self,
        train_scores: Dict[str, np.ndarray],
        train_labels: pd.Series,
    ):
        """
        Calculate detector-specific anomaly thresholds
        using only NORMAL training scores.

        The threshold is the selected upper quantile of
        the NORMAL training baseline.
        """

        normal_mask = (
            train_labels.to_numpy()
            == "NORMAL"
        )

        for detector_name, scores in train_scores.items():

            normal_scores = scores[
                normal_mask
            ]

            if len(normal_scores) == 0:
                raise ValueError(
                    f"No NORMAL scores for {detector_name}."
                )

            threshold = float(
                np.quantile(
                    normal_scores,
                    self.threshold_quantile,
                )
            )

            self.thresholds[
                detector_name
            ] = threshold

    # =========================================================
    # PREDICTIONS
    # =========================================================

    @staticmethod
    def _binary_predictions(
        scores: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Convert anomaly scores into binary predictions.

        0 = NORMAL
        1 = ATTACK
        """

        return (
            scores >= threshold
        ).astype(int)

    # =========================================================
    # METRICS
    # =========================================================

    @staticmethod
    def _calculate_metrics(
        y_true: np.ndarray,
        scores: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict:
        """
        Calculate binary classification metrics.
        """

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        total = tn + fp + fn + tp

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        # ROC-AUC requires both classes to exist.
        if len(np.unique(y_true)) == 2:
            roc_auc = float(
                roc_auc_score(
                    y_true,
                    scores,
                )
            )
        else:
            roc_auc = None

        # PR-AUC can still be calculated when positive
        # examples exist.
        if np.sum(y_true) > 0:
            pr_auc = float(
                average_precision_score(
                    y_true,
                    scores,
                )
            )
        else:
            pr_auc = None

        metrics = {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            "false_positive_rate": float(
                false_positive_rate
            ),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "total": int(total),
        }

        return metrics

    # =========================================================
    # ATTACK-WISE ANALYSIS
    # =========================================================

    def attack_detection_rates(
        self,
        test_df: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Calculate detection rate separately for every
        attack category.
        """

        rows = []

        attack_types = [
            attack
            for attack in sorted(
                test_df["attack_type"].unique()
            )
            if attack != "NORMAL"
        ]

        for (
            detector_name,
            detector_predictions,
        ) in predictions.items():

            for attack_type in attack_types:

                mask = (
                    test_df[
                        "attack_type"
                    ].to_numpy()
                    == attack_type
                )

                total = int(
                    mask.sum()
                )

                detected = int(
                    detector_predictions[
                        mask
                    ].sum()
                )

                rate = (
                    detected / total
                    if total > 0
                    else 0.0
                )

                rows.append(
                    {
                        "detector": detector_name,
                        "attack_type": attack_type,
                        "total_events": total,
                        "detected_events": detected,
                        "detection_rate": rate,
                    }
                )

        return pd.DataFrame(rows)

    # =========================================================
    # FULL EVALUATION
    # =========================================================

    def evaluate(
        self,
        dataset_path: str | Path,
    ) -> Dict:
        """
        Execute the complete evaluation pipeline.
        """

        # -----------------------------------------------------
        # Load dataset
        # -----------------------------------------------------

        df = self.load_dataset(
            dataset_path
        )

        # -----------------------------------------------------
        # Feature engineering
        # -----------------------------------------------------

        feature_df, feature_columns = (
            create_detection_features(
                df
            )
        )

        self.feature_columns = (
            feature_columns
        )

        # -----------------------------------------------------
        # Train / test split
        # -----------------------------------------------------

        train_df, test_df = (
            self.split_dataset(
                feature_df
            )
        )

        self.train_df = train_df
        self.test_df = test_df

        train_features = train_df[
            feature_columns
        ]

        test_features = test_df[
            feature_columns
        ]

        self.train_features = (
            train_features
        )

        self.test_features = (
            test_features
        )

        # -----------------------------------------------------
        # Fit detectors
        # -----------------------------------------------------

        self.fit(
            train_features,
            feature_columns,
            train_df["attack_type"],
        )

        # -----------------------------------------------------
        # Generate training scores
        # -----------------------------------------------------

        self.train_scores = (
            self._score_dataframe(
                train_features
            )
        )

        # -----------------------------------------------------
        # Generate test scores
        # -----------------------------------------------------

        self.test_scores = (
            self._score_dataframe(
                test_features
            )
        )

        # -----------------------------------------------------
        # Calibrate thresholds using NORMAL training data
        # -----------------------------------------------------

        self.calibrate_thresholds(
            self.train_scores,
            train_df["attack_type"],
        )

        # -----------------------------------------------------
        # Ground-truth labels
        #
        # NORMAL = 0
        # ATTACK = 1
        # -----------------------------------------------------

        y_true = (
            test_df["attack_type"]
            != "NORMAL"
        ).astype(int).to_numpy()

        all_metrics = []
        predictions = {}

        # -----------------------------------------------------
        # Evaluate every detector
        # -----------------------------------------------------

        for detector_name, scores in (
            self.test_scores.items()
        ):

            threshold = self.thresholds[
                detector_name
            ]

            detector_predictions = (
                self._binary_predictions(
                    scores,
                    threshold,
                )
            )

            predictions[
                detector_name
            ] = detector_predictions

            metrics = (
                self._calculate_metrics(
                    y_true,
                    scores,
                    detector_predictions,
                )
            )

            metrics[
                "detector"
            ] = detector_name

            metrics[
                "threshold"
            ] = threshold

            all_metrics.append(
                metrics
            )

        metrics_df = pd.DataFrame(
            all_metrics
        )

        # -----------------------------------------------------
        # Attack-wise detection
        # -----------------------------------------------------

        attack_df = (
            self.attack_detection_rates(
                test_df,
                predictions,
            )
        )

        # -----------------------------------------------------
        # Save individual detector scores
        # -----------------------------------------------------

        score_df = test_df[
            [
                "event_id",
                "timestamp",
                "attack_type",
            ]
        ].copy()

        for detector_name, scores in (
            self.test_scores.items()
        ):

            score_df[
                f"{detector_name}_score"
            ] = scores

            score_df[
                f"{detector_name}_prediction"
            ] = predictions[
                detector_name
            ]

        # -----------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------

        confusion_rows = []

        for detector_name, pred in (
            predictions.items()
        ):

            cm = confusion_matrix(
                y_true,
                pred,
                labels=[0, 1],
            )

            confusion_rows.extend(
                [
                    {
                        "detector": detector_name,
                        "actual": "NORMAL",
                        "predicted": "NORMAL",
                        "count": int(
                            cm[0, 0]
                        ),
                    },
                    {
                        "detector": detector_name,
                        "actual": "NORMAL",
                        "predicted": "ATTACK",
                        "count": int(
                            cm[0, 1]
                        ),
                    },
                    {
                        "detector": detector_name,
                        "actual": "ATTACK",
                        "predicted": "NORMAL",
                        "count": int(
                            cm[1, 0]
                        ),
                    },
                    {
                        "detector": detector_name,
                        "actual": "ATTACK",
                        "predicted": "ATTACK",
                        "count": int(
                            cm[1, 1]
                        ),
                    },
                ]
            )

        confusion_df = pd.DataFrame(
            confusion_rows
        )

        # -----------------------------------------------------
        # Return complete results
        # -----------------------------------------------------

        return {
            "dataset_size": len(df),
            "training_size": len(train_df),
            "test_size": len(test_df),
            "feature_count": len(
                feature_columns
            ),
            "feature_columns": feature_columns,
            "metrics": metrics_df,
            "attack_detection": attack_df,
            "confusion_matrix": confusion_df,
            "scores": score_df,
            "thresholds": self.thresholds,
        }

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    def save_results(
        self,
        results: Dict,
        output_dir: str | Path = "results",
    ):
        """
        Save evaluation results to CSV and JSON files.
        """

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        results[
            "metrics"
        ].to_csv(
            output_dir
            / "evaluation_metrics.csv",
            index=False,
        )

        # -----------------------------------------------------
        # Attack detection
        # -----------------------------------------------------

        results[
            "attack_detection"
        ].to_csv(
            output_dir
            / "attack_detection.csv",
            index=False,
        )

        # -----------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------

        results[
            "confusion_matrix"
        ].to_csv(
            output_dir
            / "confusion_matrix.csv",
            index=False,
        )

        # -----------------------------------------------------
        # Detector scores
        # -----------------------------------------------------

        results[
            "scores"
        ].to_csv(
            output_dir
            / "detector_scores.csv",
            index=False,
        )

        # -----------------------------------------------------
        # Summary JSON
        # -----------------------------------------------------

        summary = {
            "dataset_size": results[
                "dataset_size"
            ],
            "training_size": results[
                "training_size"
            ],
            "test_size": results[
                "test_size"
            ],
            "feature_count": results[
                "feature_count"
            ],
            "feature_columns": results[
                "feature_columns"
            ],
            "thresholds": results[
                "thresholds"
            ],
        }

        with open(
            output_dir
            / "evaluation_summary.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )