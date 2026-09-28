from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


# ================================================================
# ENTROPY FUNCTIONS
# ================================================================

def byte_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of byte data.

    Returns a value between 0 and 8 for byte-level data.
    """

    if not data:
        return 0.0

    values = np.frombuffer(
        data,
        dtype=np.uint8
    )

    counts = np.bincount(
        values,
        minlength=256
    )

    probabilities = (
        counts[counts > 0] /
        len(values)
    )

    return float(
        -np.sum(
            probabilities *
            np.log2(probabilities)
        )
    )


def message_entropy(message: str) -> float:
    """
    Calculate entropy of a message string.
    """

    return byte_entropy(
        message.encode(
            errors="replace"
        )
    )


# ================================================================
# SAFE NUMERICAL HELPERS
# ================================================================

def _safe_numeric(
    series: pd.Series,
    default: float = 0.0
) -> pd.Series:
    """
    Convert a pandas series to numeric values safely.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(default)


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def create_detection_features(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert raw security events into numerical detection features.

    Existing project features are preserved and additional
    security-behavior features are added.
    """

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    result = df.copy()

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------

    required_columns = [
        "timestamp",
        "message",
        "message_size",
        "signature_size",
        "verification_failure",
        "verification_success",
        "replay_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ------------------------------------------------------------
    # Ensure numeric columns are numeric
    # ------------------------------------------------------------

    numeric_columns = [
        "timestamp",
        "message_size",
        "signature_size",
        "verification_failure",
        "verification_success",
        "replay_count",
    ]

    for column in numeric_columns:
        result[column] = _safe_numeric(
            result[column]
        )

    # ------------------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------------------

    result = result.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # ============================================================
    # 1. MESSAGE ENTROPY
    # ============================================================

    result["message_entropy"] = (
        result["message"]
        .astype(str)
        .apply(message_entropy)
    )

    # ============================================================
    # 2. REPLAY INTENSITY
    # ============================================================

    result["replay_intensity"] = (
        np.log1p(
            result["replay_count"]
        )
    )

    # ============================================================
    # 3. VERIFICATION FAILURE SIGNAL
    # ============================================================

    result["failure_signal"] = (
        result["verification_failure"]
    )

    # ============================================================
    # 4. SIGNATURE / MESSAGE SIZE RATIO
    # ============================================================

    result["signature_message_ratio"] = (
        result["signature_size"] /
        result["message_size"].clip(
            lower=1
        )
    )

    # ============================================================
    # 5. TIME GAP
    # ============================================================

    result["time_gap"] = (
        result["timestamp"]
        .diff()
        .fillna(10.0)
        .clip(
            lower=0
        )
    )

    # Avoid zero values because they would create
    # unrealistically large frequency scores.
    result["time_gap"] = result[
        "time_gap"
    ].clip(
        lower=0.01
    )

    # ============================================================
    # 6. FREQUENCY SCORE
    # ============================================================

    result["frequency_score"] = (
        1 /
        result["time_gap"]
    )

    # ============================================================
    # 7. SHA-256 DIGEST ENTROPY
    # ============================================================

    result["digest_entropy"] = (
        result["message"]
        .astype(str)
        .apply(
            lambda x: byte_entropy(
                hashlib.sha256(
                    x.encode(
                        errors="replace"
                    )
                ).digest()
            )
        )
    )

    # ============================================================
    # ADDITIONAL SECURITY FEATURES
    # ============================================================

    # ------------------------------------------------------------
    # 8. VERIFICATION ATTEMPT RATE
    # ------------------------------------------------------------

    if "verification_attempts" in result.columns:

        result[
            "verification_attempt_rate"
        ] = (
            result["verification_attempts"] /
            result["time_gap"].clip(
                lower=0.01
            )
        )

    else:

        result[
            "verification_attempt_rate"
        ] = (
            1 /
            result["time_gap"].clip(
                lower=0.01
            )
        )

    # ------------------------------------------------------------
    # 9. BURST INDICATOR
    # ------------------------------------------------------------

    # High frequency activity is represented by
    # a high frequency score.

    frequency_threshold = (
        result["frequency_score"]
        .quantile(0.75)
    )

    result["burst_indicator"] = (
        result["frequency_score"]
        >= frequency_threshold
    ).astype(int)

    # ------------------------------------------------------------
    # 10. REPLAY INDICATOR
    # ------------------------------------------------------------

    result["replay_indicator"] = (
        result["replay_count"] > 0
    ).astype(int)

    # ------------------------------------------------------------
    # 11. SIGNATURE VALIDITY SIGNAL
    # ------------------------------------------------------------

    if "signature_valid" in result.columns:

        result[
            "signature_validity_signal"
        ] = result["signature_valid"]

    else:

        result[
            "signature_validity_signal"
        ] = result[
            "verification_success"
        ]

    # ------------------------------------------------------------
    # 12. VERIFICATION FAILURE RATE
    # ------------------------------------------------------------

    if "verification_attempts" in result.columns:

        result[
            "verification_failure_rate"
        ] = (
            result["verification_failure"] /
            result["verification_attempts"].clip(
                lower=1
            )
        )

    else:

        result[
            "verification_failure_rate"
        ] = result[
            "verification_failure"
        ]

    # ------------------------------------------------------------
    # 13. MESSAGE SIZE DEVIATION
    # ------------------------------------------------------------

    normal_message_size = (
        result["message_size"]
        .median()
    )

    result[
        "message_size_deviation"
    ] = (
        abs(
            result["message_size"] -
            normal_message_size
        ) /
        max(
            normal_message_size,
            1
        )
    )

    # ============================================================
    # FINAL DETECTOR FEATURES
    # ============================================================

    feature_columns = [

        # Original project features
        "message_size",
        "signature_size",
        "message_entropy",
        "verification_failure",
        "replay_intensity",
        "signature_message_ratio",
        "frequency_score",
        "digest_entropy",

        # Additional security features
        "verification_attempt_rate",
        "burst_indicator",
        "replay_indicator",
        "signature_validity_signal",
        "verification_failure_rate",
        "message_size_deviation",
    ]

    # Replace possible infinite values.
    result[feature_columns] = (
        result[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    return result, feature_columns