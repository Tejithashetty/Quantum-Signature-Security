import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

EVENT_DATA_PATHS = [
    DATA_DIR / "security_events.csv",
    DATA_DIR / "raw" / "security_events.csv",
]

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"
ATTACK_PATH = RESULTS_DIR / "attack_detection.csv"
CONFUSION_PATH = RESULTS_DIR / "confusion_matrix.csv"
SCORES_PATH = RESULTS_DIR / "detector_scores.csv"
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Signature Security",
    page_icon="QS",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-note {
        color: #6b7280;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_json(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_event_dataset():
    candidates = []

    for path in EVENT_DATA_PATHS:
        if path.exists():
            try:
                candidate = pd.read_csv(path)
                candidates.append((path, candidate))
            except Exception:
                continue

    if not candidates:
        return None

    # Prefer a structured security-event dataset containing the expected
    # event fields. This prevents a malformed one-column CSV from being
    # selected simply because it appears first in the search list.
    preferred_columns = {
        "event_id",
        "timestamp",
        "message_size",
        "signature_size",
        "verification_success",
    }

    for _, candidate in candidates:
        if preferred_columns.intersection(candidate.columns):
            if len(candidate.columns) >= 5:
                return candidate

    # Otherwise use the widest available candidate.
    return max(candidates, key=lambda item: len(item[1].columns))[1]


def _normalize_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def first_existing(columns, candidates):
    # First try exact names.
    for candidate in candidates:
        if candidate in columns:
            return candidate

    # Then match common naming variations such as:
    # Quantum-Inspired_score, quantum_inspired_score,
    # Quantum-Inspired Score, etc.
    normalized_columns = {
        _normalize_column_name(column): column
        for column in columns
    }

    for candidate in candidates:
        normalized = _normalize_column_name(candidate)
        if normalized in normalized_columns:
            return normalized_columns[normalized]

    return None


def numeric_series(df, column):
    if column is None or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def format_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


# ============================================================
# LOAD DATA
# ============================================================

df = load_event_dataset()
metrics_df = load_csv(str(METRICS_PATH))
attack_df = load_csv(str(ATTACK_PATH))
confusion_df = load_csv(str(CONFUSION_PATH))
scores_df = load_csv(str(SCORES_PATH))
summary = load_json(str(SUMMARY_PATH))


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Quantum Signature Security</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Quantum-inspired cyber-threat detection for digital-signature security"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

pages = [
    "Overview",
    "Event Analysis",
    "Evaluation",
    "Attack Analysis",
    "Quantum Analysis",
    "Security Alerts",
    "Dataset",
    "About / Methodology",
]

selected_page = st.sidebar.radio("Go to", pages)

st.sidebar.divider()

st.sidebar.caption("Project")
st.sidebar.write("Quantum-inspired cyber-threat detection")

if df is not None:
    st.sidebar.metric("Events", len(df))

if metrics_df is not None:
    st.sidebar.metric("Evaluated detectors", len(metrics_df))


# ============================================================
# DATA AVAILABILITY
# ============================================================

if df is None:
    st.error(
        "Security event dataset was not found. Expected "
        "data/security_events.csv or data/raw/security_events.csv."
    )
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if selected_page == "Overview":
    st.header("Overview")
    st.write(
        "This dashboard presents the security-event dataset, detector "
        "outputs, evaluation results, attack-wise detection performance, "
        "and quantum-inspired analysis."
    )

    total_events = len(df)

    attack_col = first_existing(
        df.columns,
        ["attack_type", "attack", "label", "event_type"],
    )

    valid_col = first_existing(
        df.columns,
        ["verification_success", "signature_valid"],
    )

    replay_col = first_existing(
        df.columns,
        ["replay_count", "replay_intensity"],
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Events", total_events)

    with c2:
        if attack_col:
            st.metric("Attack Events", int((df[attack_col] != "NORMAL").sum()))
        else:
            st.metric("Attack Events", "N/A")

    with c3:
        if attack_col:
            st.metric("Normal Events", int((df[attack_col] == "NORMAL").sum()))
        else:
            st.metric("Normal Events", "N/A")

    with c4:
        if valid_col:
            valid_rate = numeric_series(df, valid_col).mean()
            st.metric("Verification Success", format_pct(valid_rate))
        else:
            st.metric("Verification Success", "N/A")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Event Distribution")

        if attack_col:
            distribution = (
                df[attack_col]
                .value_counts()
                .rename_axis("Attack Type")
                .to_frame("Events")
            )
            st.bar_chart(distribution)
        else:
            st.info("Attack-type column is not available.")

    with right:
        st.subheader("Signature Verification")

        if valid_col:
            verification = (
                numeric_series(df, valid_col)
                .map({1: "Valid", 0: "Invalid"})
                .value_counts()
                .rename_axis("Verification")
                .to_frame("Events")
            )
            st.bar_chart(verification)
        else:
            st.info("Verification column is not available.")

    st.divider()

    if scores_df is not None:
        st.subheader("Latest Detector Results")

        display_cols = [
            c for c in [
                "event_id",
                "attack_type",
                "statistical_score",
                "ml_score",
                "quantum_score",
                "hybrid_score",
                "threshold",
            ]
            if c in scores_df.columns
        ]

        if display_cols:
            st.dataframe(
                scores_df[display_cols].tail(10),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# EVENT ANALYSIS
# ============================================================

elif selected_page == "Event Analysis":
    st.header("Event Analysis")

    event_id_col = first_existing(
        df.columns,
        ["event_id", "id"],
    )

    if event_id_col is None:
        st.warning("No event ID column was found.")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        event_ids = df[event_id_col].astype(str).tolist()

        selected_id = st.selectbox(
            "Select an event",
            event_ids,
        )

        selected_event = df[
            df[event_id_col].astype(str) == selected_id
        ].iloc[0]

        attack_col = first_existing(
            df.columns,
            ["attack_type", "attack", "label", "event_type"],
        )

        st.subheader("Event Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Event ID",
                str(selected_event[event_id_col]),
            )

        with c2:
            st.metric(
                "Attack Type",
                str(selected_event[attack_col])
                if attack_col
                else "N/A",
            )

        with c3:
            verification_col = first_existing(
                df.columns,
                ["verification_success", "signature_valid"],
            )
            value = (
                selected_event[verification_col]
                if verification_col
                else "N/A"
            )
            st.metric("Verification", str(value))

        with c4:
            replay_col = first_existing(
                df.columns,
                ["replay_count", "replay_intensity"],
            )
            value = (
                selected_event[replay_col]
                if replay_col
                else "N/A"
            )
            st.metric("Replay Indicator", str(value))

        st.divider()

        st.subheader("Event Details")

        event_table = (
            selected_event
            .to_frame("Value")
            .reset_index()
            .rename(columns={"index": "Field"})
        )

        st.dataframe(
            event_table,
            use_container_width=True,
            hide_index=True,
        )

        if scores_df is not None and event_id_col in scores_df.columns:
            score_match = scores_df[
                scores_df[event_id_col].astype(str) == selected_id
            ]

            if not score_match.empty:
                st.subheader("Detector Scores")

                score_cols = [
                    c for c in [
                        "statistical_score",
                        "ml_score",
                        "quantum_score",
                        "hybrid_score",
                        "threshold",
                        "risk",
                        "severity",
                        "prediction",
                    ]
                    if c in score_match.columns
                ]

                if score_cols:
                    st.dataframe(
                        score_match[score_cols],
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# EVALUATION
# ============================================================

elif selected_page == "Evaluation":
    st.header("Detector Evaluation")

    if metrics_df is None:
        st.warning(
            "Evaluation results are not available yet. "
            "Run run_evaluation.py first."
        )
    else:
        st.write(
            "These values are loaded directly from the generated "
            "evaluation results."
        )

        st.subheader("Performance Metrics")

        metric_columns = [
            c for c in [
                "detector",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "roc_auc",
                "pr_auc",
                "threshold",
            ]
            if c in metrics_df.columns
        ]

        st.dataframe(
            metrics_df[metric_columns],
            use_container_width=True,
            hide_index=True,
        )

        score_metric = first_existing(
            metrics_df.columns,
            ["f1", "F1", "f1_score"],
        )

        if score_metric:
            chart_df = metrics_df.set_index("detector")[[score_metric]]
            st.subheader("F1 Comparison")
            st.bar_chart(chart_df)

        st.divider()

        st.subheader("Confusion Matrices")

        if confusion_df is not None:
            st.dataframe(
                confusion_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Confusion-matrix results are not available.")

        st.divider()

        if summary is not None:
            st.subheader("Evaluation Summary")
            st.json(summary)


# ============================================================
# ATTACK ANALYSIS
# ============================================================

elif selected_page == "Attack Analysis":
    st.header("Attack Analysis")

    if attack_df is None:
        st.warning(
            "Attack-wise evaluation results are not available yet."
        )
    else:
        st.subheader("Detection Rate by Attack Type")

        attack_columns = list(attack_df.columns)

        detector_col = first_existing(
            attack_columns,
            ["detector", "model", "method"],
        )

        attack_type_col = first_existing(
            attack_columns,
            ["attack_type", "attack", "label"],
        )

        rate_col = first_existing(
            attack_columns,
            [
                "detection_rate",
                "rate",
                "recall",
                "detected_rate",
            ],
        )

        if detector_col and attack_type_col and rate_col:
            attack_plot = attack_df.copy()
            attack_plot[rate_col] = pd.to_numeric(
                attack_plot[rate_col],
                errors="coerce",
            )

            pivot = attack_plot.pivot_table(
                index=attack_type_col,
                columns=detector_col,
                values=rate_col,
                aggfunc="mean",
            )

            st.bar_chart(pivot)

        st.dataframe(
            attack_df,
            use_container_width=True,
            hide_index=True,
        )

        if attack_type_col:
            st.divider()
            st.subheader("Attack Distribution")

            distribution = (
                attack_df[attack_type_col]
                .value_counts()
                .rename_axis("Attack Type")
                .to_frame("Records")
            )

            st.bar_chart(distribution)


# ============================================================
# QUANTUM ANALYSIS
# ============================================================

elif selected_page == "Quantum Analysis":
    st.header("Quantum-Inspired Analysis")

    st.write(
        "The quantum-inspired detector uses a classical representation "
        "inspired by probability amplitudes, entropy, state purity, "
        "concentration, and related state-distribution measures. "
        "It does not require a physical quantum computer."
    )

    if scores_df is None:
        st.warning(
            "Detector score results are not available. "
            "Run the evaluation pipeline first."
        )
    else:
        quantum_col = first_existing(
            scores_df.columns,
            [
                "quantum_score",
                "quantum_anomaly_score",
                "quantum_inspired_score",
            ],
        )

        attack_col = first_existing(
            scores_df.columns,
            ["attack_type", "attack", "label"],
        )

        if quantum_col:
            st.subheader("Quantum Score Distribution")

            chart = scores_df[[quantum_col]].copy()
            chart.index = np.arange(len(chart))
            st.line_chart(chart)

            if attack_col:
                st.subheader("Average Quantum Score by Event Type")

                grouped = (
                    scores_df
                    .groupby(attack_col)[quantum_col]
                    .mean()
                    .sort_values(ascending=False)
                    .to_frame("Average Quantum Score")
                )

                st.bar_chart(grouped)

        else:
            st.info(
                "No quantum score column was found in detector_scores.csv."
            )

        quantum_feature_columns = [
            c for c in scores_df.columns
            if "quantum" in c.lower()
            or "purity" in c.lower()
            or "amplitude" in c.lower()
            or "entropy" in c.lower()
            or "dimension" in c.lower()
        ]

        if quantum_feature_columns:
            st.divider()
            st.subheader("Available Quantum-Inspired Features")
            st.dataframe(
                scores_df[quantum_feature_columns],
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# SECURITY ALERTS
# ============================================================

elif selected_page == "Security Alerts":
    st.header("Security Alerts")

    if scores_df is None:
        st.warning(
            "Detector scores are not available."
        )
    else:
        hybrid_col = first_existing(
            scores_df.columns,
            [
                "hybrid_score",
                "Hybrid_score",
                "Hybrid Score",
                "hybrid_threat_score",
            ],
        )

        # detector_scores.csv contains detector scores and predictions,
        # while the calibrated detector thresholds are stored in
        # evaluation_metrics.csv. Obtain the Hybrid threshold from there.
        threshold_value = None

        if metrics_df is not None and hybrid_col:
            detector_col = first_existing(
                metrics_df.columns,
                ["detector", "model", "method"],
            )

            threshold_metric_col = first_existing(
                metrics_df.columns,
                ["threshold", "adaptive_threshold"],
            )

            if detector_col and threshold_metric_col:
                hybrid_rows = metrics_df[
                    metrics_df[detector_col].astype(str).str.strip().str.lower()
                    == "hybrid"
                ]

                if not hybrid_rows.empty:
                    threshold_value = pd.to_numeric(
                        hybrid_rows.iloc[0][threshold_metric_col],
                        errors="coerce",
                    )

        if hybrid_col and threshold_value is not None and not pd.isna(
            threshold_value
        ):
            alert_df = scores_df.copy()

            alert_df["alert_threshold"] = float(threshold_value)

            alert_df["alert_required"] = (
                pd.to_numeric(
                    alert_df[hybrid_col],
                    errors="coerce",
                ).fillna(0)
                >= float(threshold_value)
            )

            alert_only = alert_df[
                alert_df["alert_required"]
            ].copy()

            st.subheader("Alert Summary")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Events Requiring Alert",
                    len(alert_only),
                )

            with c2:
                st.metric(
                    "Events Below Threshold",
                    len(alert_df) - len(alert_only),
                )

            with c3:
                st.metric(
                    "Alert Rate",
                    format_pct(
                        len(alert_only) / len(alert_df)
                        if len(alert_df)
                        else 0
                    ),
                )

            with c4:
                st.metric(
                    "Hybrid Threshold",
                    f"{float(threshold_value):.4f}",
                )

            st.divider()

            if not alert_only.empty:
                display_cols = [
                    c for c in [
                        "event_id",
                        "attack_type",
                        "statistical_score",
                        "ml_score",
                        "quantum_score",
                        hybrid_col,
                        "alert_threshold",
                        "severity",
                        "risk",
                    ]
                    if c in alert_only.columns
                ]

                st.dataframe(
                    alert_only[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "No events exceeded their recorded adaptive threshold."
                )

        else:
            st.info(
                "Hybrid score data and the calibrated Hybrid threshold "
                "from evaluation_metrics.csv are required to generate "
                "alert analysis."
            )


# ============================================================
# DATASET
# ============================================================

elif selected_page == "Dataset":
    st.header("Security Event Dataset")

    attack_col = first_existing(
        df.columns,
        ["attack_type", "attack", "label", "event_type"],
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows", len(df))

    with c2:
        st.metric("Columns", len(df.columns))

    with c3:
        st.metric(
            "Attack Classes",
            df[attack_col].nunique()
            if attack_col
            else "N/A",
        )

    st.divider()

    st.subheader("Dataset Preview")

    if len(df.columns) == 1:
        st.warning(
            "The selected CSV contains only one column. "
            "The dashboard could not locate the structured security-event dataset."
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Column Information")

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[c].dtype)
                for c in df.columns
            ],
            "Missing Values": [
                int(df[c].isna().sum())
                for c in df.columns
            ],
        }
    )

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ABOUT / METHODOLOGY
# ============================================================

else:
    st.header("About / Methodology")

    st.subheader("Project Objective")

    st.write(
        "The project investigates quantum-inspired cyber-threat "
        "detection for digital-signature security by combining "
        "statistical anomaly detection, machine-learning anomaly "
        "detection, and a quantum-inspired feature representation."
    )

    st.subheader("Security Events")

    st.markdown(
        """
        - **NORMAL** — legitimate signed activity.
        - **TAMPERING** — the signed message is modified after signing.
        - **REPLAY** — a previously valid signed message is reused.
        - **VERIFICATION_BURST** — repeated rapid verification failures.
        """
    )

    st.subheader("Detection Pipeline")

    st.markdown(
        """
        1. Generate or collect digital-signature security events.
        2. Extract behavioral and signature-related security features.
        3. Calculate statistical anomaly evidence.
        4. Calculate machine-learning anomaly evidence.
        5. Encode features into a quantum-inspired state representation.
        6. Combine detector outputs using the hybrid threat score.
        7. Compare the threat score with an adaptive threshold.
        8. Generate an explainable security alert.
        9. Evaluate the detectors using held-out normal and attack events.
        """
    )

    st.subheader("Important Research Interpretation")

    st.info(
        "Quantum-inspired analysis in this project is a classical "
        "computational approach inspired by quantum concepts. "
        "It is not a physical quantum computer and it should not "
        "be described as post-quantum cryptography."
    )

    st.subheader("Evaluation Files")

    for path in [
        METRICS_PATH,
        ATTACK_PATH,
        CONFUSION_PATH,
        SCORES_PATH,
        SUMMARY_PATH,
    ]:
        status = "Available" if path.exists() else "Missing"
        st.write(f"{path.relative_to(PROJECT_ROOT)} — {status}")

    st.caption(
        "Dashboard data is loaded from the existing project data and "
        "evaluation result files. No new project directory is created."
    )
