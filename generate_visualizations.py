from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

metrics = pd.read_csv(RESULTS_DIR / "evaluation_metrics.csv")
attacks = pd.read_csv(RESULTS_DIR / "attack_detection.csv")
scores = pd.read_csv(RESULTS_DIR / "detector_scores.csv")


# ============================================================
# 1. DETECTOR PERFORMANCE COMPARISON
# ============================================================

performance = metrics.set_index("detector")[
    ["accuracy", "precision", "recall", "f1_score"]
]

ax = performance.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Detector Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Detector")
ax.set_ylim(0, 1.05)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "detector_performance_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. ROC-AUC AND PR-AUC
# ============================================================

auc = metrics.set_index("detector")[
    ["roc_auc", "pr_auc"]
]

ax = auc.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title("ROC-AUC and PR-AUC Comparison")
ax.set_ylabel("AUC")
ax.set_xlabel("Detector")
ax.set_ylim(0, 1.05)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "auc_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. ATTACK-WISE DETECTION RATE
# ============================================================

attack_pivot = attacks.pivot_table(
    index="attack_type",
    columns="detector",
    values="detection_rate",
    aggfunc="mean"
)

ax = attack_pivot.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Attack-wise Detection Rate")
ax.set_ylabel("Detection Rate")
ax.set_xlabel("Attack Type")
ax.set_ylim(0, 1.05)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "attack_wise_detection_rate.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. CONFUSION MATRICES
# ============================================================

for _, row in metrics.iterrows():

    detector = str(row["detector"])

    matrix = np.array([
        [
            row["true_negative"],
            row["false_positive"]
        ],
        [
            row["false_negative"],
            row["true_positive"]
        ]
    ])

    fig, ax = plt.subplots(
        figsize=(5.5, 5)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest"
    )

    ax.set_title(
        f"{detector} Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_xticks(
        [0, 1],
        ["Normal", "Attack"]
    )

    ax.set_yticks(
        [0, 1],
        ["Normal", "Attack"]
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    safe_name = (
        detector
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        FIGURES_DIR /
        f"{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 5. DETECTOR SCORE DISTRIBUTIONS
# ============================================================

score_columns = {
    "Statistical": "Statistical_score",
    "ML": "ML_score",
    "Quantum-Inspired": "Quantum-Inspired_score",
    "Hybrid": "Hybrid_score"
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

axes = axes.ravel()

for ax, (label, column) in zip(
    axes,
    score_columns.items()
):

    normal_scores = scores.loc[
        scores["attack_type"] == "NORMAL",
        column
    ]

    attack_scores = scores.loc[
        scores["attack_type"] != "NORMAL",
        column
    ]

    ax.hist(
        normal_scores,
        bins=15,
        alpha=0.7,
        label="NORMAL"
    )

    ax.hist(
        attack_scores,
        bins=15,
        alpha=0.7,
        label="ATTACK"
    )

    ax.set_title(
        f"{label} Score Distribution"
    )

    ax.set_xlabel(
        "Anomaly Score"
    )

    ax.set_ylabel(
        "Events"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "detector_score_distributions.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. HYBRID SCORE VS THRESHOLD
# ============================================================

hybrid_threshold = float(
    metrics.loc[
        metrics["detector"]
        .astype(str)
        .str.lower()
        == "hybrid",
        "threshold"
    ].iloc[0]
)

hybrid_scores = scores[
    "Hybrid_score"
].to_numpy()

event_numbers = np.arange(
    1,
    len(hybrid_scores) + 1
)

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    event_numbers,
    hybrid_scores,
    marker=".",
    linewidth=1,
    label="Hybrid Threat Score"
)

ax.axhline(
    hybrid_threshold,
    linestyle="--",
    linewidth=2,
    label=(
        f"Hybrid Threshold "
        f"({hybrid_threshold:.4f})"
    )
)

ax.set_title(
    "Hybrid Threat Score vs Adaptive Threshold"
)

ax.set_xlabel(
    "Evaluation Event"
)

ax.set_ylabel(
    "Hybrid Threat Score"
)

ax.set_ylim(
    0,
    1.05
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "hybrid_score_vs_threshold.png",
    dpi=300
)

plt.close()


print()
print("=" * 60)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 60)
print()
print(
    f"Output directory: {FIGURES_DIR}"
)
print()

for path in sorted(
    FIGURES_DIR.glob("*.png")
):
    print(path.name)