from pathlib import Path

from src.evaluation import SecurityEvaluation


def main():

    project_root = Path(__file__).resolve().parent

    dataset_path = (
        project_root
        / "data"
        / "raw"
        / "security_events.csv"
    )

    output_dir = (
        project_root
        / "results"
    )

    print("=" * 70)
    print("QUANTUM-INSPIRED SECURITY SYSTEM")
    print("AUTOMATED EVALUATION")
    print("=" * 70)

    print(
        f"\nDataset: {dataset_path}"
    )

    evaluator = SecurityEvaluation(
        train_ratio=0.70,
        threshold_quantile=0.95
    )

    results = evaluator.evaluate(
        dataset_path
    )

    evaluator.save_results(
        results,
        output_dir
    )

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Total events : "
        f"{results['dataset_size']}"
    )

    print(
        f"Training     : "
        f"{results['training_size']}"
    )

    print(
        f"Testing      : "
        f"{results['test_size']}"
    )

    print(
        f"Features     : "
        f"{results['feature_count']}"
    )

    print("\n" + "=" * 70)
    print("DETECTOR PERFORMANCE")
    print("=" * 70)

    metrics = results["metrics"]

    display_columns = [
        "detector",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "roc_auc",
        "pr_auc",
        "threshold",
    ]

    print(
        metrics[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("ATTACK-WISE DETECTION")
    print("=" * 70)

    print(
        results[
            "attack_detection"
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("THRESHOLDS")
    print("=" * 70)

    for name, threshold in (
        results["thresholds"].items()
    ):
        print(
            f"{name:20s}: "
            f"{threshold:.6f}"
        )

    print("\n" + "=" * 70)
    print("RESULT FILES")
    print("=" * 70)

    for file in sorted(
        output_dir.iterdir()
    ):
        print(
            f"✓ {file.name}"
        )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()