import numpy as np

from src.hybrid_detector import HybridThreatDetector
from src.adaptive_threshold import AdaptiveThreshold


def create_demo_dataset():

    rng = np.random.default_rng(42)

    normal = rng.normal(
        loc=0,
        scale=1,
        size=(500, 8)
    )

    attacks = rng.normal(
        loc=3,
        scale=1.5,
        size=(30, 8)
    )

    return normal, attacks


def main():

    normal, attacks = create_demo_dataset()

    detector = HybridThreatDetector()

    detector.fit(normal)

    threshold = AdaptiveThreshold()

    print("\nQuantum-Inspired Digital Signature Security")
    print("=" * 50)

    for i, event in enumerate(attacks[:10]):

        result = detector.predict(event)

        score = result["hybrid_score"]

        threshold.update(score)

        classification = threshold.classify(
            score
        )

        print(
            f"\nEvent {i + 1}"
        )

        print(
            f"Statistical : "
            f"{result['statistical_score']:.3f}"
        )

        print(
            f"ML          : "
            f"{result['ml_score']:.3f}"
        )

        print(
            f"Quantum     : "
            f"{result['quantum_score']:.3f}"
        )

        print(
            f"Hybrid      : "
            f"{score:.3f}"
        )

        print(
            f"Threshold   : "
            f"{threshold.threshold:.3f}"
        )

        print(
            f"Risk        : "
            f"{classification}"
        )


if __name__ == "__main__":
    main()