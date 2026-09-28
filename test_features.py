import pandas as pd

from src.feature_engineering import (
    create_detection_features
)


def main():

    df = pd.read_csv(
        "data/raw/security_events.csv"
    )

    processed, columns = (
        create_detection_features(df)
    )

    print("\nDetection features:")
    print("=" * 60)

    print(columns)

    print("\nFirst five events:")

    print(
        processed[
            [
                "event_id",
                "attack_type",
                *columns
            ]
        ].head()
    )


if __name__ == "__main__":
    main()