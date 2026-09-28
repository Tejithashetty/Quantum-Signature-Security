from pathlib import Path

from src.event_generator import SecurityEventGenerator


def main():

    output_dir = Path("data/raw")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        "security_events.csv"
    )

    generator = SecurityEventGenerator()

    df = generator.generate_dataset(
        normal_count=100,
        tampering_count=30,
        replay_count=30,
        burst_count=20,
    )

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("SECURITY DATASET GENERATED")
    print("=" * 60)

    print(f"\nLocation: {output_file}")

    print(
        f"\nTotal events: {len(df)}"
    )

    print("\nEvent distribution:")

    print(
        df["attack_type"].value_counts()
    )

    print("\nVerification results:")

    print(
        df["verification_success"]
        .value_counts()
    )


if __name__ == "__main__":
    main()