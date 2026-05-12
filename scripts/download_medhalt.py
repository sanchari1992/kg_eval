from pathlib import Path

from datasets import load_dataset


def main():
    dataset_name = "openlifescienceai/Med-HALT"
    config_name = "reasoning_FCT"

    print(f"Downloading dataset: {dataset_name}")
    print(f"Using config: {config_name}")

    dataset = load_dataset(
        dataset_name,
        config_name
    )

    output_dir = Path("data/raw/medhalt")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(dataset)

    for split_name, split_dataset in dataset.items():
        df = split_dataset.to_pandas()

        parquet_path = output_dir / f"{split_name}.parquet"
        csv_path = output_dir / f"{split_name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        print(f"Saved {split_name}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()