from pathlib import Path

from datasets import load_dataset


def main():
    dataset_name = "openlifescienceai/Med-HALT"

    print(f"Downloading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)

    output_dir = Path("data/raw/medhalt")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Available splits:")
    print(dataset)

    # Save each split as parquet + csv
    for split_name, split_dataset in dataset.items():
        print(f"\nProcessing split: {split_name}")

        df = split_dataset.to_pandas()

        parquet_path = output_dir / f"{split_name}.parquet"
        csv_path = output_dir / f"{split_name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        print(f"Saved:")
        print(f"  {parquet_path}")
        print(f"  {csv_path}")

        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()