from pathlib import Path

from datasets import load_dataset


def download_hf_dataset(
    dataset_name: str,
    output_dir: str,
    config_name: str | None = None,
    save_csv: bool = True,
    save_parquet: bool = True
):
    """
    Download a Hugging Face dataset and save locally.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset name.

    output_dir : str
        Local directory to save files.

    config_name : str | None
        Dataset configuration name.

    save_csv : bool
        Whether to save CSV files.

    save_parquet : bool
        Whether to save parquet files.
    """

    print(f"\nDownloading dataset: {dataset_name}")

    if config_name:
        print(f"Using config: {config_name}")
        dataset = load_dataset(dataset_name, config_name)
    else:
        dataset = load_dataset(dataset_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nDataset structure:")
    print(dataset)

    for split_name, split_dataset in dataset.items():

        print(f"\nProcessing split: {split_name}")

        df = split_dataset.to_pandas()

        if save_csv:
            csv_path = output_path / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")

        if save_parquet:
            parquet_path = output_path / f"{split_name}.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"Saved parquet: {parquet_path}")

        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

    print("\nDownload complete.")

    return dataset