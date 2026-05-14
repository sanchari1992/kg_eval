from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="GM07/medhal",
        config_name="default",
        output_dir="data/raw/medhal"
    )


if __name__ == "__main__":
    main()