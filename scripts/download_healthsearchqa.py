from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="katielink/healthsearchqa",
        config_name="all_data",
        output_dir="data/raw/healthsearchqa"
    )


if __name__ == "__main__":
    main()