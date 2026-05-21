from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="openai/healthbench",
        config_name=None,
        output_dir="data/raw/healthbench"
    )


if __name__ == "__main__":
    main()