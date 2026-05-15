from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="openlifescienceai/medmcqa",
        config_name=None,
        output_dir="data/raw/medmcqa"
    )


if __name__ == "__main__":
    main()