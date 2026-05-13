from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="UTAustin-AIHealth/MedHallu",
        config_name="pqa_labeled",
        output_dir="data/raw/medhallu"
    )


if __name__ == "__main__":
    main()