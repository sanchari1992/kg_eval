from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="lavita/MedQuAD",
        config_name=None,
        output_dir="data/raw/medquad"
    )


if __name__ == "__main__":
    main()