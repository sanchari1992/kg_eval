from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="qiaojin/PubMedQA",
        config_name="pqa_artificial",
        output_dir="data/raw/pubmedqa"
    )


if __name__ == "__main__":
    main()