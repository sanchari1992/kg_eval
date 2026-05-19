from kg_eval.ingestion.hf_loader import download_hf_dataset


def main():

    download_hf_dataset(
        dataset_name="bigbio/mediqa_qa",
        config_name="mediqa_qa_bigbio_qa",
        output_dir="data/raw/mediqa"
    )


if __name__ == "__main__":
    main()