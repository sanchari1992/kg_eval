import json
import pandas as pd


def jsonl_to_csv(jsonl_path, csv_path):
    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False)

    print(f"Converted {jsonl_path} -> {csv_path}")


if __name__ == "__main__":

    jsonl_to_csv(
        jsonl_path="data/raw/medqa/US_qbank.jsonl",
        csv_path="data.csv"
    )