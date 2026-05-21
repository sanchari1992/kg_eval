# scripts/plot_report_metrics.py

import os
import json
import matplotlib.pyplot as plt

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "metric_plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store metric values across folders
all_metrics = {}

# Traverse each dataset folder
for folder_name in os.listdir(RAW_DIR):
    folder_path = os.path.join(RAW_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    report_path = os.path.join(folder_path, "report.json")

    if not os.path.exists(report_path):
        print(f"[WARNING] report.json not found in {folder_name}")
        continue

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        summary = report_data.get("summary", {})

        for metric_name, metric_value in summary.items():

            # Skip non-numeric metrics
            if not isinstance(metric_value, (int, float)):
                continue

            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}

            all_metrics[metric_name][folder_name] = metric_value

    except Exception as e:
        print(f"[ERROR] Failed reading {report_path}: {e}")

# Create one graph per metric
for metric_name, dataset_values in all_metrics.items():

    folders = list(dataset_values.keys())
    values = list(dataset_values.values())

    plt.figure(figsize=(12, 6))
    plt.bar(folders, values)

    plt.xlabel("Dataset Folder")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Across Datasets")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{metric_name}.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(f"[SAVED] {output_path}")

print("\nAll metric plots generated successfully.")