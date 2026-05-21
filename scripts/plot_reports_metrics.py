# scripts/plot_report_metrics.py

import os
import json
import matplotlib.pyplot as plt

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "metric_plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CUSTOM DATASET ORDER
# =========================

DATASET_ORDER = [
    # Scientific datasets
    "pubmedqa",
    "medhallu",
    "medhalt",
    "bioasq",
    "medmcqa",
    "medqa",
    "medrevqa",
    "medchangeqa",
    "covidqa",

    # Consumer datasets
    "healthbench",
    "biqa",
    "medaesqa",
    "medquad",
    "mediqa",
    "healthsearchqa",
    "medicationqa",
]

# Scientific vs Consumer colors
SCIENTIFIC_COLOR = "steelblue"
CONSUMER_COLOR = "darkorange"

SCIENTIFIC_DATASETS = {
    "pubmedqa",
    "medhallu",
    "medhalt",
    "bioasq",
    "medmcqa",
    "medqa",
    "medrevqa",
    "medchangeqa",
    "covidqa",
}

# =========================
# LOAD METRICS
# =========================

all_metrics = {}

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

            if not isinstance(metric_value, (int, float)):
                continue

            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}

            all_metrics[metric_name][folder_name] = metric_value

    except Exception as e:
        print(f"[ERROR] Failed reading {report_path}: {e}")

# =========================
# PLOT METRICS
# =========================

for metric_name, dataset_values in all_metrics.items():

    ordered_folders = []
    ordered_values = []
    bar_colors = []

    for dataset_name in DATASET_ORDER:

        if dataset_name not in dataset_values:
            continue

        ordered_folders.append(dataset_name)
        ordered_values.append(dataset_values[dataset_name])

        if dataset_name in SCIENTIFIC_DATASETS:
            bar_colors.append(SCIENTIFIC_COLOR)
        else:
            bar_colors.append(CONSUMER_COLOR)

    plt.figure(figsize=(14, 6))

    plt.bar(
        ordered_folders,
        ordered_values,
        color=bar_colors
    )

    plt.xlabel("Datasets")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Across Medical QA Benchmarks")

    plt.xticks(rotation=45, ha="right")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=SCIENTIFIC_COLOR, label="Scientific Datasets"),
        Patch(facecolor=CONSUMER_COLOR, label="Consumer Datasets")
    ]

    plt.legend(handles=legend_elements)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{metric_name}.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[SAVED] {output_path}")

print("\nAll metric plots generated successfully.")