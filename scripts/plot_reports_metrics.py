# scripts/plot_report_metrics.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch


# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "metric_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# CUSTOM DATASET ORDER
# =========================

DATASET_ORDER = [

    # ---------------------------------
    # Scientific datasets
    # ---------------------------------

    "pubmedqa",
    "medhallu",
    "medhalt",
    "bioasq",
    "medmcqa",
    "medqa",
    "medrevqa",
    "medchangeqa",
    "covidqa",

    # ---------------------------------
    # Consumer datasets
    # ---------------------------------

    "healthbench",
    "biqa",
    "medaesqa",
    "medquad",
    "mediqa",
    "healthsearchqa",
    "medicationqa",
    "healthbench"
]


# =========================
# COLORS
# =========================

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
# CONFERENCE-WORTHY METRICS
# =========================

KEY_METRICS = {

    "avg_entity_interaction_density":
        "Entity Interaction Density",

    "avg_reasoning_path_complexity":
        "Reasoning Path Complexity",

    "avg_context_expansion_ratio":
        "Context Expansion Ratio",

    "avg_ontology_depth_score":
        "Ontology Depth Score",
}


# =========================
# LOAD METRICS
# =========================

all_metrics = {}

for folder_name in os.listdir(RAW_DIR):

    folder_path = os.path.join(
        RAW_DIR,
        folder_name
    )

    if not os.path.isdir(folder_path):
        continue

    report_path = os.path.join(
        folder_path,
        "report.json"
    )

    if not os.path.exists(report_path):

        print(
            f"[WARNING] report.json not found "
            f"in {folder_name}"
        )

        continue

    try:

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as f:

            report_data = json.load(f)

        summary = report_data.get(
            "summary",
            {}
        )

        for metric_name, metric_value in summary.items():

            if not isinstance(
                metric_value,
                (int, float)
            ):
                continue

            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}

            all_metrics[metric_name][
                folder_name
            ] = metric_value

    except Exception as e:

        print(
            f"[ERROR] Failed reading "
            f"{report_path}: {e}"
        )


# ======================================================
# GENERIC BAR PLOTS
# ======================================================

for metric_name, dataset_values in all_metrics.items():

    ordered_folders = []
    ordered_values = []
    bar_colors = []

    for dataset_name in DATASET_ORDER:

        if dataset_name not in dataset_values:
            continue

        ordered_folders.append(dataset_name)

        ordered_values.append(
            dataset_values[dataset_name]
        )

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

    plt.title(
        f"{metric_name} Across Medical QA Benchmarks"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    legend_elements = [

        Patch(
            facecolor=SCIENTIFIC_COLOR,
            label="Scientific Datasets"
        ),

        Patch(
            facecolor=CONSUMER_COLOR,
            label="Consumer Datasets"
        )
    ]

    plt.legend(handles=legend_elements)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{metric_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"[SAVED] {output_path}")


# ======================================================
# SPECIALIZED CONFERENCE PLOTS
# ======================================================

for metric_key, metric_title in KEY_METRICS.items():

    if metric_key not in all_metrics:

        print(
            f"[WARNING] Missing metric: "
            f"{metric_key}"
        )

        continue

    dataset_values = all_metrics[metric_key]

    ordered_folders = []
    ordered_values = []
    bar_colors = []

    scientific_values = []
    consumer_values = []

    for dataset_name in DATASET_ORDER:

        if dataset_name not in dataset_values:
            continue

        value = dataset_values[dataset_name]

        ordered_folders.append(dataset_name)

        ordered_values.append(value)

        if dataset_name in SCIENTIFIC_DATASETS:

            scientific_values.append(value)

            bar_colors.append(
                SCIENTIFIC_COLOR
            )

        else:

            consumer_values.append(value)

            bar_colors.append(
                CONSUMER_COLOR
            )

    # --------------------------------------------------
    # Main comparison plot
    # --------------------------------------------------

    plt.figure(figsize=(16, 7))

    bars = plt.bar(
        ordered_folders,
        ordered_values,
        color=bar_colors
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=10
    )

    plt.ylabel(
        metric_title,
        fontsize=12
    )

    plt.xlabel(
        "Medical QA Datasets",
        fontsize=12
    )

    plt.title(
        f"{metric_title}: "
        f"Scientific vs Consumer QA",
        fontsize=14,
        fontweight="bold"
    )

    # value labels
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

    legend_elements = [

        Patch(
            facecolor=SCIENTIFIC_COLOR,
            label="Scientific / Generated"
        ),

        Patch(
            facecolor=CONSUMER_COLOR,
            label="Consumer-Collected"
        )
    ]

    plt.legend(handles=legend_elements)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"conference_{metric_key}.png"
    )

    plt.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVED] conference plot: "
        f"{output_path}"
    )

    # --------------------------------------------------
    # Aggregate category comparison
    # --------------------------------------------------

    plt.figure(figsize=(6, 6))

    categories = [
        "Scientific",
        "Consumer"
    ]

    means = [

        np.mean(scientific_values)
        if scientific_values else 0,

        np.mean(consumer_values)
        if consumer_values else 0
    ]

    stds = [

        np.std(scientific_values)
        if scientific_values else 0,

        np.std(consumer_values)
        if consumer_values else 0
    ]

    plt.bar(
        categories,
        means,
        yerr=stds,
        capsize=8,
        color=[
            SCIENTIFIC_COLOR,
            CONSUMER_COLOR
        ]
    )

    plt.ylabel(metric_title)

    plt.title(
        f"Average {metric_title}\n"
        f"Scientific vs Consumer QA",
        fontsize=13,
        fontweight="bold"
    )

    for i, mean_val in enumerate(means):

        plt.text(
            i,
            mean_val,
            f"{mean_val:.2f}",
            ha='center',
            va='bottom',
            fontsize=11
        )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"grouped_{metric_key}.png"
    )

    plt.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVED] grouped plot: "
        f"{output_path}"
    )


# ======================================================
# COMBINED RADAR-LIKE NORMALIZED COMPARISON
# ======================================================

conference_metrics = list(KEY_METRICS.keys())

scientific_means = []
consumer_means = []

for metric_key in conference_metrics:

    if metric_key not in all_metrics:
        continue

    dataset_values = all_metrics[metric_key]

    sci_vals = []
    con_vals = []

    for dataset_name, value in dataset_values.items():

        if dataset_name in SCIENTIFIC_DATASETS:
            sci_vals.append(value)
        else:
            con_vals.append(value)

    scientific_means.append(np.mean(sci_vals))
    consumer_means.append(np.mean(con_vals))


# normalize
all_vals = scientific_means + consumer_means

global_max = max(all_vals) if all_vals else 1

scientific_norm = [
    v / global_max
    for v in scientific_means
]

consumer_norm = [
    v / global_max
    for v in consumer_means
]

x = np.arange(len(conference_metrics))

width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width/2,
    scientific_norm,
    width,
    label="Scientific",
    color=SCIENTIFIC_COLOR
)

plt.bar(
    x + width/2,
    consumer_norm,
    width,
    label="Consumer",
    color=CONSUMER_COLOR
)

metric_labels = [

    KEY_METRICS[m]
    for m in conference_metrics
]

plt.xticks(
    x,
    metric_labels,
    rotation=20
)

plt.ylabel("Normalized Complexity Score")

plt.title(
    "Normalized Benchmark Complexity Comparison",
    fontsize=14,
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "normalized_complexity_comparison.png"
)

plt.savefig(
    output_path,
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print(
    f"[SAVED] normalized comparison plot: "
    f"{output_path}"
)

print("\nAll metric plots generated successfully.")