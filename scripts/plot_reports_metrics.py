import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "metric_plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# DATASET ORDER
# =========================

DATASET_ORDER = [
    "pubmedqa",
    "medhallu",
    # "medhalt",
    "bioasq",
    "medmcqa",
    "medrevqa",
    "medchangeqa",
    "covidqa",
    "biqa",
    "medaesqa",
    "medquad",
    "mediqa",
    "healthsearchqa",
    "medicationqa",
    "healthbench",
]

# Existing benchmarks = all traditional benchmarks
EXISTING_BENCHMARKS = set(DATASET_ORDER[:13])

# HealthBench benchmark
HEALTHBENCH_BENCHMARK = set(DATASET_ORDER[13:])


# =========================
# METRICS
# =========================

KEY_METRICS = {

    # -----------------------------------------
    # Structural-semantic metrics
    # -----------------------------------------

    "avg_entity_interaction_density":
        "Entity Interaction Density",

    "avg_reasoning_path_complexity":
        "Reasoning Path Complexity",

    "avg_context_expansion_ratio":
        "Context Expansion Ratio",

    # -----------------------------------------
    # Grammar / linguistic metrics
    # -----------------------------------------

    "avg_sentence_complexity":
        "Sentence Complexity",

    "avg_lexical_diversity":
        "Lexical Diversity",

    "avg_token_length":
        "Average Token Length",

    "avg_content_word_ratio":
        "Content Word Ratio",

    "avg_clause_proxy_complexity":
        "Clause Complexity Proxy",

    # -----------------------------------------
    # Novel paper metric
    # -----------------------------------------

    "avg_clinical_linguistic_complexity_index":
        "Clinical Linguistic Complexity Index (CLCI)",
}

METRIC_GROUPS = {

    "Reasoning Metrics": [

        "avg_entity_interaction_density",
        "avg_reasoning_path_complexity",
        "avg_context_expansion_ratio",
    ],

    "Linguistic Metrics": [

        "avg_sentence_complexity",
        "avg_lexical_diversity",
        "avg_token_length",
        "avg_content_word_ratio",
        "avg_clause_proxy_complexity",
    ],

    "Hybrid Complexity Metrics": [

        "avg_clinical_linguistic_complexity_index"
    ]
}


# =========================
# LOAD METRICS
# =========================

all_metrics = {}

for folder in os.listdir(RAW_DIR):

    path = os.path.join(RAW_DIR, folder)
    report_path = os.path.join(path, "report.json")

    if not os.path.exists(report_path):
        continue

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("summary", {})

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, {})[folder] = v

    except Exception as e:
        print(f"[ERROR] {folder}: {e}")


# =========================
# HELPERS
# =========================

def safe_mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    return np.mean(values) if values else 0.0


# =========================
# GENERIC PLOTS
# =========================

for metric_name, dataset_values in all_metrics.items():

    labels, values, colors = [], [], []

    for d in DATASET_ORDER:
        if d not in dataset_values:
            continue

        labels.append(d)
        values.append(dataset_values[d])

        colors.append(
            "steelblue"
            if d in EXISTING_BENCHMARKS
            else "darkorange"
        )

    if not labels:
        continue

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values, color=colors)

    plt.xticks(rotation=45, ha="right")
    plt.title(metric_name)
    plt.ylabel(metric_name)

    plt.legend(handles=[
        Patch(facecolor="steelblue", label="Existing Benchmarks"),
        Patch(facecolor="darkorange", label="HealthBench")
    ])

    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"{metric_name}.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[SAVED] {out}")


# =========================
# CONFERENCE PLOTS
# =========================

for metric_key, metric_title in KEY_METRICS.items():

    if metric_key not in all_metrics:
        print(f"[WARNING] Missing metric: {metric_key}")
        continue

    dataset_values = all_metrics[metric_key]

    existing_vals, healthbench_vals = [], []

    labels, values, colors = [], [], []

    for d in DATASET_ORDER:

        v = dataset_values.get(d, np.nan)

        if not isinstance(v, (int, float)) or np.isnan(v):
            continue

        labels.append(d)
        values.append(v)

        if d in EXISTING_BENCHMARKS:
            existing_vals.append(v)
            colors.append("steelblue")
        else:
            healthbench_vals.append(v)
            colors.append("darkorange")

    # -------------------------
    # per-dataset plot
    # -------------------------

    plt.figure(figsize=(16, 7))

    bars = plt.bar(
        labels,
        values,
        color=colors
    )

    plt.xticks(rotation=45, ha="right")
    plt.title(metric_title)

    for b in bars:

        h = b.get_height()

        plt.text(
            b.get_x() + b.get_width()/2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.legend(handles=[
        Patch(facecolor="steelblue", label="Existing Benchmarks"),
        Patch(facecolor="darkorange", label="HealthBench")
    ])

    plt.tight_layout()

    out = os.path.join(
        OUTPUT_DIR,
        f"conference_{metric_key}.png"
    )

    plt.savefig(
        out,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {out}")

    # -------------------------
    # grouped plot
    # -------------------------

    means = [
        safe_mean(existing_vals),
        safe_mean(healthbench_vals)
    ]

    stds = [
        np.std(existing_vals) if existing_vals else 0,
        np.std(healthbench_vals) if healthbench_vals else 0
    ]

    plt.figure(figsize=(6, 6))

    plt.bar(
        ["Existing Benchmarks", "HealthBench"],
        means,
        yerr=stds,
        capsize=8,
        color=["steelblue", "darkorange"]
    )

    plt.title(f"Average {metric_title}")

    plt.tight_layout()

    out = os.path.join(
        OUTPUT_DIR,
        f"grouped_{metric_key}.png"
    )

    plt.savefig(
        out,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {out}")


# =========================
# NORMALIZED COMPARISON
# =========================

metric_keys = list(KEY_METRICS.keys())

existing_means, healthbench_means = [], []

for m in metric_keys:

    d = all_metrics.get(m, {})

    existing = [
        v for k, v in d.items()
        if k in EXISTING_BENCHMARKS
    ]

    healthbench = [
        v for k, v in d.items()
        if k in HEALTHBENCH_BENCHMARK
    ]

    existing_means.append(
        safe_mean(existing)
    )

    healthbench_means.append(
        safe_mean(healthbench)
    )

global_max = max(
    existing_means +
    healthbench_means +
    [1e-8]
)

existing_norm = [
    v / global_max
    for v in existing_means
]

healthbench_norm = [
    v / global_max
    for v in healthbench_means
]

x = np.arange(len(metric_keys))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width/2,
    existing_norm,
    width,
    label="Existing Benchmarks",
    color="steelblue"
)

plt.bar(
    x + width/2,
    healthbench_norm,
    width,
    label="HealthBench",
    color="darkorange"
)

plt.xticks(
    x,
    [KEY_METRICS[m] for m in metric_keys],
    rotation=20
)

plt.title(
    "Normalized Benchmark Complexity Comparison"
)

plt.legend()

plt.tight_layout()

out = os.path.join(
    OUTPUT_DIR,
    "normalized_complexity_comparison.png"
)

plt.savefig(
    out,
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print(f"[SAVED] {out}")
print("\nAll metric plots generated successfully.")


# =========================
# CATEGORY-LEVEL COMPARISON
# =========================

for group_name, metric_list in METRIC_GROUPS.items():

    existing_scores = []
    healthbench_scores = []

    labels = []

    for metric_key in metric_list:

        if metric_key not in all_metrics:
            continue

        dataset_values = all_metrics[metric_key]

        existing_vals = [
            v for k, v in dataset_values.items()
            if k in EXISTING_BENCHMARKS
        ]

        healthbench_vals = [
            v for k, v in dataset_values.items()
            if k in HEALTHBENCH_BENCHMARK
        ]

        existing_scores.append(
            safe_mean(existing_vals)
        )

        healthbench_scores.append(
            safe_mean(healthbench_vals)
        )

        labels.append(
            KEY_METRICS[metric_key]
        )

    if not labels:
        continue

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width/2,
        existing_scores,
        width,
        label="Existing Benchmarks",
        color="steelblue"
    )

    plt.bar(
        x + width/2,
        healthbench_scores,
        width,
        label="HealthBench",
        color="darkorange"
    )

    plt.xticks(
        x,
        labels,
        rotation=20,
        ha="right"
    )

    plt.ylabel("Average Score")

    plt.title(
        f"{group_name}: Existing Benchmarks vs HealthBench",
        fontsize=14,
        fontweight="bold"
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{group_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {output_path}")