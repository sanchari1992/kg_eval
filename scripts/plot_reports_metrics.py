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
    "medhalt",
    "bioasq",
    "medmcqa",
    "medrevqa",
    "medchangeqa",
    "covidqa",

    "healthbench",
    "biqa",
    "medaesqa",
    "medquad",
    "mediqa",
    "healthsearchqa",
    "medicationqa",
]

SCIENTIFIC_DATASETS = set(DATASET_ORDER[:8])


# =========================
# METRICS
# =========================

KEY_METRICS = {
    "avg_entity_interaction_density": "Entity Interaction Density",
    "avg_reasoning_path_complexity": "Reasoning Path Complexity",
    "avg_context_expansion_ratio": "Context Expansion Ratio",

    # grammar-based metrics
    "avg_sentence_complexity": "Sentence Complexity",
    "avg_lexical_diversity": "Lexical Diversity",
    "avg_avg_token_length": "Average Token Length",
    "avg_content_word_ratio": "Content Word Ratio",
    "avg_clause_proxy_complexity": "Clause Complexity Proxy",
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
        colors.append("steelblue" if d in SCIENTIFIC_DATASETS else "darkorange")

    if not labels:
        continue

    plt.figure(figsize=(14, 6))
    plt.bar(labels, values, color=colors)

    plt.xticks(rotation=45, ha="right")
    plt.title(metric_name)
    plt.ylabel(metric_name)

    plt.legend(handles=[
        Patch(facecolor="steelblue", label="Scientific"),
        Patch(facecolor="darkorange", label="Consumer")
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

    sci_vals, con_vals = [], []

    labels, values, colors = [], [], []

    for d in DATASET_ORDER:

        v = dataset_values.get(d, np.nan)
        if not isinstance(v, (int, float)) or np.isnan(v):
            continue

        labels.append(d)
        values.append(v)

        if d in SCIENTIFIC_DATASETS:
            sci_vals.append(v)
            colors.append("steelblue")
        else:
            con_vals.append(v)
            colors.append("darkorange")

    # -------------------------
    # per-dataset plot
    # -------------------------

    plt.figure(figsize=(16, 7))
    bars = plt.bar(labels, values, color=colors)

    plt.xticks(rotation=45, ha="right")
    plt.title(metric_title)

    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"conference_{metric_key}.png")
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out}")

    # -------------------------
    # grouped plot
    # -------------------------

    means = [
        safe_mean(sci_vals),
        safe_mean(con_vals)
    ]

    stds = [
        np.std(sci_vals) if sci_vals else 0,
        np.std(con_vals) if con_vals else 0
    ]

    plt.figure(figsize=(6, 6))
    plt.bar(
        ["Scientific", "Consumer"],
        means,
        yerr=stds,
        capsize=8,
        color=["steelblue", "darkorange"]
    )

    plt.title(f"Average {metric_title}")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"grouped_{metric_key}.png")
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out}")


# =========================
# NORMALIZED COMPARISON (FIXED & SAFE)
# =========================

metric_keys = list(KEY_METRICS.keys())

sci_means, con_means = [], []

for m in metric_keys:

    d = all_metrics.get(m, {})

    sci = [v for k, v in d.items() if k in SCIENTIFIC_DATASETS]
    con = [v for k, v in d.items() if k not in SCIENTIFIC_DATASETS]

    sci_means.append(safe_mean(sci))
    con_means.append(safe_mean(con))


global_max = max(sci_means + con_means + [1e-8])

sci_norm = [v / global_max for v in sci_means]
con_norm = [v / global_max for v in con_means]

x = np.arange(len(metric_keys))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(x - width/2, sci_norm, width, label="Scientific", color="steelblue")
plt.bar(x + width/2, con_norm, width, label="Consumer", color="darkorange")

plt.xticks(x, [KEY_METRICS[m] for m in metric_keys], rotation=20)
plt.title("Normalized Benchmark Complexity Comparison")

plt.legend()
plt.tight_layout()

out = os.path.join(OUTPUT_DIR, "normalized_complexity_comparison.png")
plt.savefig(out, dpi=400, bbox_inches="tight")
plt.close()

print(f"[SAVED] {out}")
print("\nAll metric plots generated successfully.")