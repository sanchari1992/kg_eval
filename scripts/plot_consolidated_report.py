import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# DATA
# =========================================================

metrics = [
    "Entity Interaction Density",
    "Reasoning Path Complexity",
    "Context Expansion Ratio",
    "Clinical Linguistic Complexity Index"
]

existing_benchmarks = [0.09, 0.3, 0.1, 0.09]
healthbench = [0.5, 12.8, 1.0, 3.41]


# =========================================================
# SETUP
# =========================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

x = np.array([0, 0.6])
labels = ["Existing Benchmarks", "HealthBench"]

# Two distinct colors
colors = ["#4C72B0", "#DD8452"]


# =========================================================
# PLOT EACH METRIC
# =========================================================

for i, ax in enumerate(axes):

    values = [existing_benchmarks[i], healthbench[i]]

    bars = ax.bar(
        x,
        values,
        color=colors,
        width=0.38
    )

    # Bar styling
    for bar in bars:
        bar.set_linewidth(0)
        bar.set_alpha(0.95)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(values) * 0.03),
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=10.5,
            fontweight='bold'
        )

    # Extra vertical space for labels
    ax.set_ylim(0, max(values) * 1.22)

    ax.set_title(
        metrics[i],
        fontsize=12.5,
        pad=10
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=15,
        fontsize=10.5
    )

    ax.tick_params(axis='y', labelsize=10.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#555555")


# =========================================================
# GLOBAL TITLE + LAYOUT
# =========================================================

fig.suptitle(
    "Structural Metric Comparison: Existing Benchmarks vs HealthBench",
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()