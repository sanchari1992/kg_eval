
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# DATA
# =========================================================

# metrics = [
#     "Entity Interaction Density",
#     "Reasoning Path Complexity",
#     "Context Expansion Ratio",
#     "Sentence Complexity",
#     "Lexical Diversity",
#     "Content Word Ratio",
#     "Clause Complexity Proxy",
#     "Clinical Linguistic Complexity Index"
# ]

# existing_benchmarks = [0.09, 0.3, 0.1, 2.9, 0.99, 0.73, 0.0018, 0.09]
# healthbench = [0.5, 12.8, 1.0, 2.8, 0.98, 0.73, 0.0021, 3.41]

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

# fig, axes = plt.subplots(2, 4, figsize=(18, 8))
# axes = axes.flatten()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

x = np.arange(2)
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
        width=0.55
    )

    # Rounded bar edges (works visually with edge styling)
    for bar in bars:
        bar.set_linewidth(0)
        bar.set_alpha(0.95)

    ax.set_title(metrics[i], fontsize=11, pad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=9)

    ax.tick_params(axis='y', labelsize=9)

    # Cleaner look
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

