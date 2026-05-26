import pandas as pd


# =========================================================
# PATHS
# =========================================================

INPUT_FILE = "data/llm_outputs/bioasq_evaluation_results.csv"

OUTPUT_FILE = "data/llm_outputs/bioasq_hall_metrics.csv"


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("Loaded rows:", len(df))


# =========================================================
# CLEAN (just in case)
# =========================================================

df["exact_match"] = df["exact_match"].fillna(0)


# =========================================================
# GROUPED ACCURACY METRICS
# =========================================================

summary = df.groupby(
    ["model", "metric", "extreme_type"]
).agg(

    accuracy=("exact_match", "mean"),
    total=("exact_match", "count")

).reset_index()


# =========================================================
# OPTIONAL: pivot-style view (useful for papers)
# =========================================================

pivot = summary.pivot_table(

    index=["metric", "extreme_type"],
    columns="model",
    values="accuracy"

).reset_index()


# =========================================================
# SAVE OUTPUTS
# =========================================================

summary.to_csv(OUTPUT_FILE, index=False)

pivot.to_csv(
    "data/llm_outputs/bioasq_hall_metrics_pivot.csv",
    index=False
)


# =========================================================
# PRINT
# =========================================================

print("\n=== SUMMARY (LONG FORMAT) ===\n")
print(summary)

print("\nSaved:")
print(OUTPUT_FILE)
print("data/llm_outputs/bioasq_hall_metrics_pivot.csv")