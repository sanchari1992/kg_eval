import json
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

REPORT_PATH = "data/raw/medchangeqa/report.json"

TOP_N = 20

OUTPUT_CSV = "data/prompts_hallucination/medchangeqa_metric_extremes.csv"

# ---------------------------------------------------------
# filter trivial graphs
# ---------------------------------------------------------

MIN_NODES = 2
MIN_EDGES = 1

# ---------------------------------------------------------
# metric rounding for distinctness
# ---------------------------------------------------------

ROUND_DIGITS = 3


# =========================================================
# METRICS TO ANALYZE
# =========================================================

TARGET_METRICS = {
    "entity_interaction_density": "EID",
    "reasoning_path_complexity": "RPC",
    "context_expansion_ratio": "CER",
    "clinical_linguistic_complexity_index": "CLCI"
}


# =========================================================
# LOAD REPORT
# =========================================================

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

entries = report["per_graph_metrics"]

print(f"\nLoaded {len(entries)} graph entries")


# =========================================================
# BUILD DATAFRAME
# =========================================================

rows = []

for idx, entry in enumerate(entries):

    metrics = entry["metrics"]

    row = {

        "graph_id": idx,

        "question": entry.get("question", ""),

        "num_nodes": metrics.get("num_nodes", 0),

        "num_edges": metrics.get("num_edges", 0)
    }

    # -----------------------------------------------------
    # add target metrics
    # -----------------------------------------------------

    for metric_name in TARGET_METRICS.keys():

        row[metric_name] = metrics.get(metric_name, 0.0)

    rows.append(row)

df = pd.DataFrame(rows)

print("\nOriginal dataframe shape:")
print(df.shape)


# =========================================================
# ROUND METRICS
# =========================================================

for metric_name in TARGET_METRICS.keys():

    df[metric_name] = (
        df[metric_name]
        .round(ROUND_DIGITS)
    )


# =========================================================
# FILTER TRIVIAL GRAPHS
# =========================================================

filtered_df = df[
    (df["num_nodes"] >= MIN_NODES) &
    (df["num_edges"] >= MIN_EDGES)
].copy()

print("\nFiltered dataframe shape:")
print(filtered_df.shape)

print(
    f"\nRetained "
    f"{len(filtered_df)} / {len(df)} graphs "
    f"after filtering"
)


# =========================================================
# FIND DISTINCT EXTREMES
# =========================================================

results = []

for metric_name, short_name in TARGET_METRICS.items():

    print("\n" + "=" * 80)
    print(f"{short_name} ({metric_name})")
    print("=" * 80)

    # =====================================================
    # HIGHEST DISTINCT
    # =====================================================

    highest = (
        filtered_df

        .sort_values(
            by=metric_name,
            ascending=False
        )

        # remove duplicate prompts
        .drop_duplicates(
            subset=["question"]
        )

        # remove repeated metric values
        .drop_duplicates(
            subset=[metric_name]
        )

        .head(TOP_N)
    )

    print(f"\nTOP {TOP_N} DISTINCT HIGHEST\n")

    for rank, (_, row) in enumerate(
        highest.iterrows(),
        start=1
    ):

        print(f"{rank}. Value: {row[metric_name]}")
        print(f"Question: {row['question']}")
        print(
            f"Nodes: {row['num_nodes']} | "
            f"Edges: {row['num_edges']}"
        )
        print()

        results.append({
            "metric": short_name,
            "extreme_type": "highest",
            "rank": rank,
            "value": row[metric_name],
            "question": row["question"],
            "num_nodes": row["num_nodes"],
            "num_edges": row["num_edges"]
        })

    # =====================================================
    # LOWEST DISTINCT
    # =====================================================

    lowest = (
        filtered_df

        .sort_values(
            by=metric_name,
            ascending=True
        )

        .drop_duplicates(
            subset=["question"]
        )

        .drop_duplicates(
            subset=[metric_name]
        )

        .head(TOP_N)
    )

    print(f"\nTOP {TOP_N} DISTINCT LOWEST\n")

    for rank, (_, row) in enumerate(
        lowest.iterrows(),
        start=1
    ):

        print(f"{rank}. Value: {row[metric_name]}")
        print(f"Question: {row['question']}")
        print(
            f"Nodes: {row['num_nodes']} | "
            f"Edges: {row['num_edges']}"
        )
        print()

        results.append({
            "metric": short_name,
            "extreme_type": "lowest",
            "rank": rank,
            "value": row[metric_name],
            "question": row["question"],
            "num_nodes": row["num_nodes"],
            "num_edges": row["num_edges"]
        })


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"\nSaved results to: {OUTPUT_CSV}")