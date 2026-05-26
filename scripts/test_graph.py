import argparse
import time
from pathlib import Path

from kg_eval.pipeline.process_dataset import (
    load_dataset,
    extract_graphs
)

from kg_eval.metrics.graph_metrics import (
    compute_dataset_metrics
)

from kg_eval.reports.report_writer import (
    save_metrics_report
)

from kg_eval.visualization.graph_viz import (
    draw_graph
)


def main():

    # =====================================================
    # START TIMER
    # =====================================================

    start_time = time.time()

    # =====================================================
    # ARGUMENT PARSER
    # =====================================================

    parser = argparse.ArgumentParser(
        description="Run KG evaluation pipeline"
    )

    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to CSV or parquet dataset"
    )

    parser.add_argument(
        "--question_col",
        type=str,
        required=True,
        help="Column containing question text"
    )

    parser.add_argument(
        "--report_path",
        type=str,
        default="report.json",
        help="Output JSON report path"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize first graph"
    )

    args = parser.parse_args()

    # =====================================================
    # LOAD DATASET
    # =====================================================

    df = load_dataset(args.file_path)

    print("\nDataset loaded")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # =====================================================
    # EXTRACT QUESTIONS
    # =====================================================

    questions = (
        df[args.question_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"\nLoaded {len(questions)} questions")

    # =====================================================
    # BUILD GRAPHS
    # =====================================================

    graphs = extract_graphs(
        df,
        args.question_col
    )

    print(f"\nGenerated {len(graphs)} graphs")

    # =====================================================
    # OPTIONAL VISUALIZATION
    # =====================================================

    if args.visualize and graphs:

        draw_graph(
            graphs[0],
            title="Sample Knowledge Graph"
        )

    # =====================================================
    # COMPUTE METRICS
    # =====================================================

    summary, per_graph_metrics = compute_dataset_metrics(
        graphs,
        questions
    )

    # =====================================================
    # PRINT DATASET SUMMARY
    # =====================================================

    print("\n=== DATASET SUMMARY ===")

    for k, v in summary.items():
        print(f"{k}: {v}")

    # =====================================================
    # SAMPLE GRAPH PREVIEW
    # =====================================================

    print("\n=== SAMPLE GRAPH METRICS ===")

    for i, entry in enumerate(per_graph_metrics[:3]):

        print(f"\n=================================================")
        print(f"Graph {i + 1}")
        print(f"=================================================")

        # -------------------------------------------------
        # QUESTION
        # -------------------------------------------------

        print("\nQuestion:")
        print(entry["question"])

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        print("\nMetrics:")

        for k, v in entry["metrics"].items():
            print(f"  {k}: {v}")

        # -------------------------------------------------
        # GRAPH NODES
        # -------------------------------------------------

        print("\nNodes:")

        for node in entry["graph"]["nodes"]:
            print(node)

        # -------------------------------------------------
        # GRAPH EDGES
        # -------------------------------------------------

        print("\nEdges:")

        for edge in entry["graph"]["edges"]:
            print(edge)

    # =====================================================
    # SAVE REPORT
    # =====================================================

    report_path = Path(args.report_path)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_metrics_report(
        summary,
        per_graph_metrics,
        str(report_path)
    )

    print(f"\nReport saved to: {report_path}")

    # =====================================================
    # END TIMER
    # =====================================================

    end_time = time.time()

    total_time = end_time - start_time

    print(f"\nTotal runtime: {total_time:.2f} seconds")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()