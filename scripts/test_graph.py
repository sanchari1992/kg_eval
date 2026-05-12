from kg_eval.pipeline.process_dataset import load_dataset, extract_graphs
from kg_eval.metrics.graph_metrics import compute_dataset_metrics
from kg_eval.reports.report_writer import save_metrics_report


def main():
    file_path = "data/raw/medhalt/train.csv"
    question_col = "question"

    df = load_dataset(file_path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns)

    graphs = extract_graphs(df, question_col)

    print("\nTotal graphs:", len(graphs))

    summary, per_graph_metrics = compute_dataset_metrics(graphs)

    print("\n=== DATASET SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== SAMPLE GRAPH METRICS (first 3 graphs) ===")
    for i, m in enumerate(per_graph_metrics[:3]):
        print(f"\nGraph {i+1}:")
        for k, v in m.items():
            print(f"  {k}: {v}")
    save_metrics_report(
        summary,
        per_graph_metrics,
        "data/raw/medhalt/report.json"
    )


if __name__ == "__main__":
    main()