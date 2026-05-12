import json


def save_metrics_report(summary, per_graph_metrics, output_path: str):
    """
    Save metrics report as JSON.
    """

    report = {
        "summary": summary,
        "per_graph_metrics": per_graph_metrics
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {output_path}")