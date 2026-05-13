import networkx as nx


def compute_basic_metrics(G: nx.Graph):
    """
    Compute structural metrics for a single graph.
    """

    metrics = {}

    # --- size ---
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()

    # --- density ---
    metrics["density"] = nx.density(G) if metrics["num_nodes"] > 1 else 0.0

    # --- average degree ---
    degrees = [d for _, d in G.degree()]
    metrics["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0.0

    # --- connected components ---
    metrics["num_connected_components"] = nx.number_connected_components(G)

    # --- diameter (only if connected) ---
    if nx.is_connected(G) and metrics["num_nodes"] > 1:
        metrics["diameter"] = nx.diameter(G)
    else:
        metrics["diameter"] = None

    return metrics


def compute_dataset_metrics(graphs):
    """
    Aggregate metrics across multiple graphs.
    """

    all_metrics = []

    for G in graphs:
        all_metrics.append(compute_basic_metrics(G))

    # aggregate
    if len(all_metrics) == 0:
        return {}, []

    summary = {
        "num_graphs": len(graphs),
        "avg_nodes": sum(m["num_nodes"] for m in all_metrics) / len(all_metrics),
        "avg_edges": sum(m["num_edges"] for m in all_metrics) / len(all_metrics),
        "avg_density": sum(m["density"] for m in all_metrics) / len(all_metrics),
        "avg_degree": sum(m["avg_degree"] for m in all_metrics) / len(all_metrics),
    }

    return summary, all_metrics