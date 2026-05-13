import networkx as nx

from kg_eval.metrics.semantic_metrics import (
    compute_entity_type_diversity,
    compute_unique_entity_ratio,
    compute_relation_entropy
)


def compute_basic_metrics(G: nx.Graph):
    """
    Compute structural + semantic metrics for a single graph.
    """

    metrics = {}

    # -------------------------
    # structural metrics
    # -------------------------

    metrics["num_nodes"] = G.number_of_nodes()

    metrics["num_edges"] = G.number_of_edges()

    metrics["density"] = (
        nx.density(G)
        if metrics["num_nodes"] > 1
        else 0.0
    )

    # average degree
    degrees = [d for _, d in G.degree()]

    metrics["avg_degree"] = (
        sum(degrees) / len(degrees)
        if degrees
        else 0.0
    )

    # connected components
    metrics["num_connected_components"] = (
        nx.number_connected_components(G)
    )

    # diameter
    if metrics["num_nodes"] > 1 and nx.is_connected(G):

        metrics["diameter"] = nx.diameter(G)

    else:
        metrics["diameter"] = None

    # clustering coefficient
    metrics["clustering_coefficient"] = (
        nx.average_clustering(G)
        if metrics["num_nodes"] > 1
        else 0.0
    )

    # -------------------------
    # semantic metrics
    # -------------------------

    metrics["entity_type_diversity"] = (
        compute_entity_type_diversity(G)
    )

    metrics["unique_entity_ratio"] = (
        compute_unique_entity_ratio(G)
    )

    metrics["relation_entropy"] = (
        compute_relation_entropy(G)
    )

    return metrics


def compute_dataset_metrics(graphs):
    """
    Aggregate metrics across multiple graphs.
    """

    all_metrics = []

    for G in graphs:

        graph_metrics = compute_basic_metrics(G)

        all_metrics.append(graph_metrics)

    if len(all_metrics) == 0:
        return {}, []

    summary = {

        "num_graphs": len(graphs),

        "avg_nodes":
            sum(m["num_nodes"] for m in all_metrics)
            / len(all_metrics),

        "avg_edges":
            sum(m["num_edges"] for m in all_metrics)
            / len(all_metrics),

        "avg_density":
            sum(m["density"] for m in all_metrics)
            / len(all_metrics),

        "avg_degree":
            sum(m["avg_degree"] for m in all_metrics)
            / len(all_metrics),

        "avg_clustering_coefficient":
            sum(m["clustering_coefficient"] for m in all_metrics)
            / len(all_metrics),

        "avg_entity_type_diversity":
            sum(m["entity_type_diversity"] for m in all_metrics)
            / len(all_metrics),

        "avg_unique_entity_ratio":
            sum(m["unique_entity_ratio"] for m in all_metrics)
            / len(all_metrics),

        "avg_relation_entropy":
            sum(m["relation_entropy"] for m in all_metrics)
            / len(all_metrics),
    }

    return summary, all_metrics