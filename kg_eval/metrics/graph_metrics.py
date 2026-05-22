import networkx as nx

from kg_eval.metrics.semantic_metrics import (
    compute_entity_type_diversity,
    compute_unique_entity_ratio,
    compute_relation_entropy,
    compute_entity_interaction_density,
    compute_context_expansion_ratio,
    compute_sentence_complexity_score,
    compute_lexical_diversity,
    compute_avg_token_length,
    compute_content_word_ratio,
    compute_clause_proxy_complexity,
    compute_clinical_linguistic_complexity_index
)

from kg_eval.metrics.reasoning_metrics import (
    compute_constraint_complexity,
    compute_graph_expansion_ratio,
    compute_structural_reasoning_index,
    compute_avg_shortest_path,
    compute_branching_factor,
    compute_graph_centralization,
    compute_reasoning_path_complexity
)


# =========================================================
# BASIC GRAPH METRICS
# =========================================================

def compute_basic_metrics(G):

    metrics = {}

    # =====================================================
    # STRUCTURAL METRICS
    # =====================================================

    metrics["num_nodes"] = G.number_of_nodes()

    metrics["num_edges"] = G.number_of_edges()

    metrics["density"] = (
        nx.density(G)
        if metrics["num_nodes"] > 1 else 0.0
    )

    degrees = [d for _, d in G.degree()]

    metrics["avg_degree"] = (
        sum(degrees) / len(degrees)
        if degrees else 0.0
    )

    metrics["num_connected_components"] = (
        nx.number_connected_components(G)
    )

    metrics["clustering_coefficient"] = (
        nx.average_clustering(G)
        if metrics["num_nodes"] > 1 else 0.0
    )

    # =====================================================
    # SEMANTIC METRICS
    # =====================================================

    metrics["entity_type_diversity"] = (
        compute_entity_type_diversity(G)
    )

    metrics["unique_entity_ratio"] = (
        compute_unique_entity_ratio(G)
    )

    metrics["relation_entropy"] = (
        compute_relation_entropy(G)
    )

    metrics["entity_interaction_density"] = (
        compute_entity_interaction_density(G)
    )

    metrics["context_expansion_ratio"] = (
        compute_context_expansion_ratio(G)
    )

    # =====================================================
    # REASONING METRICS
    # =====================================================

    metrics["constraint_complexity"] = (
        compute_constraint_complexity(G)
    )

    metrics["graph_expansion_ratio"] = (
        compute_graph_expansion_ratio(G)
    )

    metrics["structural_reasoning_index"] = (
        compute_structural_reasoning_index(G)
    )

    metrics["avg_shortest_path"] = (
        compute_avg_shortest_path(G)
    )

    metrics["branching_factor"] = (
        compute_branching_factor(G)
    )

    metrics["graph_centralization"] = (
        compute_graph_centralization(G)
    )

    metrics["reasoning_path_complexity"] = (
        compute_reasoning_path_complexity(G)
    )

    # =========================
    # LINGUISTIC / NLP
    # =========================

    metrics["sentence_complexity"] = compute_sentence_complexity_score(G)
    metrics["lexical_diversity"] = compute_lexical_diversity(G)
    metrics["avg_token_length"] = compute_avg_token_length(G)
    metrics["content_word_ratio"] = compute_content_word_ratio(G)
    metrics["clause_proxy_complexity"] = compute_clause_proxy_complexity(G)

    # =====================================================
    # COMBINED CLINICAL LINGUISTIC COMPLEXITY INDEX
    # =====================================================

    metrics["clinical_linguistic_complexity_index"] = (
        compute_clinical_linguistic_complexity_index(
            metrics["sentence_complexity"],
            metrics["clause_proxy_complexity"],
            metrics["entity_interaction_density"],
            metrics["reasoning_path_complexity"],
            metrics["content_word_ratio"]
        )
    )

    return metrics


# =========================================================
# DATASET-LEVEL AGGREGATION
# =========================================================

def compute_dataset_metrics(graphs):

    all_metrics = []

    for G in graphs:

        graph_metrics = compute_basic_metrics(G)

        all_metrics.append(graph_metrics)

    if len(all_metrics) == 0:
        return {}, []

    summary = {

        # =================================================
        # STRUCTURAL
        # =================================================

        "num_graphs":
            len(graphs),

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

        # =================================================
        # SEMANTIC
        # =================================================

        "avg_entity_type_diversity":
            sum(m["entity_type_diversity"] for m in all_metrics)
            / len(all_metrics),

        "avg_unique_entity_ratio":
            sum(m["unique_entity_ratio"] for m in all_metrics)
            / len(all_metrics),

        "avg_relation_entropy":
            sum(m["relation_entropy"] for m in all_metrics)
            / len(all_metrics),

        "avg_entity_interaction_density":
            sum(m["entity_interaction_density"] for m in all_metrics)
            / len(all_metrics),

        "avg_context_expansion_ratio":
            sum(m["context_expansion_ratio"] for m in all_metrics)
            / len(all_metrics),

        # =================================================
        # REASONING
        # =================================================

        "avg_constraint_complexity":
            sum(m["constraint_complexity"] for m in all_metrics)
            / len(all_metrics),

        "avg_graph_expansion_ratio":
            sum(m["graph_expansion_ratio"] for m in all_metrics)
            / len(all_metrics),

        "avg_structural_reasoning_index":
            sum(m["structural_reasoning_index"] for m in all_metrics)
            / len(all_metrics),

        "avg_shortest_path":
            sum(m["avg_shortest_path"] for m in all_metrics)
            / len(all_metrics),

        "avg_branching_factor":
            sum(m["branching_factor"] for m in all_metrics)
            / len(all_metrics),

        "avg_graph_centralization":
            sum(m["graph_centralization"] for m in all_metrics)
            / len(all_metrics),

        "avg_reasoning_path_complexity":
            sum(m["reasoning_path_complexity"] for m in all_metrics)
            / len(all_metrics),

        # =================================================
        # NLP / LINGUISTIC
        # =================================================

        "avg_sentence_complexity":
            sum(m["sentence_complexity"] for m in all_metrics)
            / len(all_metrics),

        "avg_lexical_diversity":
            sum(m["lexical_diversity"] for m in all_metrics)
            / len(all_metrics),

        "avg_avg_token_length":
            sum(m["avg_token_length"] for m in all_metrics)
            / len(all_metrics),

        "avg_content_word_ratio":
            sum(m["content_word_ratio"] for m in all_metrics)
            / len(all_metrics),

        "avg_clause_proxy_complexity":
            sum(m["clause_proxy_complexity"] for m in all_metrics)
            / len(all_metrics),

        # =================================================
        # COMBINED INDEX
        # =================================================

        "avg_clinical_linguistic_complexity_index":
            sum(
                m["clinical_linguistic_complexity_index"]
                for m in all_metrics
            ) / len(all_metrics),
    }

    return summary, all_metrics