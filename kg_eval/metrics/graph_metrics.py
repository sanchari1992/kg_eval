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
    compute_clause_proxy_complexity
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


def compute_basic_metrics(G):

    metrics = {}

    # -------------------------
    # structure
    # -------------------------

    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()

    metrics["density"] = (
        nx.density(G) if metrics["num_nodes"] > 1 else 0.0
    )

    degrees = [d for _, d in G.degree()]
    metrics["avg_degree"] = sum(degrees)/len(degrees) if degrees else 0.0

    metrics["num_connected_components"] = nx.number_connected_components(G)

    metrics["clustering_coefficient"] = (
        nx.average_clustering(G) if metrics["num_nodes"] > 1 else 0.0
    )

    # -------------------------
    # semantic
    # -------------------------

    metrics["entity_type_diversity"] = compute_entity_type_diversity(G)
    metrics["unique_entity_ratio"] = compute_unique_entity_ratio(G)
    metrics["relation_entropy"] = compute_relation_entropy(G)

    # -------------------------
    # reasoning
    # -------------------------

    metrics["constraint_complexity"] = compute_constraint_complexity(G)
    metrics["graph_expansion_ratio"] = compute_graph_expansion_ratio(G)
    metrics["structural_reasoning_index"] = compute_structural_reasoning_index(G)
    metrics["avg_shortest_path"] = compute_avg_shortest_path(G)
    metrics["branching_factor"] = compute_branching_factor(G)
    metrics["graph_centralization"] = compute_graph_centralization(G)

    metrics["entity_interaction_density"] = compute_entity_interaction_density(G)
    metrics["context_expansion_ratio"] = compute_context_expansion_ratio(G)

    # -------------------------
    # NEW grammar-based metrics
    # -------------------------

    metrics["sentence_complexity"] = compute_sentence_complexity_score(G)
    metrics["lexical_diversity"] = compute_lexical_diversity(G)
    metrics["avg_token_length"] = compute_avg_token_length(G)
    metrics["content_word_ratio"] = compute_content_word_ratio(G)
    metrics["clause_proxy_complexity"] = compute_clause_proxy_complexity(G)

    metrics["reasoning_path_complexity"] = compute_reasoning_path_complexity(G)

    return metrics