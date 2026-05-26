# graph_metrics.py

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

    # =====================================================
    # NLP / LINGUISTIC
    # =====================================================

    metrics["sentence_complexity"] = (
        compute_sentence_complexity_score(G)
    )

    metrics["lexical_diversity"] = (
        compute_lexical_diversity(G)
    )

    metrics["avg_token_length"] = (
        compute_avg_token_length(G)
    )

    metrics["content_word_ratio"] = (
        compute_content_word_ratio(G)
    )

    metrics["clause_proxy_complexity"] = (
        compute_clause_proxy_complexity(G)
    )

    # =====================================================
    # COMBINED INDEX
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
# SERIALIZATION HELPERS
# =========================================================

def serialize_graph(G):

    nodes = []

    for node_id in G.nodes():

        node_data = dict(G.nodes[node_id])

        nodes.append({
            "id": node_id,
            **node_data
        })

    edges = []

    for u, v in G.edges():

        edges.append({
            "source": u,
            "target": v
        })

    return {
        "nodes": nodes,
        "edges": edges
    }


# =========================================================
# DATASET-LEVEL AGGREGATION
# =========================================================

def compute_dataset_metrics(graphs, questions=None):

    all_entries = []

    # =====================================================
    # PER-GRAPH PROCESSING
    # =====================================================

    for idx, G in enumerate(graphs):

        graph_metrics = compute_basic_metrics(G)

        question = (
            questions[idx]
            if questions is not None
            else None
        )

        graph_data = serialize_graph(G)

        entry = {

            "question": question,

            "graph": {
                "nodes": graph_data["nodes"],
                "edges": graph_data["edges"]
            },

            "metrics": graph_metrics
        }

        all_entries.append(entry)

    # =====================================================
    # EMPTY SAFETY
    # =====================================================

    if len(all_entries) == 0:
        return {}, []

    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    summary = {

        # =================================================
        # STRUCTURAL
        # =================================================

        "num_graphs":
            len(graphs),

        "avg_nodes":
            sum(
                m["metrics"]["num_nodes"]
                for m in all_entries
            ) / len(all_entries),

        "avg_edges":
            sum(
                m["metrics"]["num_edges"]
                for m in all_entries
            ) / len(all_entries),

        "avg_density":
            sum(
                m["metrics"]["density"]
                for m in all_entries
            ) / len(all_entries),

        "avg_degree":
            sum(
                m["metrics"]["avg_degree"]
                for m in all_entries
            ) / len(all_entries),

        "avg_clustering_coefficient":
            sum(
                m["metrics"]["clustering_coefficient"]
                for m in all_entries
            ) / len(all_entries),

        # =================================================
        # SEMANTIC
        # =================================================

        "avg_entity_type_diversity":
            sum(
                m["metrics"]["entity_type_diversity"]
                for m in all_entries
            ) / len(all_entries),

        "avg_unique_entity_ratio":
            sum(
                m["metrics"]["unique_entity_ratio"]
                for m in all_entries
            ) / len(all_entries),

        "avg_relation_entropy":
            sum(
                m["metrics"]["relation_entropy"]
                for m in all_entries
            ) / len(all_entries),

        "avg_entity_interaction_density":
            sum(
                m["metrics"]["entity_interaction_density"]
                for m in all_entries
            ) / len(all_entries),

        "avg_context_expansion_ratio":
            sum(
                m["metrics"]["context_expansion_ratio"]
                for m in all_entries
            ) / len(all_entries),

        # =================================================
        # REASONING
        # =================================================

        "avg_constraint_complexity":
            sum(
                m["metrics"]["constraint_complexity"]
                for m in all_entries
            ) / len(all_entries),

        "avg_graph_expansion_ratio":
            sum(
                m["metrics"]["graph_expansion_ratio"]
                for m in all_entries
            ) / len(all_entries),

        "avg_structural_reasoning_index":
            sum(
                m["metrics"]["structural_reasoning_index"]
                for m in all_entries
            ) / len(all_entries),

        "avg_shortest_path":
            sum(
                m["metrics"]["avg_shortest_path"]
                for m in all_entries
            ) / len(all_entries),

        "avg_branching_factor":
            sum(
                m["metrics"]["branching_factor"]
                for m in all_entries
            ) / len(all_entries),

        "avg_graph_centralization":
            sum(
                m["metrics"]["graph_centralization"]
                for m in all_entries
            ) / len(all_entries),

        "avg_reasoning_path_complexity":
            sum(
                m["metrics"]["reasoning_path_complexity"]
                for m in all_entries
            ) / len(all_entries),

        # =================================================
        # NLP / LINGUISTIC
        # =================================================

        "avg_sentence_complexity":
            sum(
                m["metrics"]["sentence_complexity"]
                for m in all_entries
            ) / len(all_entries),

        "avg_lexical_diversity":
            sum(
                m["metrics"]["lexical_diversity"]
                for m in all_entries
            ) / len(all_entries),

        "avg_avg_token_length":
            sum(
                m["metrics"]["avg_token_length"]
                for m in all_entries
            ) / len(all_entries),

        "avg_content_word_ratio":
            sum(
                m["metrics"]["content_word_ratio"]
                for m in all_entries
            ) / len(all_entries),

        "avg_clause_proxy_complexity":
            sum(
                m["metrics"]["clause_proxy_complexity"]
                for m in all_entries
            ) / len(all_entries),

        # =================================================
        # COMBINED INDEX
        # =================================================

        "avg_clinical_linguistic_complexity_index":
            sum(
                m["metrics"][
                    "clinical_linguistic_complexity_index"
                ]
                for m in all_entries
            ) / len(all_entries),
    }

    return summary, all_entries