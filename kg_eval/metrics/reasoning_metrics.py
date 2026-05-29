import networkx as nx
import numpy as np

from kg_eval.metrics.semantic_metrics import (
    compute_relation_entropy,
    compute_sentence_complexity_score,
    compute_clause_proxy_complexity,
    compute_content_word_ratio,
    compute_lexical_diversity
)


def compute_constraint_complexity(G):
    num_nodes = G.number_of_nodes()

    entity_types = {
        attrs.get("entity_type")
        for _, attrs in G.nodes(data=True)
        if attrs.get("entity_type")
    }

    return num_nodes + len(entity_types)


def compute_graph_expansion_ratio(G):
    base = G.number_of_nodes()
    if base == 0:
        return 0.0

    cc = compute_constraint_complexity(G)
    return (base + 0.5 * cc) / base


def compute_structural_reasoning_index(G):
    return (
        compute_constraint_complexity(G)
        * compute_graph_expansion_ratio(G)
    )


def compute_avg_shortest_path(G):
    if G.number_of_nodes() <= 1 or not nx.is_connected(G):
        return 0.0
    return nx.average_shortest_path_length(G)


def compute_branching_factor(G):
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return G.number_of_edges() / n


def compute_graph_centralization(G):
    if G.number_of_nodes() == 0:
        return 0.0

    degrees = [d for _, d in G.degree()]
    max_d = max(degrees)

    num = sum(max_d - d for d in degrees)
    den = (G.number_of_nodes() - 1) * (G.number_of_nodes() - 2)

    return num / den if den else 0.0


def compute_reasoning_path_complexity(G):
    if G.number_of_nodes() <= 1 or not nx.is_connected(G):
        return 0.0

    return (
        nx.average_shortest_path_length(G)
        * nx.diameter(G)
    )


# ==========================================================
# NORMALIZATION HELPERS (SAFE VERSION)
# ==========================================================

def _safe_norm(x, max_val):
    if max_val == 0:
        return 0.0
    return x / max_val


def compute_derived_structural_complexity(G):
    """
    SC = weighted normalized combination of:
    - avg shortest path
    - branching factor
    - relation entropy
    - graph centralization
    """

    sp = compute_avg_shortest_path(G)
    bf = compute_branching_factor(G)
    gc = compute_graph_centralization(G)
    re = compute_relation_entropy(G)

    sp_n = sp / (sp + 1.0) if sp > 0 else 0.0
    bf_n = bf / (bf + 1.0) if bf > 0 else 0.0
    re_n = np.log1p(re)
    gc_n = gc

    sc = 0.25 * (sp_n + bf_n + re_n + gc_n)

    return float(sc)


def compute_derived_linguistic_complexity(G):
    """
    LC = average of normalized linguistic signals
    """

    sc = compute_sentence_complexity_score(G)
    cd = compute_clause_proxy_complexity(G)
    cwr = compute_content_word_ratio(G)
    ttr = compute_lexical_diversity(G)

    sc_n = sc / (sc + 1.0) if sc > 0 else 0.0

    lc = 0.25 * (sc_n + cd + cwr + ttr)

    return float(lc)


def compute_overall_complexity_index(G, alpha=0.5):
    """
    TC = alpha * SC + (1 - alpha) * LC
    """

    sc = compute_derived_structural_complexity(G)
    lc = compute_derived_linguistic_complexity(G)

    return float(alpha * sc + (1.0 - alpha) * lc)