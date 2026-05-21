from collections import Counter
import math


# =========================
# existing metrics unchanged
# =========================

def extract_entity_types(G):
    return [
        attrs.get("entity_type")
        for _, attrs in G.nodes(data=True)
        if attrs.get("entity_type")
    ]


def compute_entity_type_diversity(G):

    entity_types = extract_entity_types(G)

    if not entity_types:
        return 0.0

    return len(set(entity_types)) / len(entity_types)


def compute_unique_entity_ratio(G):

    nodes = list(G.nodes())

    if not nodes:
        return 0.0

    return len(set(nodes)) / len(nodes)


def compute_relation_entropy(G):

    degrees = [d for _, d in G.degree()]

    if not degrees:
        return 0.0

    total = sum(degrees)

    if total == 0:
        return 0.0

    entropy = 0.0

    for d in degrees:
        p = d / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_entity_interaction_density(G):

    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return G.number_of_edges() / n


def compute_context_expansion_ratio(G):

    context = 0
    intent = 0

    for _, attrs in G.nodes(data=True):

        role = attrs.get("node_role")

        if role == "context":
            context += 1
        elif role == "intent":
            intent += 1

    if intent == 0:
        return float(context)

    return context / intent


# ==========================================================
# NEW: GRAMMAR / LEXICAL METRICS (replaces ontology)
# ==========================================================

def compute_sentence_complexity_score(G):
    """
    Extracted from graph-level grammar feature.
    """

    return G.graph.get("sentence_complexity", 0.0)


def compute_lexical_diversity(G):
    """
    Type-token ratio over entity tokens.
    """

    tokens = list(G.nodes())

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def compute_avg_token_length(G):
    """
    Proxy for lexical difficulty.
    """

    tokens = list(G.nodes())

    if not tokens:
        return 0.0

    return sum(len(t) for t in tokens) / len(tokens)


def compute_content_word_ratio(G):
    """
    Heuristic: long tokens likely content words.
    """

    tokens = list(G.nodes())

    if not tokens:
        return 0.0

    content = [t for t in tokens if len(t) > 5]

    return len(content) / len(tokens)


def compute_clause_proxy_complexity(G):
    """
    Proxy clause complexity:
    combines edges + sentence complexity.
    """

    n = G.number_of_nodes()
    if n == 0:
        return 0.0

    base = G.number_of_edges() / n
    sent = compute_sentence_complexity_score(G)

    return base * (1 + sent)