from collections import Counter
import math
import numpy as np

# ==========================================================
# BASIC ENTITY METRICS
# ==========================================================

def extract_entity_types(G):

    entity_types = []

    for _, attrs in G.nodes(data=True):

        entity_type = attrs.get("entity_type")

        if entity_type:
            entity_types.append(entity_type)

    return entity_types


def compute_entity_type_diversity(G):

    entity_types = extract_entity_types(G)

    if len(entity_types) == 0:
        return 0.0

    return len(set(entity_types)) / len(entity_types)


def compute_unique_entity_ratio(G):

    entities = list(G.nodes())

    if len(entities) == 0:
        return 0.0

    return len(set(entities)) / len(entities)


def compute_relation_entropy(G):

    degrees = [d for _, d in G.degree()]

    if len(degrees) == 0:
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


# ==========================================================
# STRUCTURAL SEMANTIC METRICS
# ==========================================================

def compute_entity_interaction_density(G):

    num_nodes = G.number_of_nodes()

    if num_nodes == 0:
        return 0.0

    return G.number_of_edges() / num_nodes


def compute_context_expansion_ratio(G):

    context_nodes = 0
    intent_nodes = 0

    for _, attrs in G.nodes(data=True):

        role = attrs.get("node_role")

        if role == "context":
            context_nodes += 1

        elif role == "intent":
            intent_nodes += 1

    if intent_nodes == 0:
        return float(context_nodes)

    return context_nodes / intent_nodes


# ==========================================================
# NLP / GRAMMAR METRICS
# ==========================================================

def compute_sentence_complexity_score(G):

    depths = []

    for _, attrs in G.nodes(data=True):

        depth = attrs.get("dependency_depth")

        if depth is not None:
            depths.append(depth)

    if not depths:
        return 0.0

    return np.mean(depths)


def compute_lexical_diversity(G):

    tokens = []

    for _, attrs in G.nodes(data=True):

        token = attrs.get("token_text")

        if token:
            tokens.append(token.lower())

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def compute_avg_token_length(G):

    lengths = []

    for _, attrs in G.nodes(data=True):

        token = attrs.get("token_text")

        if token:
            lengths.append(len(token))

    if not lengths:
        return 0.0

    return np.mean(lengths)


def compute_content_word_ratio(G):

    content_count = 0
    total = 0

    CONTENT_POS = {
        "NOUN",
        "VERB",
        "ADJ",
        "ADV",
        "PROPN"
    }

    for _, attrs in G.nodes(data=True):

        pos = attrs.get("pos")

        if pos:
            total += 1

            if pos in CONTENT_POS:
                content_count += 1

    if total == 0:
        return 0.0

    return content_count / total


def compute_clause_proxy_complexity(G):

    clause_markers = 0

    CLAUSE_DEPS = {
        "advcl",
        "ccomp",
        "xcomp",
        "relcl",
        "acl"
    }

    for _, attrs in G.nodes(data=True):

        dep = attrs.get("dep")

        if dep in CLAUSE_DEPS:
            clause_markers += 1

    num_nodes = G.number_of_nodes()

    if num_nodes == 0:
        return 0.0

    return clause_markers / num_nodes


# ==========================================================
# NOVEL PAPER METRIC
# ==========================================================

def compute_clinical_linguistic_complexity_index(
    sentence_complexity,
    clause_complexity,
    entity_density,
    reasoning_complexity,
    content_word_ratio
):

    return (
        (
            sentence_complexity
            * clause_complexity
            * entity_density
        )
        +
        (
            reasoning_complexity
            * content_word_ratio
        )
    ) / 2


# # ==========================================================
# # DERIVED LINGUISTIC COMPLEXITY
# # ==========================================================

# def compute_derived_linguistic_complexity(G):
#     """
#     LC = average of normalized:
#         - sentence_complexity
#         - clause_proxy_complexity
#         - content_word_ratio
#         - lexical_diversity
#     """

#     sc = compute_sentence_complexity_score(G)
#     cd = compute_clause_proxy_complexity(G)
#     cwr = compute_content_word_ratio(G)
#     ttr = compute_lexical_diversity(G)

#     # normalization (bounded-safe transforms)
#     sc_n = sc / (sc + 1.0) if sc > 0 else 0.0
#     cd_n = cd
#     cwr_n = cwr
#     ttr_n = ttr

#     lc = 0.25 * (sc_n + cd_n + cwr_n + ttr_n)

#     return float(lc)

# def compute_overall_complexity_index(G, alpha=0.5):
#     """
#     TC = alpha * SC + (1 - alpha) * LC
#     """

#     sc = compute_derived_structural_complexity(G)
#     lc = compute_derived_linguistic_complexity(G)

#     tc = alpha * sc + (1.0 - alpha) * lc

#     return float(tc)