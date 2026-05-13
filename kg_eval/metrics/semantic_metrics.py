from collections import Counter
import math


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

    unique_types = len(set(entity_types))

    return unique_types / len(entity_types)


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


def compute_top_entity_types(G, top_k=5):

    entity_types = extract_entity_types(G)

    counter = Counter(entity_types)

    return counter.most_common(top_k)