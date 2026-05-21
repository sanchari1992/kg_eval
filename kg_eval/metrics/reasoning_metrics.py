import networkx as nx


def compute_constraint_complexity(G):
    """
    Approximate reasoning constraints.

    Uses:
    - node count
    - entity diversity
    """

    num_nodes = G.number_of_nodes()

    entity_types = set()

    for _, attrs in G.nodes(data=True):

        entity_type = attrs.get("entity_type")

        if entity_type:
            entity_types.add(entity_type)

    return num_nodes + len(entity_types)


def compute_graph_expansion_ratio(G):
    """
    Approximate implicit reasoning burden.

    GER > 1 implies additional reasoning needed.
    """

    base_size = G.number_of_nodes()

    if base_size == 0:
        return 0.0

    constraint_complexity = (
        compute_constraint_complexity(G)
    )

    expanded_size = (
        base_size
        + 0.5 * constraint_complexity
    )

    return expanded_size / base_size


def compute_structural_reasoning_index(G):
    """
    Combined reasoning difficulty metric.
    """

    cc = compute_constraint_complexity(G)

    ger = compute_graph_expansion_ratio(G)

    return cc * ger


def compute_avg_shortest_path(G):
    """
    Multi-hop reasoning approximation.
    """

    if (
        G.number_of_nodes() <= 1
        or not nx.is_connected(G)
    ):
        return 0.0

    return nx.average_shortest_path_length(G)


def compute_branching_factor(G):
    """
    Average branching per node.
    """

    num_nodes = G.number_of_nodes()

    if num_nodes == 0:
        return 0.0

    return G.number_of_edges() / num_nodes


def compute_graph_centralization(G):
    """
    Measures dependency on central hub nodes.
    """

    if G.number_of_nodes() == 0:
        return 0.0

    degrees = [d for _, d in G.degree()]

    max_degree = max(degrees)

    numerator = sum(
        max_degree - d
        for d in degrees
    )

    denominator = (
        (G.number_of_nodes() - 1)
        * (G.number_of_nodes() - 2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator