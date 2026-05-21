import networkx as nx


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