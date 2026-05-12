import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(G, title="Knowledge Graph", max_nodes=50):
    """
    Visualize a graph (limited to avoid clutter).
    """

    plt.figure(figsize=(8, 6))

    # optionally reduce size for readability
    nodes = list(G.nodes())[:max_nodes]
    subgraph = G.subgraph(nodes)

    pos = nx.spring_layout(subgraph, seed=42)

    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_size=800,
        font_size=8
    )

    plt.title(title)
    plt.show()