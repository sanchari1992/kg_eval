import networkx as nx
import re


def tokenize(text: str):
    """
    Very simple tokenizer for MVP.
    Later you will replace this with biomedical entity extraction (UMLS, scispacy).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def build_question_graph(question: str):
    """
    Build a simple co-occurrence graph from a question.
    Nodes = tokens
    Edges = sequential token relationships
    """
    tokens = tokenize(question)

    G = nx.Graph()

    # add nodes
    for t in tokens:
        G.add_node(t)

    # add edges (simple adjacency)
    for i in range(len(tokens) - 1):
        G.add_edge(tokens[i], tokens[i + 1])

    return G