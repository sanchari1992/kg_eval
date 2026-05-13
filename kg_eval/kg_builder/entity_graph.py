import networkx as nx
import spacy


# load NLP model once
nlp = spacy.load("en_core_web_sm")


def build_entity_graph(question: str):
    """
    Build a simple entity-based graph using spaCy NER.

    Nodes = named entities
    Edges = entity co-occurrence in sequence
    """

    doc = nlp(question)

    entities = []

    for ent in doc.ents:
        entity_text = ent.text.strip().lower()

        if entity_text:
            entities.append(entity_text)

    G = nx.Graph()

    # add entity nodes
    for e in entities:
        G.add_node(e)

    # connect neighboring entities
    for i in range(len(entities) - 1):
        G.add_edge(entities[i], entities[i + 1])

    return G