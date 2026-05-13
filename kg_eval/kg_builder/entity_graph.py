import networkx as nx
import spacy


nlp = spacy.load("en_core_web_sm")


def build_entity_graph(question: str):
    """
    Build entity graph using spaCy entities.

    Nodes:
        entity text + semantic label

    Edges:
        sequential entity co-occurrence
    """

    doc = nlp(question)

    entities = []

    for ent in doc.ents:

        entity_text = ent.text.strip().lower()

        if not entity_text:
            continue

        entities.append(
            {
                "text": entity_text,
                "label": ent.label_
            }
        )

    G = nx.Graph()

    # -------------------------
    # add nodes with metadata
    # -------------------------

    for entity in entities:

        G.add_node(
            entity["text"],
            entity_type=entity["label"]
        )

    # -------------------------
    # add edges
    # -------------------------

    for i in range(len(entities) - 1):

        source = entities[i]["text"]
        target = entities[i + 1]["text"]

        G.add_edge(source, target)

    return G