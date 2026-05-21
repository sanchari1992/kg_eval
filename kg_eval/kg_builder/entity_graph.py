import networkx as nx
import spacy

from tqdm import tqdm


# ---------------------------------------------------
# Load spaCy model ONCE
# ---------------------------------------------------

nlp = spacy.load(
    "en_core_web_sm",
    disable=[
        "tagger",
        "lemmatizer",
        "attribute_ruler"
    ]
)


# ===================================================
# Node role assignment
# ===================================================

def assign_node_role(entity, total_entities, idx):
    """
    Heuristic:
    - early entities = context
    - late entities = intent
    """

    if total_entities <= 2:
        return "intent"

    if idx >= total_entities * 0.6:
        return "intent"

    return "context"


# ===================================================
# Grammar-based sentence complexity helpers
# ===================================================

def compute_dependency_depth(token):
    """
    Approximate syntactic tree depth using recursion.
    """

    if not list(token.children):
        return 1

    return 1 + max(
        compute_dependency_depth(child)
        for child in token.children
    )


def sentence_complexity(doc):
    """
    Mean dependency depth across tokens.
    """
    depths = [compute_dependency_depth(t) for t in doc]
    return sum(depths) / len(depths) if depths else 0.0


# ===================================================
# Graph builder
# ===================================================

def build_entity_graph(question: str):
    """
    Build entity graph using spaCy entities + grammar signals.
    """

    doc = nlp(question)

    entities = []

    for ent in doc.ents:

        text = ent.text.strip().lower()
        if not text:
            continue

        entities.append({
            "text": text,
            "label": ent.label_
        })

    G = nx.Graph()

    # store sentence-level grammar feature at graph level
    G.graph["sentence_complexity"] = sentence_complexity(doc)

    # -------------------------
    # nodes
    # -------------------------
    for idx, ent in enumerate(entities):

        G.add_node(
            ent["text"],
            entity_type=ent["label"],
            node_role=assign_node_role(ent, len(entities), idx)
        )

    # -------------------------
    # edges
    # -------------------------
    for i in range(len(entities) - 1):
        G.add_edge(
            entities[i]["text"],
            entities[i + 1]["text"]
        )

    return G


# ===================================================
# Batch builder
# ===================================================

def build_entity_graphs_batch(
    questions,
    batch_size=64,
    show_progress=True
):

    graph_cache = {}
    unique_questions = []

    for q in questions:

        norm = q.strip().lower()

        if norm not in graph_cache:
            graph_cache[norm] = None
            unique_questions.append(q)

    docs = nlp.pipe(unique_questions, batch_size=batch_size)

    if show_progress:
        docs = tqdm(docs, total=len(unique_questions), desc="Building graphs")

    for question, doc in zip(unique_questions, docs):

        entities = []

        for ent in doc.ents:

            text = ent.text.strip().lower()
            if not text:
                continue

            entities.append({
                "text": text,
                "label": ent.label_
            })

        G = nx.Graph()

        G.graph["sentence_complexity"] = sentence_complexity(doc)

        for idx, ent in enumerate(entities):

            G.add_node(
                ent["text"],
                entity_type=ent["label"],
                node_role=assign_node_role(ent, len(entities), idx)
            )

        for i in range(len(entities) - 1):
            G.add_edge(
                entities[i]["text"],
                entities[i + 1]["text"]
            )

        graph_cache[question.strip().lower()] = G

    return [
        graph_cache[q.strip().lower()]
        for q in questions
    ]