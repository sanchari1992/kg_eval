import networkx as nx
import spacy

from tqdm import tqdm


# ---------------------------------------------------
# Load spaCy model ONCE
# Disable unnecessary components for speed
# ---------------------------------------------------

nlp = spacy.load(
    "en_core_web_sm",
    disable=[
        "tagger",
        "parser",
        "lemmatizer",
        "attribute_ruler"
    ]
)


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

    # ---------------------------------------------------
    # Extract entities
    # ---------------------------------------------------

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

    # ---------------------------------------------------
    # Add nodes
    # ---------------------------------------------------

    for entity in entities:

        G.add_node(
            entity["text"],
            entity_type=entity["label"]
        )

    # ---------------------------------------------------
    # Add sequential edges
    # ---------------------------------------------------

    for i in range(len(entities) - 1):

        source = entities[i]["text"]
        target = entities[i + 1]["text"]

        G.add_edge(source, target)

    return G


def build_entity_graphs_batch(
    questions,
    batch_size=64,
    show_progress=True
):
    """
    Faster batch graph generation using spaCy pipe.

    Avoids recomputing duplicate questions
    using an internal graph cache.

    Parameters
    ----------
    questions : list[str]

    batch_size : int

    show_progress : bool

    Returns
    -------
    list[nx.Graph]
    """

    graphs = []

    # ---------------------------------------------------
    # Cache already-built graphs
    # ---------------------------------------------------

    graph_cache = {}

    # ---------------------------------------------------
    # Find unique questions only
    # ---------------------------------------------------

    unique_questions = []

    for question in questions:

        normalized_question = question.strip().lower()

        if normalized_question not in graph_cache:

            graph_cache[normalized_question] = None

            unique_questions.append(question)

    # ---------------------------------------------------
    # Run spaCy only on unique questions
    # ---------------------------------------------------

    docs = nlp.pipe(
        unique_questions,
        batch_size=batch_size
    )

    # ---------------------------------------------------
    # Optional progress bar
    # ---------------------------------------------------

    if show_progress:

        docs = tqdm(
            docs,
            total=len(unique_questions),
            desc="Building entity graphs"
        )

    # ---------------------------------------------------
    # Build graphs for unique questions
    # ---------------------------------------------------

    for question, doc in zip(unique_questions, docs):

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

        for entity in entities:

            G.add_node(
                entity["text"],
                entity_type=entity["label"]
            )

        for i in range(len(entities) - 1):

            source = entities[i]["text"]
            target = entities[i + 1]["text"]

            G.add_edge(source, target)

        normalized_question = question.strip().lower()

        graph_cache[normalized_question] = G

    # ---------------------------------------------------
    # Reconstruct graph list in original order
    # ---------------------------------------------------

    for question in questions:

        normalized_question = question.strip().lower()

        graphs.append(
            graph_cache[normalized_question]
        )

    return graphs