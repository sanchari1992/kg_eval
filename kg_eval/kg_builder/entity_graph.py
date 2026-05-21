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
        "parser",
        "lemmatizer",
        "attribute_ruler"
    ]
)


# ===================================================
# Lightweight ontology depth estimates
# ===================================================

ONTOLOGY_DEPTHS = {

    # ---------------------------------
    # simple consumer concepts
    # ---------------------------------

    "cold": 1,
    "common cold": 1,
    "blister": 1,
    "zinc lozenges": 2,

    # ---------------------------------
    # moderate clinical concepts
    # ---------------------------------

    "hypertension": 3,
    "hypercholesterolemia": 3,
    "arthritis": 3,
    "knee pain": 3,
    "tantrums": 2,

    # ---------------------------------
    # advanced biomedical concepts
    # ---------------------------------

    "autophagic cell death": 7,
    "doxorubicin-resistant": 8,
    "xenografts": 8,
    "sirtuin 1": 7,
    "mcf-7/adr": 8,
    "monosodium urate crystals": 6,
    "spinocerebellar ataxia type 3": 7,
    "pilomatricoma": 6,
}


def lookup_ontology_depth(entity_text):
    """
    Approximate biomedical specialization depth.

    Higher values indicate:
    - more technical concepts
    - ontology-deep biomedical entities
    """

    return ONTOLOGY_DEPTHS.get(
        entity_text.lower(),
        2
    )


# ===================================================
# Node role assignment
# ===================================================

def assign_node_role(entity, total_entities, idx):
    """
    Approximate contextual role.

    Heuristic:
    - earlier entities tend to be context
    - later entities closer to question intent

    Helps distinguish:
    - clinical narrative questions
    - direct consumer questions
    """

    if total_entities <= 2:
        return "intent"

    # later entities = likely intent
    if idx >= total_entities * 0.6:
        return "intent"

    return "context"


# ===================================================
# Single graph builder
# ===================================================

def build_entity_graph(question: str):
    """
    Build entity graph using spaCy entities.

    Nodes:
        entity text + semantic annotations

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
    # Add nodes with semantic annotations
    # ---------------------------------------------------

    for idx, entity in enumerate(entities):

        G.add_node(

            entity["text"],

            entity_type=entity["label"],

            node_role=assign_node_role(
                entity,
                len(entities),
                idx
            ),

            ontology_depth=lookup_ontology_depth(
                entity["text"]
            )
        )

    # ---------------------------------------------------
    # Add sequential edges
    # ---------------------------------------------------

    for i in range(len(entities) - 1):

        source = entities[i]["text"]
        target = entities[i + 1]["text"]

        G.add_edge(source, target)

    return G


# ===================================================
# Batch graph builder
# ===================================================

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

        # ---------------------------------------------
        # Extract entities
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Add annotated nodes
        # ---------------------------------------------

        for idx, entity in enumerate(entities):

            G.add_node(

                entity["text"],

                entity_type=entity["label"],

                node_role=assign_node_role(
                    entity,
                    len(entities),
                    idx
                ),

                ontology_depth=lookup_ontology_depth(
                    entity["text"]
                )
            )

        # ---------------------------------------------
        # Add sequential edges
        # ---------------------------------------------

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