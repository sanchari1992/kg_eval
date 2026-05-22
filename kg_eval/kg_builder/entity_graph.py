import networkx as nx
import spacy
from tqdm import tqdm

# ===================================================
# spaCy model
# ===================================================

nlp = spacy.load("en_core_web_sm")


# ===================================================
# ROLE ASSIGNMENT
# ===================================================

def assign_node_role(entity, total_entities, idx):

    if total_entities <= 2:
        return "intent"

    if idx >= total_entities * 0.6:
        return "intent"

    return "context"


# ===================================================
# DEPENDENCY HELPERS
# ===================================================

def get_dependency_depth(token):

    depth = 0
    current = token

    while current.head != current:
        depth += 1
        current = current.head

    return depth


def compute_dependency_depth(token):

    children = list(token.children)

    if not children:
        return 1

    return 1 + max(compute_dependency_depth(c) for c in children)


def sentence_complexity(doc):

    depths = [compute_dependency_depth(t) for t in doc]
    return sum(depths) / len(depths) if depths else 0.0


# ===================================================
# SAFE TEXT
# ===================================================

def clean_question(q):

    if q is None:
        return ""

    q = str(q).strip().lower()

    return q


# ===================================================
# GRAPH BUILDER
# ===================================================

def build_entity_graph(question: str):

    G = nx.Graph()

    question = clean_question(question)

    if not question:
        return G

    doc = nlp(question)

    # -------------------------
    # graph-level features
    # -------------------------

    alpha_tokens = [t for t in doc if t.is_alpha]

    G.graph["sentence_complexity"] = sentence_complexity(doc)

    G.graph["num_tokens"] = len(alpha_tokens)

    G.graph["avg_token_length"] = (
        sum(len(t.text) for t in alpha_tokens)
        / len(alpha_tokens)
        if alpha_tokens else 0.0
    )

    G.graph["content_word_ratio"] = (
        len([t for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}])
        / len(doc)
        if len(doc) > 0 else 0.0
    )

    G.graph["clause_count"] = len([
        t for t in doc
        if t.dep_ in {"ccomp", "xcomp", "advcl", "relcl"}
    ])

    # -------------------------
    # entities
    # -------------------------

    entities = []

    for ent in doc.ents:

        text = ent.text.strip().lower()
        if not text:
            continue

        entities.append(ent)

    # -------------------------
    # nodes
    # -------------------------

    for idx, ent in enumerate(entities):

        root = ent.root

        G.add_node(
            ent.text.lower(),

            entity_type=ent.label_,
            node_role=assign_node_role(ent, len(entities), idx),

            token_text=root.text,
            lemma=root.lemma_.lower(),
            pos=root.pos_,
            dep=root.dep_,
            dependency_depth=get_dependency_depth(root)
        )

    # -------------------------
    # edges
    # -------------------------

    for i in range(len(entities) - 1):

        G.add_edge(
            entities[i].text.lower(),
            entities[i + 1].text.lower()
        )

    return G


# ===================================================
# BATCH BUILDER
# ===================================================

def build_entity_graphs_batch(questions, batch_size=64, show_progress=True):

    questions = [clean_question(q) for q in questions]
    questions = [q for q in questions if q]

    docs = nlp.pipe(questions, batch_size=batch_size)

    if show_progress:
        docs = tqdm(docs, total=len(questions), desc="Building graphs")

    graphs = []

    for doc in docs:

        G = nx.Graph()

        alpha_tokens = [t for t in doc if t.is_alpha]

        G.graph["sentence_complexity"] = sentence_complexity(doc)

        G.graph["num_tokens"] = len(alpha_tokens)

        G.graph["avg_token_length"] = (
            sum(len(t.text) for t in alpha_tokens)
            / len(alpha_tokens)
            if alpha_tokens else 0.0
        )

        G.graph["content_word_ratio"] = (
            len([t for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}])
            / len(doc)
            if len(doc) > 0 else 0.0
        )

        G.graph["clause_count"] = len([
            t for t in doc
            if t.dep_ in {"ccomp", "xcomp", "advcl", "relcl"}
        ])

        entities = [ent for ent in doc.ents if ent.text.strip()]

        for idx, ent in enumerate(entities):

            root = ent.root

            G.add_node(
                ent.text.lower(),
                entity_type=ent.label_,
                node_role=assign_node_role(ent, len(entities), idx),
                token_text=root.text,
                lemma=root.lemma_.lower(),
                pos=root.pos_,
                dep=root.dep_,
                dependency_depth=get_dependency_depth(root)
            )

        for i in range(len(entities) - 1):

            G.add_edge(
                entities[i].text.lower(),
                entities[i + 1].text.lower()
            )

        graphs.append(G)

    return graphs