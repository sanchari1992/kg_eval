import pandas as pd
from kg_eval.kg_builder.entity_graph import build_entity_graph


def load_dataset(file_path: str):

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)

    else:
        raise ValueError("Unsupported format")


def extract_graphs(df: pd.DataFrame, question_col: str):

    graphs = []

    for q in df[question_col]:

        if q is None:
            continue

        q = str(q).strip()

        if not q:
            continue

        G = build_entity_graph(q)

        if G.number_of_nodes() > 0:
            graphs.append(G)

    return graphs