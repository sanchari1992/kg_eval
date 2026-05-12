import pandas as pd

from kg_eval.kg_builder.simple_graph import build_question_graph


def load_dataset(file_path: str):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format")


def extract_graphs(df: pd.DataFrame, question_col: str):
    graphs = []

    for q in df[question_col].dropna():
        G = build_question_graph(q)
        graphs.append(G)

    return graphs