import re
import pandas as pd


# =========================================================
# PATHS
# =========================================================

LLM_FILE = "data/llm_outputs/bioasq_benchmark_results.csv"

GOLD_FILE = "data/raw/bioasq/train.csv"

OUTPUT_FILE = "data/llm_outputs/bioasq_evaluation_results.csv"

DETAILED_LOG_FILE = "data/llm_outputs/bioasq_detailed_logs.csv"


# =========================================================
# LOAD DATA
# =========================================================

llm_df = pd.read_csv(LLM_FILE)

gold_df = pd.read_csv(GOLD_FILE)

print("LLM rows:", len(llm_df))
print("Gold rows:", len(gold_df))


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# EXTRACT GOLD ANSWER
# =========================================================

def extract_gold_answer(text):

    if pd.isna(text):
        return ""

    match = re.search(
        r"<answer>\s*(.*?)\s*<context>",
        str(text),
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


# =========================================================
# PREP GROUND TRUTH
# =========================================================

gold_df["gold_answer"] = gold_df["text"].apply(extract_gold_answer)

gold_df["question_norm"] = gold_df["question"].apply(normalize)

gold_lookup = dict(
    zip(
        gold_df["question_norm"],
        gold_df["gold_answer"]
    )
)


def get_gold(question):

    return gold_lookup.get(normalize(question), None)


# =========================================================
# ATTACH GOLD TO LLM OUTPUT
# =========================================================

llm_df["gold_answer"] = llm_df["question"].apply(get_gold)


# =========================================================
# METRICS
# =========================================================

def exact_match(pred, gold):

    if not gold or not pred:
        return 0

    return int(normalize(pred) == normalize(gold))


def contains_match(pred, gold):

    if not gold or not pred:
        return 0

    return int(normalize(gold) in normalize(pred))


# =========================================================
# APPLY METRICS
# =========================================================

llm_df["exact_match"] = llm_df.apply(
    lambda x: exact_match(x["response"], x["gold_answer"]),
    axis=1
)

llm_df["contains_match"] = llm_df.apply(
    lambda x: contains_match(x["response"], x["gold_answer"]),
    axis=1
)


# =========================================================
# CREATE DETAILED LOGS (IMPORTANT FOR PAPER)
# =========================================================

detailed_logs = llm_df[[
    "metric",
    "extreme_type",
    "rank",
    "model",
    "question",
    "gold_answer",
    "response",
    "exact_match",
    "contains_match"
]].copy()

# Optional: rename for clarity in paper tables
detailed_logs.rename(columns={
    "question": "prompt",
    "response": "model_answer"
}, inplace=True)


# =========================================================
# AGGREGATED RESULTS
# =========================================================

summary = llm_df.groupby(
    ["model", "metric", "extreme_type"]
).agg(
    accuracy=("exact_match", "mean"),
    soft_accuracy=("contains_match", "mean"),
    total=("exact_match", "count")
).reset_index()


print("\n=== SUMMARY ===\n")
print(summary)


# =========================================================
# SAVE FILES
# =========================================================

llm_df.to_csv(OUTPUT_FILE, index=False)

summary.to_csv(
    "data/llm_outputs/summary_metrics.csv",
    index=False
)

detailed_logs.to_csv(
    DETAILED_LOG_FILE,
    index=False
)


# =========================================================
# DONE
# =========================================================

print("\nSaved files:")
print(OUTPUT_FILE)
print("data/llm_outputs/summary_metrics.csv")
print(DETAILED_LOG_FILE)