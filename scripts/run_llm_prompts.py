# run_llm_benchmark.py

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# OPTIONAL DEEPSEEK
# ---------------------------------------------------------
# Uncomment later when using DeepSeek
#
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "data/prompts_hallucination/medchangeqa_metric_extremes.csv"

OUTPUT_DIR = "data/llm_outputs"

OUTPUT_TEXT_FILE = "medchangeqa_benchmark_results.txt"

OUTPUT_CSV_FILE = "medchangeqa_benchmark_results.csv"

TEMPERATURE = 0.0

MAX_TOKENS = 50

SLEEP_TIME = 1


# =========================================================
# MODELS
# =========================================================

MODELS = {

    # -----------------------------------------------------
    # GPT
    # -----------------------------------------------------

    "gpt-4.1-mini": {
        "provider": "openai",
        "model_name": "gpt-4.1-mini"
    },

    # -----------------------------------------------------
    # DEEPSEEK
    # -----------------------------------------------------
    # Uncomment when ready
    #
    # "deepseek-chat": {
    #     "provider": "deepseek",
    #     "model_name": "deepseek-chat"
    # }
}


# =========================================================
# CLIENTS
# =========================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

# ---------------------------------------------------------
# OPTIONAL DEEPSEEK CLIENT
# ---------------------------------------------------------
# Uncomment later
#
# deepseek_client = OpenAI(
#     api_key=DEEPSEEK_API_KEY,
#     base_url="https://api.deepseek.com"
# )


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

TEXT_OUTPUT_PATH = (
    Path(OUTPUT_DIR)
    / OUTPUT_TEXT_FILE
)

CSV_OUTPUT_PATH = (
    Path(OUTPUT_DIR)
    / OUTPUT_CSV_FILE
)


# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(CSV_PATH)

print(f"\nLoaded {len(df)} prompts")


# =========================================================
# PROMPT TEMPLATE
# =========================================================

SYSTEM_PROMPT = """
You are a biomedical QA assistant.

Answer briefly and directly. Answer in either SUPPORTED if statement is true, REFUTED if statement is false or NEI for not enough information.

Use:
- one word
- one phrase
- or one short sentence

Do not explain your reasoning.
"""


# =========================================================
# QUERY FUNCTION
# =========================================================

def query_model(
    client,
    model_name,
    question
):

    response = client.chat.completions.create(

        model=model_name,

        temperature=TEMPERATURE,

        max_tokens=MAX_TOKENS,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": question
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# RUN BENCHMARK
# =========================================================

all_results = []

with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as txt_file:

    txt_file.write("\n")
    txt_file.write("=" * 100 + "\n")
    txt_file.write("LLM BENCHMARK RESULTS\n")
    txt_file.write("=" * 100 + "\n\n")

    # =====================================================
    # LOOP THROUGH PROMPTS
    # =====================================================

    for idx, row in df.iterrows():

        metric = row["metric"]

        extreme_type = row["extreme_type"]

        rank = row["rank"]

        value = row["value"]

        question = row["question"]

        print("\n" + "=" * 80)
        print(f"[{idx + 1}/{len(df)}]")
        print(f"{metric} | {extreme_type} | rank {rank}")
        print(question)
        print("=" * 80)

        # -------------------------------------------------
        # WRITE HEADER
        # -------------------------------------------------

        txt_file.write("\n")
        txt_file.write("=" * 100 + "\n")

        txt_file.write(
            f"METRIC: {metric}\n"
        )

        txt_file.write(
            f"TYPE: {extreme_type}\n"
        )

        txt_file.write(
            f"RANK: {rank}\n"
        )

        txt_file.write(
            f"VALUE: {value}\n"
        )

        txt_file.write("=" * 100 + "\n\n")

        txt_file.write(
            f"QUESTION:\n{question}\n\n"
        )

        # =================================================
        # RUN MODELS
        # =================================================

        for model_alias, model_info in MODELS.items():

            provider = model_info["provider"]

            model_name = model_info["model_name"]

            print(f"\nRunning: {model_alias}")

            try:

                # -----------------------------------------
                # OPENAI
                # -----------------------------------------

                if provider == "openai":

                    answer = query_model(
                        openai_client,
                        model_name,
                        question
                    )

                # -----------------------------------------
                # DEEPSEEK
                # -----------------------------------------

                elif provider == "deepseek":

                    answer = query_model(
                        deepseek_client,
                        model_name,
                        question
                    )

                else:

                    raise ValueError(
                        f"Unknown provider: {provider}"
                    )

                print(f"Answer: {answer}")

            except Exception as e:

                answer = f"ERROR: {str(e)}"

                print(answer)

            # =============================================
            # WRITE RESPONSE
            # =============================================

            txt_file.write(
                f"MODEL: {model_alias}\n"
            )

            txt_file.write(
                f"RESPONSE: {answer}\n\n"
            )

            # =============================================
            # SAVE STRUCTURED CSV RESULT
            # =============================================

            all_results.append({

                "metric": metric,

                "extreme_type": extreme_type,

                "rank": rank,

                "metric_value": value,

                "question": question,

                "model": model_alias,

                "response": answer
            })

            time.sleep(SLEEP_TIME)

        txt_file.write("\n")


# =========================================================
# SAVE CSV
# =========================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    CSV_OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print(f"\nText results saved to:")
print(TEXT_OUTPUT_PATH)

print(f"\nCSV results saved to:")
print(CSV_OUTPUT_PATH)