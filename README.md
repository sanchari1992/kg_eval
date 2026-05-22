# KG Eval (MedAuditKG Toolkit)

KG Eval is a lightweight toolkit for generating and evaluating knowledge graphs from text datasets.

It is the official implementation of **MedAuditKG**, a graph-based audit framework for healthcare question-answering benchmarks.

This toolkit enables structural and semantic analysis of QA datasets using graph-based representations.

---

# 📄 Paper Reference

This repository implements the methods described in:

**A Graph-Based Audit Framework for Healthcare Question-Answering Benchmarks**

> The paper link will be added upon publication.

---

# ⚠️ Naming Clarification

To avoid confusion:

- **MedAuditKG** → Name of the research framework (paper)
- **KG Eval** → Name of this codebase / implementation toolkit

Both refer to the same system.

---

# 🧠 Framework Overview

The framework is designed for:

- healthcare and biomedical NLP research
- hallucination analysis in QA systems
- dataset comparison across benchmarks
- structural and semantic graph evaluation
- benchmarking dataset complexity via graph metrics

---

# Features

- Load datasets from:
  - Hugging Face
  - CSV
  - Parquet

- Extract question/text columns

- Generate knowledge graphs using:
  - spaCy named entity recognition
  - NetworkX graph backend

- Compute graph metrics:
  - number of nodes
  - number of edges
  - graph density
  - average degree
  - connected components
  - graph diameter

- Export JSON reports

- Visualize generated graphs

---

# Current Graph Model

- Nodes:
  - named entities extracted using spaCy

- Edges:
  - sequential entity co-occurrence

- Graph backend:
  - NetworkX

---

# Project Structure

```text
kg_eval/
├── data/
│   └── raw/
├── scripts/
│   ├── download_medhalt.py
│   ├── download_medhallu.py
│   └── test_graph.py
├── kg_eval/
│   ├── ingestion/
│   ├── kg_builder/
│   ├── metrics/
│   ├── pipeline/
│   ├── reports/
│   └── visualization/
├── pyproject.toml
└── README.md
````

---

# Installation

## Option 1: Install from GitHub (recommended)

```bash
pip install git+https://github.com/YOUR_USERNAME/kg-eval.git
```

---

## Option 2: Development install

```bash
git clone https://github.com/YOUR_USERNAME/kg-eval.git
cd kg-eval
pip install -e .
```

---

## 3. Create virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4. Install spaCy language model

```bash
python -m spacy download en_core_web_sm
```

This model is required for entity extraction.

---

# Supported Datasets

Currently tested on:

### QA datasets

* MedQA
* MedMCQA
* PubMedQA
* BioASQ
* BiQA
* HealthSearchQA
* LiveQA
* MedicationQA
* MedQuAD
* MediQA
* MedRevQA
* MedChangeQA
* MedAESQA
* COVID-QA

### Non-QA datasets

* Med-HALT
* MedHallu
* Medhal

---

# Downloading Datasets

## Med-HALT

Dataset:
[https://huggingface.co/datasets/openlifescienceai/Med-HALT](https://huggingface.co/datasets/openlifescienceai/Med-HALT)

Run:

```bash
python scripts/download_medhalt.py
```

Downloaded files will be stored in:

```text
data/raw/medhalt/
```

---

## MedHallu

Dataset:
[https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)

Run:

```bash
python scripts/download_medhallu.py
```

Downloaded files will be stored in:

```text
data/raw/medhallu/
```

---

# Running the Knowledge Graph Pipeline

The main pipeline script is:

```text
scripts/test_graph.py
```

This script:

1. Loads a dataset
2. Extracts the question column
3. Builds knowledge graphs
4. Computes graph metrics
5. Saves a JSON report
6. Optionally visualizes graphs

---

# Usage

## General Command

### Windows

```bash
python scripts/test_graph.py ^
  --file_path PATH_TO_DATASET ^
  --question_col COLUMN_NAME ^
  --report_path OUTPUT_REPORT ^
  --visualize
```

### Mac/Linux

```bash
python scripts/test_graph.py \
  --file_path PATH_TO_DATASET \
  --question_col COLUMN_NAME \
  --report_path OUTPUT_REPORT \
  --visualize
```

---

# Example: Med-HALT

```bash
python scripts/test_graph.py ^
  --file_path data/raw/medhalt/train.csv ^
  --question_col question ^
  --report_path data/raw/medhalt/report.json ^
  --visualize
```

---

# Example: MedHallu

```bash
python scripts/test_graph.py ^
  --file_path data/raw/medhallu/train.csv ^
  --question_col Question ^
  --report_path data/raw/medhallu/report.json ^
  --visualize
```

---

# Output

## Console Output

* dataset information
* graph statistics
* sample metrics

## JSON Report

Example:

```text
data/raw/medhalt/report.json
```

Contains:

* dataset-level summary metrics
* per-graph metrics

## Graph Visualization

If `--visualize` is enabled:

* first graph is rendered using matplotlib

---

# Example Workflow

## 1. Download dataset

```bash
python scripts/download_medhalt.py
```

## 2. Run pipeline

```bash
python scripts/test_graph.py ^
  --file_path data/raw/medhalt/train.csv ^
  --question_col question ^
  --report_path data/raw/medhalt/report.json ^
  --visualize
```

---

# Current Limitations

Current graphs use:

* spaCy entity extraction
* sequential entity relationships

This is an MVP graph representation.

Future improvements:

* UMLS concept linking
* biomedical ontology integration
* relation extraction
* semantic graph metrics
* multi-dataset benchmarking

---

# Future Roadmap

* SciSpacy biomedical NER
* UMLS integration
* ontology-aware graphs
* semantic metrics
* graph comparison benchmarking
* CLI package support
* publication-ready visualizations

---

# License

Copyright (c) 2026 Sanchari Biswas

This project is licensed under the MIT License.
