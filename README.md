# Minimal RAG Baseline

This repository contains a traditional Retrieval-Augmented Generation (RAG) pipeline implemented in plain Python components. It is intentionally minimal so additional modules (keyword retrieval, reranking, verification, etc.) can be added later without changing the core flow.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
```

## Run the interactive platform
```bash
python interactive.py
```


