# Minimal RAG Baseline

This repository contains a traditional Retrieval-Augmented Generation (RAG) pipeline implemented in plain Python components. It is intentionally minimal so additional modules (keyword retrieval, reranking, verification, etc.) can be added later without changing the core flow.

## Pipeline stages
- Ingestion & chunking (`rag_baseline/ingestion.py`)
- Embedding (`rag_baseline/embeddings.py`)
- Vector storage (`rag_baseline/vector_store.py`)
- Retrieval (`rag_baseline/retriever.py`)
- Generation (`rag_baseline/generator.py`)
- Orchestration (`rag_baseline/pipeline.py`)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
```

## Run the example
```bash
python main.py
```

The example indexes three toy documents, performs top-k similarity search over embedded chunks, and generates an answer constrained to the retrieved text. If the retrieved context is insufficient, the prompt instructs the model to reply with:  
`The provided documents do not contain enough information to answer this question.`


