from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .embeddings import Embedder
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: Dict[str, str]


class Retriever:
    """
    Minimal retriever: embed the query, run top-k similarity search, return raw chunks.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query_vec = self.embedder.embed([query])
        search_results = self.vector_store.search(query_vec, top_k=self.top_k)[0]
        return [RetrievedChunk(text=text, score=score, metadata=meta) for text, score, meta in search_results]


