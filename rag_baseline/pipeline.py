from __future__ import annotations

from typing import Iterable, List

from .embeddings import Embedder
from .generator import Generator
from .ingestion import Chunk, Document, chunk_corpus
from .retriever import RetrievedChunk, Retriever
from .vector_store import VectorStore


class RagPipeline:
    """
    End-to-end baseline RAG pipeline with clear separation of stages:
    ingestion -> embedding -> vector storage -> retrieval -> generation.
    """

    def __init__(self, embedder: Embedder, generator: Generator, top_k: int = 5):
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.vector_store: VectorStore | None = None
        self.retriever: Retriever | None = None

    def index(self, documents: Iterable[Document], max_tokens: int = 200, overlap: int = 50) -> None:
        chunks: List[Chunk] = chunk_corpus(documents, max_tokens=max_tokens, overlap=overlap)
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        vector_store = VectorStore(dim=embeddings.shape[1])
        vector_store.add(embeddings, [chunk.text for chunk in chunks], [chunk.metadata for chunk in chunks])
        self.vector_store = vector_store
        self.retriever = Retriever(self.embedder, vector_store, top_k=self.top_k)

    def answer(self, query: str) -> tuple[str, List[RetrievedChunk]]:
        if not self.retriever:
            raise RuntimeError("The pipeline has not been indexed yet. Call index() first.")
        retrieved = self.retriever.retrieve(query)
        answer = self.generator.generate(query, retrieved)
        return answer, retrieved


