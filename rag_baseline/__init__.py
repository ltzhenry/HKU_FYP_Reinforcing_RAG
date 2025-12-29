"""
Minimal, modular Retrieval-Augmented Generation (RAG) baseline.

The package exposes small, replaceable components for ingestion, embedding,
vector storage, retrieval, and generation. Advanced behaviors such as
query rewriting or reranking are intentionally omitted to keep the baseline
simple and extensible.
"""


"""
RAG Baseline package with reasoning capabilities.
"""

from .embeddings import Embedder
from .generator import Generator
from .ingestion import Document, Chunk, chunk_corpus, chunk_document
from .pipeline import RagPipeline
from .retriever import Retriever, RetrievedChunk
from .vector_store import VectorStore

# Reasoning components
from .reasoning import QuestionAnalyzer, QuestionAnalysis, QuestionComplexity
from .decomposer import QueryDecomposer, SubQuery
from .multi_hop_retriever import MultiHopRetriever, SubQueryResult
from .evidence_integrator import EvidenceIntegrator, IntegratedEvidence
from .reasoning_pipeline import ReasoningRagPipeline

__all__ = [
    # Original components
    "Embedder",
    "Generator",
    "Document",
    "Chunk",
    "chunk_corpus",
    "chunk_document",
    "RagPipeline",
    "Retriever",
    "RetrievedChunk",
    "VectorStore",
    # Reasoning components
    "QuestionAnalyzer",
    "QuestionAnalysis",
    "QuestionComplexity",
    "QueryDecomposer",
    "SubQuery",
    "MultiHopRetriever",
    "SubQueryResult",
    "EvidenceIntegrator",
    "IntegratedEvidence",
    "ReasoningRagPipeline",
]