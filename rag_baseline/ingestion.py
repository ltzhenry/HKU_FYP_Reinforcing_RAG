from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Optional[Dict[str, str]] = None


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, str]


def _sliding_window(tokens: List[str], size: int, overlap: int) -> Iterable[List[str]]:
    step = max(size - overlap, 1)
    for start in range(0, len(tokens), step):
        yield tokens[start : start + size]


def chunk_document(
    document: Document, max_tokens: int = 200, overlap: int = 50
) -> List[Chunk]:
    """
    Naive whitespace-based chunking with overlap to preserve context.
    The function intentionally avoids advanced parsing to keep the baseline simple.
    """
    tokens = document.text.split()
    chunks: List[Chunk] = []
    for idx, token_window in enumerate(_sliding_window(tokens, max_tokens, overlap)):
        chunk_text = " ".join(token_window)
        chunk_metadata: Dict[str, str] = {"chunk_index": str(idx), "source_doc_id": document.doc_id}
        if document.metadata:
            chunk_metadata.update(document.metadata)
        chunks.append(
            Chunk(
                doc_id=document.doc_id,
                chunk_id=f"{document.doc_id}:{idx}",
                text=chunk_text,
                metadata=chunk_metadata,
            )
        )
    return chunks


def chunk_corpus(
    documents: Iterable[Document], max_tokens: int = 200, overlap: int = 50
) -> List[Chunk]:
    """Chunk multiple documents into a flat list of chunks."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, max_tokens=max_tokens, overlap=overlap))
    return all_chunks


