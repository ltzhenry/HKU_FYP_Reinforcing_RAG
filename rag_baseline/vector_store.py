from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("faiss-cpu is required. Install with `pip install faiss-cpu`.") from exc


class VectorStore:
    """
    Simple FAISS-backed vector store that keeps embeddings and metadata in memory.
    """

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, str]] = []

    def add(self, embeddings: np.ndarray, texts: Iterable[str], metadatas: Iterable[Dict[str, str]]) -> None:
        metadata_list = list(metadatas)
        text_list = list(texts)
        if len(text_list) != embeddings.shape[0] or len(metadata_list) != embeddings.shape[0]:
            raise ValueError("Embeddings, texts, and metadata must be aligned.")
        self.index.add(np.asarray(embeddings, dtype="float32"))
        self.texts.extend(text_list)
        self.metadatas.extend(metadata_list)

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple[str, float, Dict[str, str]]]]:
        distances, indices = self.index.search(np.asarray(query_embeddings, dtype="float32"), top_k)
        results: List[List[Tuple[str, float, Dict[str, str]]]] = []
        for row_distances, row_indices in zip(distances, indices):
            row_results: List[Tuple[str, float, Dict[str, str]]] = []
            for dist, idx in zip(row_distances, row_indices):
                if idx == -1:
                    continue
                row_results.append((self.texts[idx], float(dist), self.metadatas[idx]))
            results.append(row_results)
        return results


