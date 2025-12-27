from __future__ import annotations

from typing import Iterable, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "sentence-transformers is required for embedding. "
        "Install with `pip install sentence-transformers`."
    ) from exc


class Embedder:
    """
    Thin wrapper around a sentence-transformers model.
    The class is intentionally minimal so the embedding backend can be swapped.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False))


