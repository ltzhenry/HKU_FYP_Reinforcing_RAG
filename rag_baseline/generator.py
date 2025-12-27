from __future__ import annotations

import os
from typing import Iterable, List

from .retriever import RetrievedChunk

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("openai is required. Install with `pip install openai`.") from exc


class Generator:
    """
    Prompt builder and LLM caller. The prompt is intentionally simple and
    instructs the model to rely only on provided documents.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    @staticmethod
    def build_prompt(query: str, contexts: Iterable[RetrievedChunk]) -> str:
        context_texts = []
        for idx, chunk in enumerate(contexts, start=1):
            context_texts.append(f"[Document {idx}]\n{chunk.text}")
        context_block = "\n\n".join(context_texts) if context_texts else "No documents retrieved."
        instructions = (
            "You are answering a question using only the provided documents. "
            "If the documents do not contain enough information, respond with "
            '"The provided documents do not contain enough information to answer this question."'
        )
        return f"{instructions}\n\nQuestion:\n{query}\n\nDocuments:\n{context_block}\n\nAnswer:"

    def generate(self, query: str, contexts: List[RetrievedChunk]) -> str:
        prompt = self.build_prompt(query, contexts)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer concisely using only the provided documents."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


