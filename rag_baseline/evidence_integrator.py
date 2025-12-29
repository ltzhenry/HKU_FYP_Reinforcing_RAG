from __future__ import annotations

"""
Evidence integration and validation across sub-queries.
"""

from dataclasses import dataclass
from typing import List

from .multi_hop_retriever import SubQueryResult
from .retriever import RetrievedChunk


@dataclass
class IntegratedEvidence:
    """Integrated and validated evidence from multiple sub-queries."""
    all_chunks: List[RetrievedChunk]
    summary: str
    confidence_score: float


class EvidenceIntegrator:
    """
    Integrates and validates evidence from multiple sub-query results.
    """

    def integrate(self, sub_query_results: List[SubQueryResult]) -> IntegratedEvidence:
        """
        Merge and deduplicate evidence from all sub-queries.
        """
        all_chunks: List[RetrievedChunk] = []
        seen_texts = set()

        # Collect all unique chunks
        for result in sub_query_results:
            for chunk in result.retrieved_chunks:
                # Simple deduplication by exact text match
                if chunk.text not in seen_texts:
                    all_chunks.append(chunk)
                    seen_texts.add(chunk.text)

        # Sort by relevance score (lower L2 distance = more relevant)
        all_chunks.sort(key=lambda c: c.score)

        # Generate summary
        summary = self._generate_summary(sub_query_results)

        # Calculate confidence (simple heuristic: based on number of supporting chunks)
        confidence = min(1.0, len(all_chunks) / 10.0)

        return IntegratedEvidence(
            all_chunks=all_chunks,
            summary=summary,
            confidence_score=confidence
        )

    def _generate_summary(self, sub_query_results: List[SubQueryResult]) -> str:
        """Generate a summary of the retrieval process."""
        parts = []
        for result in sub_query_results:
            num_chunks = len(result.retrieved_chunks)
            parts.append(
                f"Sub-query '{result.sub_query.text}' retrieved {num_chunks} chunk(s)"
            )
        return "; ".join(parts)