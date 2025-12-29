from __future__ import annotations

"""
Multi-hop retriever that executes sub-queries in order and aggregates results.
"""

from dataclasses import dataclass
from typing import Dict, List

from .decomposer import SubQuery
from .retriever import Retriever, RetrievedChunk


@dataclass
class SubQueryResult:
    """Result from executing a single sub-query."""
    sub_query: SubQuery
    retrieved_chunks: List[RetrievedChunk]
    context_from_dependencies: str  # Aggregated context from dependent sub-queries


class MultiHopRetriever:
    """
    Executes sub-queries in dependency order and aggregates evidence.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def retrieve_multi_hop(self, sub_queries: List[SubQuery]) -> List[SubQueryResult]:
        """
        Execute sub-queries respecting their dependencies.
        """
        # Sort sub-queries by dependency order
        ordered_subqueries = self._topological_sort(sub_queries)
        
        results: Dict[str, SubQueryResult] = {}
        
        for sub_query in ordered_subqueries:
            # Gather context from dependencies
            dependency_context = self._gather_dependency_context(sub_query, results)
            
            # Enhance query with dependency context if available
            enhanced_query = self._enhance_query(sub_query.text, dependency_context)
            
            # Retrieve chunks for this sub-query
            retrieved = self.retriever.retrieve(enhanced_query)
            
            # Store result
            results[sub_query.query_id] = SubQueryResult(
                sub_query=sub_query,
                retrieved_chunks=retrieved,
                context_from_dependencies=dependency_context
            )
        
        return list(results.values())

    def _topological_sort(self, sub_queries: List[SubQuery]) -> List[SubQuery]:
        """
        Sort sub-queries so dependencies are executed first.
        Simple implementation: independent queries first, then others.
        """
        independent = [sq for sq in sub_queries if not sq.dependencies]
        dependent = [sq for sq in sub_queries if sq.dependencies]
        return independent + dependent

    def _gather_dependency_context(
        self, sub_query: SubQuery, completed_results: Dict[str, SubQueryResult]
    ) -> str:
        """Gather relevant context from completed dependency sub-queries."""
        if not sub_query.dependencies:
            return ""
        
        context_parts = []
        for dep_id in sub_query.dependencies:
            if dep_id in completed_results:
                result = completed_results[dep_id]
                # Take top chunk from each dependency
                if result.retrieved_chunks:
                    top_chunk = result.retrieved_chunks[0]
                    context_parts.append(f"From {dep_id}: {top_chunk.text}")
        
        return "\n".join(context_parts)

    def _enhance_query(self, original_query: str, dependency_context: str) -> str:
        """Enhance query with context from dependencies."""
        if not dependency_context:
            return original_query
        
        return f"{original_query}\n\nRelevant context:\n{dependency_context}"