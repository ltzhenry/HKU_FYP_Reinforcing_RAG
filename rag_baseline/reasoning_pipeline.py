from __future__ import annotations

"""
Reasoning RAG pipeline with question analysis, decomposition, and multi-hop retrieval.
"""

from typing import List, Tuple

from .decomposer import QueryDecomposer, SubQuery
from .embeddings import Embedder
from .evidence_integrator import EvidenceIntegrator, IntegratedEvidence
from .generator import Generator
from .multi_hop_retriever import MultiHopRetriever, SubQueryResult
from .pipeline import RagPipeline
from .reasoning import QuestionAnalyzer, QuestionAnalysis, QuestionComplexity
from .retriever import RetrievedChunk, Retriever


class ReasoningRagPipeline(RagPipeline):
    """
    Extended RAG pipeline with reasoning capabilities:
    - Question complexity analysis
    - Query decomposition for complex questions
    - Multi-hop retrieval
    - Evidence integration
    """

    def __init__(self, embedder: Embedder, generator: Generator, top_k: int = 5):
        super().__init__(embedder, generator, top_k)
        self.analyzer = QuestionAnalyzer()
        self.decomposer = QueryDecomposer()
        self.evidence_integrator = EvidenceIntegrator()

    def answer_with_reasoning(
        self, query: str
    ) -> Tuple[str, QuestionAnalysis, List[SubQuery], IntegratedEvidence]:
        """
        Answer with full reasoning pipeline.

        Returns:
            - answer: Generated answer text
            - analysis: Question complexity analysis
            - sub_queries: List of decomposed sub-queries (empty for simple questions)
            - evidence: Integrated evidence
        """
        if not self.retriever:
            raise RuntimeError("The pipeline has not been indexed yet. Call index() first.")

        # Step 1: Analyze question complexity
        print(f"\n[Step 1] Analyzing question complexity...")
        analysis = self.analyzer.analyze(query)
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Reasoning type: {analysis.reasoning_type}")
        print(f"  Needs decomposition: {analysis.needs_decomposition}")
        print(f"  Explanation: {analysis.explanation}")

        # Step 2: Strategy selection
        if analysis.complexity == QuestionComplexity.SIMPLE:
            print(f"\n[Step 2] Using direct retrieval (simple question)")
            # Direct retrieval for simple questions
            retrieved = self.retriever.retrieve(query)
            evidence = IntegratedEvidence(
                all_chunks=retrieved,
                summary="Direct retrieval for simple question",
                confidence_score=0.8
            )
            sub_queries = []
        else:
            print(f"\n[Step 2] Using multi-hop reasoning (complex question)")
            # Step 3: Decompose query
            print(f"\n[Step 3] Decomposing query into sub-queries...")
            sub_queries = self.decomposer.decompose(query)
            print(f"  Generated {len(sub_queries)} sub-queries:")
            for sq in sub_queries:
                deps = ", ".join(sq.dependencies) if sq.dependencies else "none"
                print(f"    - {sq.query_id}: {sq.text}")
                print(f"      Dependencies: {deps}")
                print(f"      Reasoning: {sq.reasoning}")

            # Step 4: Multi-hop retrieval
            print(f"\n[Step 4] Executing multi-hop retrieval...")
            multi_hop = MultiHopRetriever(self.retriever)
            sub_query_results = multi_hop.retrieve_multi_hop(sub_queries)

            for i, result in enumerate(sub_query_results, 1):
                print(f"  Sub-query {i}: Retrieved {len(result.retrieved_chunks)} chunks")

            # Step 5: Evidence integration
            print(f"\n[Step 5] Integrating evidence...")
            evidence = self.evidence_integrator.integrate(sub_query_results)
            print(f"  Total unique chunks: {len(evidence.all_chunks)}")
            print(f"  Confidence score: {evidence.confidence_score:.2f}")

        # Step 6: Generate answer
        print(f"\n[Step 6] Generating final answer...")
        answer = self.generator.generate(query, evidence.all_chunks)

        return answer, analysis, sub_queries, evidence