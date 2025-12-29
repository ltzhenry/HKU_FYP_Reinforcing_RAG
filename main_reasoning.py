"""
Runnable example of the reasoning RAG pipeline.
"""

from rag_baseline.embeddings import Embedder
from rag_baseline.generator import Generator
from rag_baseline.ingestion import Document
from rag_baseline.reasoning_pipeline import ReasoningRagPipeline


def main() -> None:
    # Extended document set for testing multi-hop reasoning
    documents = [
        Document(
            doc_id="1",
            text="Python is a popular programming language for data science and machine learning."
        ),
        Document(
            doc_id="2",
            text="FAISS is a library developed by Facebook AI Research for efficient similarity search over vectors."
        ),
        Document(
            doc_id="3",
            text="Sentence Transformers produce dense vector embeddings for text using transformer models."
        ),
        Document(
            doc_id="4",
            text="Vector embeddings are numerical representations of text that capture semantic meaning."
        ),
        Document(
            doc_id="5",
            text="Similarity search finds the most similar items in a database using distance metrics like L2 or cosine similarity."
        ),
        Document(
            doc_id="6",
            text="Facebook AI Research, also known as FAIR, focuses on advancing artificial intelligence through open research."
        ),
        Document(
            doc_id="7",
            text="Transformer models like BERT and GPT have revolutionized natural language processing tasks."
        ),
    ]

    # Initialize pipeline
    embedder = Embedder()
    generator = Generator()
    pipeline = ReasoningRagPipeline(embedder, generator, top_k=3)

    # Index documents
    print("Indexing documents...")
    pipeline.index(documents, max_tokens=80, overlap=20)
    print(f"Indexed {len(documents)} documents\n")

    # Test with different types of questions
    questions = [
        # Simple question - should use direct retrieval
        "What is FAISS?",

        # Complex question - should trigger decomposition
        "What library helps with similarity search and who developed it, and what technology does it use?",
    ]

    for i, question in enumerate(questions, 1):
        print("=" * 80)
        print(f"QUESTION {i}: {question}")
        print("=" * 80)

        answer, analysis, sub_queries, evidence = pipeline.answer_with_reasoning(question)

        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print('='*80)
        print(f"\nAnswer: {answer}")
        print(f"\nEvidence Summary: {evidence.summary}")
        print(f"Confidence: {evidence.confidence_score:.2f}")

        print(f"\nTop Retrieved Chunks:")
        for idx, chunk in enumerate(evidence.all_chunks[:5], 1):
            print(f"  {idx}. [score={chunk.score:.4f}] {chunk.text[:100]}...")

        print("\n")


if __name__ == "__main__":
    main()