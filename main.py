"""
Runnable example of the minimal RAG pipeline.
"""

from rag_baseline.embeddings import Embedder
from rag_baseline.generator import Generator
from rag_baseline.ingestion import Document
from rag_baseline.pipeline import RagPipeline


def main() -> None:
    documents = [
        Document(doc_id="1", text="Python is a popular programming language for data science."),
        Document(doc_id="2", text="FAISS is a library for efficient similarity search over vectors."),
        Document(doc_id="3", text="Sentence Transformers produce dense vector embeddings for text."),
    ]

    embedder = Embedder()
    generator = Generator()
    pipeline = RagPipeline(embedder, generator, top_k=3)
    pipeline.index(documents, max_tokens=80, overlap=20)

    user_question = "What library helps with similarity search over embeddings?"
    answer, retrieved = pipeline.answer(user_question)

    print("Question:", user_question)
    print("Answer:", answer)
    print("\nRetrieved chunks:")
    for chunk in retrieved:
        print(f"- score={chunk.score:.4f} text={chunk.text[:80]}...")


if __name__ == "__main__":
    main()


