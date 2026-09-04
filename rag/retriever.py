"""
Retriever module for querying the ChromaDB vector store.

Provides functions to search policy documents by semantic similarity
and return relevant chunks with metadata.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"
COLLECTION_NAME = "logisense_policies"


def get_vector_store(
    persist_directory: str | None = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Load the persisted ChromaDB vector store.

    Args:
        persist_directory: Path to the persisted vector store.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Chroma vector store instance.

    Raises:
        FileNotFoundError: If the vector store directory doesn't exist.
    """
    if persist_directory is None:
        persist_directory = str(VECTOR_STORE_DIR)

    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. "
            "Run 'python -m rag.ingest' first to create it."
        )

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    logger.info(f"Loaded vector store from {persist_directory}")
    return vector_store


def retrieve(
    query: str,
    top_k: int = 3,
    score_threshold: Optional[float] = None,
) -> list[dict]:
    """
    Query the vector store and return the top-k relevant policy chunks.

    Args:
        query: The search query text.
        top_k: Number of top results to return.
        score_threshold: Minimum similarity score (0-1). Results below
                        this threshold are filtered out. None = no filter.

    Returns:
        List of dicts, each containing:
            - text: The chunk content
            - source: Source document filename
            - page: Page number in the source document
            - score: Similarity score (lower = more similar for L2 distance)
    """
    vector_store = get_vector_store()

    logger.info(f"Querying vector store: '{query[:80]}...' (top_k={top_k})")

    # Use similarity_search_with_score for ranked results
    results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)

    retrieved_chunks = []
    for doc, score in results_with_scores:
        if score_threshold is not None and score > score_threshold:
            continue  # ChromaDB uses L2 distance; lower = better

        retrieved_chunks.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0),
            "chunk_id": doc.metadata.get("chunk_id", -1),
            "score": round(float(score), 4),
        })

    logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
    return retrieved_chunks


def retrieve_as_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant chunks and format them as a context string
    suitable for LLM prompts.

    Args:
        query: The search query text.
        top_k: Number of top results to return.

    Returns:
        Formatted string containing all relevant policy excerpts.
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return "No relevant policy information found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test queries
    test_queries = [
        "Is order eligible for refund after 5 days delay?",
        "What is the return window for electronics?",
        "What happens after 3 failed delivery attempts?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = retrieve(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (score: {result['score']}):")
            print(f"  Source: {result['source']}, Page {result['page']}")
            print(f"  Text: {result['text'][:200]}...")
