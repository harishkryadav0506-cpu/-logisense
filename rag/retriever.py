"""
Retriever module for querying policy documents using keyword/text search.

Lightweight, zero-memory-overhead policy search optimized for Render Free tier (<400MB RAM).
Extracts text from data/policies/*.pdf and performs ranked keyword/TF matching
without requiring ChromaDB, PyTorch, or SentenceTransformers.
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
COLLECTION_NAME = "logisense_policies"

# Standard verified policy context as immediate fallback
FALLBACK_POLICY_CONTEXT = (
    "Policy guidelines:\n"
    "1. Delay Policy: Orders delayed by more than 7 days are eligible for a full refund. "
    "Orders delayed by 3-7 days are eligible for a 25% partial refund or expedited reschedule.\n"
    "2. Damaged/Defective Policy: Customers reporting damaged, defective, or incorrect items "
    "are eligible for an immediate full replacement or refund upon return initiation.\n"
    "3. Return Window: Standard returns accepted within 30 days of confirmed delivery."
)


def is_vector_store_available(persist_directory: Any = None) -> bool:
    """
    Check if policy documents are available on disk.
    Maintained for backward compatibility.
    """
    return POLICIES_DIR.exists() and any(POLICIES_DIR.glob("*.pdf"))


@lru_cache(maxsize=1)
def load_policy_chunks() -> list[dict]:
    """
    Load and parse all PDF files from data/policies/ into searchable chunks.
    Cached in memory as lightweight text (~50KB).

    Returns:
        List of dicts with 'text', 'source', 'page', and 'chunk_id'.
    """
    chunks: list[dict] = []
    chunk_id = 0

    if not POLICIES_DIR.exists():
        logger.warning(f"Policies directory not found at {POLICIES_DIR}")
        return chunks

    pdf_files = sorted(POLICIES_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {POLICIES_DIR}")
        return chunks

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed. Falling back to built-in policy context.")
        return chunks

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, 1):
                raw_text = page.extract_text() or ""
                paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                if not paragraphs and raw_text.strip():
                    paragraphs = [raw_text.strip()]

                for para in paragraphs:
                    # Break very long paragraphs into ~400-char segments
                    words = para.split()
                    for i in range(0, max(1, len(words)), 60):
                        segment = " ".join(words[i:i + 60])
                        if len(segment.strip()) > 20:
                            chunks.append({
                                "text": segment.strip(),
                                "source": pdf_path.name,
                                "page": page_num,
                                "chunk_id": chunk_id,
                            })
                            chunk_id += 1
        except Exception as e:
            logger.error(f"Error reading policy PDF {pdf_path.name}: {e}")

    logger.info(f"Loaded {len(chunks)} policy chunks from {len(pdf_files)} PDFs")
    return chunks


def _score_chunk(query_tokens: set[str], chunk_text: str) -> float:
    """Score a chunk based on keyword token overlap and frequency."""
    text_lower = chunk_text.lower()
    score = 0.0
    for token in query_tokens:
        if token in text_lower:
            count = len(re.findall(rf"\b{re.escape(token)}\b", text_lower))
            score += 1.0 + 0.5 * count
    return score


def retrieve(
    query: str,
    top_k: int = 3,
    score_threshold: Optional[float] = None,
) -> list[dict]:
    """
    Search policy chunks using keyword similarity matching.

    Args:
        query: The search query text.
        top_k: Number of top results to return.
        score_threshold: Unused compatibility parameter.

    Returns:
        List of dicts with 'text', 'source', 'page', 'score', and 'chunk_id'.
    """
    all_chunks = load_policy_chunks()
    if not all_chunks:
        return []

    # Tokenize query into meaningful search words (3+ chars, lower)
    stop_words = {
        "the", "and", "for", "with", "this", "that", "from", "are", "was",
        "order", "have", "has", "can", "get", "what", "how", "why", "when"
    }
    raw_tokens = re.findall(r"\b\w{3,}\b", query.lower())
    query_tokens = {t for t in raw_tokens if t not in stop_words}
    if not query_tokens:
        query_tokens = set(raw_tokens)

    scored_chunks = []
    for chunk in all_chunks:
        score = _score_chunk(query_tokens, chunk["text"])
        if score > 0:
            scored_chunks.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 4),
            })

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    results = scored_chunks[:top_k]

    # If no keyword matched, return top general chunks
    if not results and all_chunks:
        results = [
            {
                "text": c["text"],
                "source": c["source"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "score": 0.5,
            }
            for c in all_chunks[:top_k]
        ]

    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
    return results


def retrieve_as_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant chunks and format them as context string.

    Args:
        query: Search query text.
        top_k: Number of top results.

    Returns:
        Formatted context string.
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return FALLBACK_POLICY_CONTEXT

    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def get_vector_store(*args, **kwargs) -> Any:
    """Compatibility stub for get_vector_store."""
    class DummyVectorStore:
        def similarity_search(self, query: str, k: int = 3):
            chunks = retrieve(query, top_k=k)
            class Doc:
                def __init__(self, text, meta):
                    self.page_content = text
                    self.metadata = meta
            return [Doc(c["text"], {"source": c["source"], "page": c["page"]}) for c in chunks]

        def similarity_search_with_score(self, query: str, k: int = 3):
            chunks = retrieve(query, top_k=k)
            class Doc:
                def __init__(self, text, meta):
                    self.page_content = text
                    self.metadata = meta
            return [(Doc(c["text"], {"source": c["source"], "page": c["page"]}), c["score"]) for c in chunks]

    return DummyVectorStore()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = retrieve("delay compensation and refund policy", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['source']}] {r['text'][:100]}... (Score: {r['score']})")
