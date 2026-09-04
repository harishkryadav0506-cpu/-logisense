"""
Embeddings module for the RAG pipeline.

Loads and manages the HuggingFace sentence-transformers/all-MiniLM-L6-v2
embedding model with a singleton pattern to avoid redundant reloading.
"""

import logging
from functools import lru_cache
from typing import Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding model.

    Uses lru_cache to implement singleton pattern — the model is loaded
    only once and reused across all subsequent calls.

    Args:
        model_name: The HuggingFace model identifier.
        device: Device to run the model on ('cpu', 'cuda', etc.).
                If None, auto-detects GPU availability.

    Returns:
        HuggingFaceEmbeddings instance ready for encoding text.
    """
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info(f"Loading embedding model: {model_name} on {device}")

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 64}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    logger.info("Embedding model loaded successfully")
    return embeddings


def embed_text(text: str, model_name: str = DEFAULT_MODEL_NAME) -> list[float]:
    """
    Embed a single text string and return the embedding vector.

    Args:
        text: The text to embed.
        model_name: The HuggingFace model identifier.

    Returns:
        List of floats representing the embedding vector.
    """
    model = get_embedding_model(model_name)
    return model.embed_query(text)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> list[list[float]]:
    """
    Embed multiple text strings and return their embedding vectors.

    Args:
        texts: List of texts to embed.
        model_name: The HuggingFace model identifier.

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    model = get_embedding_model()
    test_embedding = embed_text("Test embedding for refund policy")
    print(f"Model loaded. Embedding dimension: {len(test_embedding)}")
    print(f"First 5 values: {test_embedding[:5]}")
