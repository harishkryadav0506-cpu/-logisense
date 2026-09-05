"""
Tests for the RAG pipeline components.

Tests embedding model loading (when installed locally)
and lightweight keyword retriever functionality.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestEmbeddings:
    """Test the embedding model and functions (local ML validation)."""

    def test_embedding_model_loads(self):
        """Test that the embedding model loads successfully when torch/sentence-transformers are installed."""
        try:
            import sentence_transformers
            import torch
        except ImportError:
            pytest.skip("Local ML libraries (torch, sentence-transformers) omitted in lightweight cloud mode.")

        from rag.embeddings import get_embedding_model
        model = get_embedding_model()
        assert model is not None

    def test_embed_single_text(self):
        """Test embedding a single text string when installed locally."""
        try:
            import sentence_transformers
            import torch
        except ImportError:
            pytest.skip("Local ML libraries omitted in lightweight cloud mode.")

        from rag.embeddings import embed_text
        embedding = embed_text("Test sentence for embedding")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_multiple_texts(self):
        """Test embedding multiple text strings when installed locally."""
        try:
            import sentence_transformers
            import torch
        except ImportError:
            pytest.skip("Local ML libraries omitted in lightweight cloud mode.")

        from rag.embeddings import embed_texts
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = embed_texts(texts)
        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)

    def test_embedding_dimension(self):
        """Test that embedding dimension matches expected MiniLM size (384)."""
        try:
            import sentence_transformers
            import torch
        except ImportError:
            pytest.skip("Local ML libraries omitted in lightweight cloud mode.")

        from rag.embeddings import embed_text
        embedding = embed_text("Test dimension")
        assert len(embedding) == 384

    def test_singleton_pattern(self):
        """Test that the model uses singleton pattern (same instance)."""
        try:
            import sentence_transformers
            import torch
        except ImportError:
            pytest.skip("Local ML libraries omitted in lightweight cloud mode.")

        from rag.embeddings import get_embedding_model
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        assert model1 is model2


class TestRetriever:
    """Test the lightweight keyword RAG retriever."""

    def test_retrieve_returns_results(self):
        """Test that retriever returns results for a policy query."""
        from rag.retriever import retrieve

        results = retrieve("refund policy for delayed orders", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_retrieve_result_structure(self):
        """Test that retrieved results have the expected structure."""
        from rag.retriever import retrieve

        results = retrieve("return window for electronics", top_k=1)
        assert len(results) >= 1

        result = results[0]
        assert "text" in result
        assert "source" in result
        assert "page" in result
        assert "score" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    def test_retrieve_as_context(self):
        """Test formatted context output."""
        from rag.retriever import retrieve_as_context

        context = retrieve_as_context("delivery delay compensation", top_k=2)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Source:" in context or "Policy" in context

    def test_refund_query_returns_relevant_text(self):
        """Test: query about refund after 5 days delay returns relevant text."""
        from rag.retriever import retrieve

        results = retrieve("Is order eligible for refund after 5 days delay?", top_k=3)
        assert len(results) > 0

        # Check that at least one result mentions refund or delay
        all_text = " ".join(r["text"].lower() for r in results)
        assert "refund" in all_text or "delay" in all_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
