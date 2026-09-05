"""
Tests for lightweight cloud deployment compatibility (<400MB RAM safe).

Verifies:
- FastAPI /health endpoint returns 200 and healthy components
- FastAPI /resolve endpoint runs end-to-end without heavy ML or 502 crashes
- Sentiment agent keyword classification
- Policy keyword retriever functionality and fallback
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from agents.sentiment_agent import classify_severity, sentiment_agent
from rag.retriever import retrieve, retrieve_as_context

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns HTTP 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["components"]["api"] == "healthy"
    assert data["components"]["orders_db"] == "healthy"
    assert data["components"]["vector_store"] in ("healthy", "fallback_policy_ready")
    assert "keyword_classifier_ready" in data["components"]["bert_model"]


def test_resolve_endpoint_success():
    """Verify /resolve processes a valid complaint successfully without crashing."""
    payload = {
        "order_id": "ORD-00001",
        "complaint_text": "My package is delayed by 5 days and I need a refund immediately.",
    }
    response = client.post("/resolve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORD-00001"
    assert data["resolution_action"] in ("refund", "reschedule", "escalate")
    assert len(data["agent_trace"]) > 0


def test_resolve_endpoint_order_not_found():
    """Verify /resolve handles non-existent order gracefully without 500/502."""
    payload = {
        "order_id": "ORD-99999",
        "complaint_text": "Where is my package? It is late.",
    }
    response = client.post("/resolve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["resolution_action"] == "escalate"


def test_sentiment_classification_levels():
    """Test high, medium, and low severity keyword classification."""
    high_label, high_conf = classify_severity("URGENT: This is unacceptable fraud! Demand refund!")
    assert high_label == "high"
    assert high_conf >= 0.70

    med_label, med_conf = classify_severity("My delivery is delayed by a few days and late.")
    assert med_label == "medium"
    assert med_conf >= 0.65

    low_label, low_conf = classify_severity("Just checking when my package will arrive.")
    assert low_label == "low"


def test_retriever_query():
    """Test policy text retriever returns relevant excerpts."""
    results = retrieve("delay compensation and refund policy", top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]
    assert "source" in results[0]


def test_retriever_context_formatting():
    """Test retrieve_as_context produces non-empty string."""
    context = retrieve_as_context("return window", top_k=2)
    assert isinstance(context, str)
    assert len(context) > 0
