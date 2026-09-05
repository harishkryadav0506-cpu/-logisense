"""
Tests for the AI agent pipeline.

Tests individual agents and the orchestrator end-to-end.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTrackerAgent:
    """Test the tracker agent."""

    def test_tracker_finds_order(self):
        """Test tracker agent with a valid order ID."""
        from agents.tracker_agent import tracker_agent

        state = {
            "order_id": "ORD-00001",
            "complaint_text": "Test complaint",
            "agent_trace": [],
        }

        result = tracker_agent(state)
        assert result["order_info"]["found"] is True
        assert len(result["agent_trace"]) > 0
        assert "TrackerAgent" in result["agent_trace"][0]

    def test_tracker_handles_missing_order(self):
        """Test tracker agent with an invalid order ID."""
        from agents.tracker_agent import tracker_agent

        state = {
            "order_id": "ORD-99999",
            "complaint_text": "Test complaint",
            "agent_trace": [],
        }

        result = tracker_agent(state)
        assert result["order_info"]["found"] is False

    def test_tracker_preserves_state(self):
        """Test that tracker agent preserves existing state fields."""
        from agents.tracker_agent import tracker_agent

        state = {
            "order_id": "ORD-00001",
            "complaint_text": "Test complaint",
            "agent_trace": [],
            "extra_field": "should_be_preserved",
        }

        result = tracker_agent(state)
        assert result.get("extra_field") == "should_be_preserved"


class TestSentimentAgent:
    """Test the sentiment agent."""

    def test_sentiment_low_severity(self):
        """Test classification of a low severity complaint."""
        from agents.sentiment_agent import sentiment_agent

        state = {
            "complaint_text": "My order arrived a day late but everything is fine.",
            "agent_trace": [],
        }

        result = sentiment_agent(state)
        assert "severity" in result
        assert result["severity"] in ("low", "medium", "high")
        assert "severity_confidence" in result
        assert 0 <= result["severity_confidence"] <= 1

    def test_sentiment_high_severity(self):
        """Test classification of a high severity complaint."""
        from agents.sentiment_agent import sentiment_agent

        state = {
            "complaint_text": (
                "URGENT: My order NEVER arrived and I was charged TWICE! "
                "This is FRAUD! I DEMAND immediate refund NOW! "
                "I will take LEGAL action!"
            ),
            "agent_trace": [],
        }

        result = sentiment_agent(state)
        assert result["severity"] == "high"

    def test_sentiment_adds_trace(self):
        """Test that sentiment agent adds trace entry."""
        from agents.sentiment_agent import sentiment_agent

        state = {
            "complaint_text": "Test complaint",
            "agent_trace": ["Previous trace"],
        }

        result = sentiment_agent(state)
        assert len(result["agent_trace"]) > 1
        assert "SentimentAgent" in result["agent_trace"][-1]

    def test_sentiment_reports_method(self):
        """Test that sentiment agent reports which method it used."""
        from agents.sentiment_agent import sentiment_agent

        state = {
            "complaint_text": "Test complaint text",
            "agent_trace": [],
        }

        result = sentiment_agent(state)
        assert result["severity_method"] in ("bert_finetuned", "keyword_fallback", "keyword_classifier")


class TestRAGAgent:
    """Test the RAG agent."""

    def test_rag_agent_returns_context(self):
        """Test that RAG agent returns policy context."""
        from agents.rag_agent import rag_agent

        state = {
            "complaint_text": "My order is delayed, can I get a refund?",
            "order_info": {"status": "delayed", "delay_reason": "weather", "found": True},
            "agent_trace": [],
        }

        result = rag_agent(state)
        assert "policy_context" in result
        assert isinstance(result["policy_context"], str)
        assert len(result["policy_context"]) > 0

    def test_rag_agent_adds_trace(self):
        """Test that RAG agent adds trace entry."""
        from agents.rag_agent import rag_agent

        state = {
            "complaint_text": "Test query",
            "order_info": {"status": "delivered", "found": True},
            "agent_trace": [],
        }

        result = rag_agent(state)
        assert len(result["agent_trace"]) > 0
        assert "RAGAgent" in result["agent_trace"][-1]


class TestResolverAgent:
    """Test the resolver agent."""

    def test_resolver_refund_decision(self):
        """Test resolver with high severity delayed order (should refund)."""
        from agents.resolver_agent import resolver_agent

        state = {
            "order_id": "ORD-00001",
            "complaint_text": "Order delayed 10 days, want refund",
            "order_info": {
                "order_id": "ORD-00001",
                "customer_name": "Test User",
                "status": "delayed",
                "carrier": "FedEx",
                "delay_reason": "weather",
                "found": True,
            },
            "policy_context": "Orders delayed more than 7 days get full refund.",
            "severity": "high",
            "agent_trace": [],
        }

        result = resolver_agent(state)
        assert "resolution_action" in result
        assert result["resolution_action"] in ("refund", "reschedule", "escalate")
        assert "email_draft" in result
        assert result["email_draft"] is not None

    def test_resolver_generates_email(self):
        """Test that resolver always generates an email draft."""
        from agents.resolver_agent import resolver_agent

        state = {
            "order_id": "ORD-00001",
            "complaint_text": "Minor issue",
            "order_info": {
                "order_id": "ORD-00001",
                "customer_name": "Test User",
                "status": "delivered",
                "found": True,
            },
            "policy_context": "Standard policy applies.",
            "severity": "low",
            "agent_trace": [],
        }

        result = resolver_agent(state)
        assert result["email_draft"] is not None
        assert "email_body" in result["email_draft"]
        assert "subject" in result["email_draft"]


class TestOrchestrator:
    """Test the orchestrator end-to-end."""

    def test_orchestrator_valid_order(self):
        """Test full pipeline with a valid order."""
        from agents.orchestrator import resolve_complaint

        result = resolve_complaint(
            order_id="ORD-00001",
            complaint_text="My order is delayed and I want to know what's happening.",
        )

        assert "resolution_action" in result
        assert result["resolution_action"] in ("refund", "reschedule", "escalate")
        assert "severity" in result
        assert "email_draft" in result
        assert "agent_trace" in result
        assert len(result["agent_trace"]) >= 3  # At least tracker + sentiment + resolver

    def test_orchestrator_invalid_order(self):
        """Test full pipeline with an invalid order."""
        from agents.orchestrator import resolve_complaint

        result = resolve_complaint(
            order_id="ORD-99999",
            complaint_text="Where is my order?",
        )

        # Should handle gracefully via error handler
        assert "error" in result or result.get("resolution_action") == "escalate"

    def test_orchestrator_returns_all_state(self):
        """Test that orchestrator returns complete state."""
        from agents.orchestrator import resolve_complaint

        result = resolve_complaint(
            order_id="ORD-00010",
            complaint_text="Order arrived damaged. Need replacement.",
        )

        assert "order_id" in result
        assert "complaint_text" in result
        assert "agent_trace" in result
        assert isinstance(result["agent_trace"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
