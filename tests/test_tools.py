"""
Tests for the tool functions.

Tests order_db, refund_tool, and email_tool.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestOrderDB:
    """Test the order database tool."""

    def test_get_order_valid_id(self):
        """Test fetching a valid order."""
        from tools.order_db import get_order_status

        result = get_order_status("ORD-00001")
        assert result["found"] is True
        assert result["order_id"] == "ORD-00001"
        assert "customer_name" in result
        assert "product" in result
        assert "status" in result

    def test_get_order_invalid_id(self):
        """Test fetching a non-existent order."""
        from tools.order_db import get_order_status

        result = get_order_status("ORD-99999")
        assert result["found"] is False
        assert "error" in result

    def test_get_order_case_insensitive(self):
        """Test case-insensitive order ID matching."""
        from tools.order_db import get_order_status

        result = get_order_status("ord-00001")
        assert result["found"] is True

    def test_get_order_has_all_fields(self):
        """Test that order result contains all expected fields."""
        from tools.order_db import get_order_status

        result = get_order_status("ORD-00001")
        expected_fields = [
            "order_id", "customer_name", "product", "order_date",
            "status", "carrier", "found",
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_search_orders_by_status(self):
        """Test searching orders by status."""
        from tools.order_db import search_orders

        results = search_orders(status="delivered", limit=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        for order in results:
            assert order["status"] == "delivered"


class TestRefundTool:
    """Test the refund processing tool."""

    def test_process_refund_valid(self):
        """Test processing a valid refund."""
        from tools.refund_tool import process_refund

        result = process_refund("ORD-00001", 49.99, "Test refund")
        assert result["status"] == "approved"
        assert result["transaction_id"] is not None
        assert result["amount"] == 49.99
        assert "REF-" in result["transaction_id"]

    def test_process_refund_negative_amount(self):
        """Test refund with negative amount is rejected."""
        from tools.refund_tool import process_refund

        result = process_refund("ORD-00001", -10.00)
        assert result["status"] == "rejected"

    def test_process_refund_zero_amount(self):
        """Test refund with zero amount is rejected."""
        from tools.refund_tool import process_refund

        result = process_refund("ORD-00001", 0)
        assert result["status"] == "rejected"

    def test_process_refund_high_amount(self):
        """Test refund exceeding limit requires review."""
        from tools.refund_tool import process_refund

        result = process_refund("ORD-00001", 15000.00)
        assert result["status"] == "pending_review"

    def test_process_refund_empty_order_id(self):
        """Test refund with empty order ID is rejected."""
        from tools.refund_tool import process_refund

        result = process_refund("", 49.99)
        assert result["status"] == "rejected"

    def test_refund_creates_log(self):
        """Test that refund creates a log entry."""
        from tools.refund_tool import REFUND_LOG_PATH, process_refund

        process_refund("ORD-TEST-LOG", 9.99, "Log test")
        assert REFUND_LOG_PATH.exists()


class TestEmailTool:
    """Test the email drafting tool."""

    def test_draft_email_refund(self):
        """Test drafting a refund email."""
        from tools.email_tool import draft_email

        email = draft_email(
            customer_name="Test User",
            order_id="ORD-00001",
            resolution="Full refund processed",
            resolution_type="refund",
        )
        assert isinstance(email, str)
        assert "Test User" in email
        assert "ORD-00001" in email
        assert "refund" in email.lower()

    def test_draft_email_reschedule(self):
        """Test drafting a reschedule email."""
        from tools.email_tool import draft_email

        email = draft_email(
            customer_name="Test User",
            order_id="ORD-00002",
            resolution="Delivery rescheduled",
            resolution_type="reschedule",
        )
        assert "Delivery" in email or "delivery" in email
        assert "ORD-00002" in email

    def test_draft_email_escalate(self):
        """Test drafting an escalation email."""
        from tools.email_tool import draft_email

        email = draft_email(
            customer_name="Test User",
            order_id="ORD-00003",
            resolution="Case escalated",
            resolution_type="escalate",
        )
        assert "escalat" in email.lower()

    def test_draft_email_with_metadata(self):
        """Test email draft with metadata."""
        from tools.email_tool import draft_email_with_metadata

        result = draft_email_with_metadata(
            customer_name="Test User",
            order_id="ORD-00001",
            resolution="Test resolution",
            resolution_type="refund",
        )
        assert "email_body" in result
        assert "subject" in result
        assert "to" in result
        assert "reference_id" in result
        assert result["to"] == "Test User"

    def test_draft_email_general_fallback(self):
        """Test that unknown resolution type falls back to general template."""
        from tools.email_tool import draft_email

        email = draft_email(
            customer_name="Test User",
            order_id="ORD-00001",
            resolution="Some resolution",
            resolution_type="unknown_type",
        )
        assert isinstance(email, str)
        assert "Test User" in email


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
