"""
Refund processing tool.

Simulates refund processing and logs all refund transactions
to a CSV file for audit trail.
"""

import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFUND_LOG_PATH = PROJECT_ROOT / "data" / "refund_log.csv"

# Refund log columns
REFUND_LOG_COLUMNS = [
    "transaction_id",
    "order_id",
    "amount",
    "status",
    "timestamp",
    "notes",
]


def _ensure_refund_log() -> None:
    """Create the refund log file with headers if it doesn't exist."""
    if not REFUND_LOG_PATH.exists():
        REFUND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REFUND_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REFUND_LOG_COLUMNS)
        logger.info(f"Created refund log at {REFUND_LOG_PATH}")


def process_refund(
    order_id: str,
    amount: float,
    notes: str = "",
) -> dict:
    """
    Process a refund for the given order.

    This is a simulated refund — it logs the transaction to a CSV file
    and returns a confirmation. In production, this would integrate with
    a payment gateway.

    Args:
        order_id: The order ID to refund.
        amount: The refund amount in USD.
        notes: Optional notes about the refund reason.

    Returns:
        Dict containing:
            - transaction_id: Unique refund transaction ID
            - order_id: The refunded order ID
            - amount: Refund amount
            - status: 'approved' or 'rejected'
            - timestamp: ISO timestamp of the refund
            - message: Human-readable status message
    """
    try:
        # Validate inputs
        if not order_id or not order_id.strip():
            return {
                "transaction_id": None,
                "order_id": order_id,
                "amount": amount,
                "status": "rejected",
                "timestamp": datetime.now().isoformat(),
                "message": "Invalid order ID provided.",
            }

        if amount <= 0:
            return {
                "transaction_id": None,
                "order_id": order_id,
                "amount": amount,
                "status": "rejected",
                "timestamp": datetime.now().isoformat(),
                "message": f"Invalid refund amount: ${amount:.2f}. Must be positive.",
            }

        if amount > 10000:
            return {
                "transaction_id": None,
                "order_id": order_id,
                "amount": amount,
                "status": "pending_review",
                "timestamp": datetime.now().isoformat(),
                "message": f"Refund of ${amount:.2f} exceeds $10,000 limit. Requires supervisor review.",
            }

        # Generate transaction ID
        transaction_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().isoformat()

        # Log the refund
        _ensure_refund_log()
        with open(REFUND_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                transaction_id,
                order_id,
                f"{amount:.2f}",
                "approved",
                timestamp,
                notes,
            ])

        result = {
            "transaction_id": transaction_id,
            "order_id": order_id,
            "amount": amount,
            "status": "approved",
            "timestamp": timestamp,
            "message": f"Refund of ${amount:.2f} approved for order {order_id}. "
                       f"Transaction ID: {transaction_id}",
        }

        logger.info(f"Refund processed: {transaction_id} for {order_id} (${amount:.2f})")
        return result

    except Exception as e:
        logger.error(f"Error processing refund for {order_id}: {e}")
        return {
            "transaction_id": None,
            "order_id": order_id,
            "amount": amount,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "message": f"Error processing refund: {str(e)}",
        }


def get_refund_history(order_id: str) -> list[dict]:
    """
    Get refund history for a specific order.

    Args:
        order_id: The order ID to look up.

    Returns:
        List of refund transaction dicts for the order.
    """
    if not REFUND_LOG_PATH.exists():
        return []

    import pandas as pd

    df = pd.read_csv(REFUND_LOG_PATH)
    matches = df[df["order_id"] == order_id]
    return matches.to_dict(orient="records")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test refund processing
    print("Testing process_refund:")
    result = process_refund("ORD-00001", 49.99, "Delayed delivery - customer compensation")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nTesting invalid refund:")
    result = process_refund("ORD-00001", -10)
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
