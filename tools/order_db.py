"""
Order database tool for querying order information.

Provides function-calling interface to look up order status
and details from the orders.csv dataset.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORDERS_PATH = PROJECT_ROOT / "data" / "orders.csv"


@lru_cache(maxsize=1)
def _load_orders(orders_path: str | None = None) -> pd.DataFrame:
    """
    Load and cache the orders DataFrame.

    Args:
        orders_path: Path to orders.csv file.

    Returns:
        DataFrame with order data.
    """
    if orders_path is None:
        orders_path = str(ORDERS_PATH)

    logger.info(f"Loading orders from {orders_path}")
    df = pd.read_csv(orders_path)
    logger.info(f"Loaded {len(df)} orders")
    return df


def get_order_status(order_id: str) -> dict:
    """
    Look up the status and details of an order by its ID.

    This function is designed to be used as a tool by LangChain agents.

    Args:
        order_id: The order ID to look up (e.g., 'ORD-00001').

    Returns:
        Dict containing order details:
            - order_id: The order ID
            - customer_name: Customer's full name
            - product: Product name
            - order_date: Date the order was placed
            - status: Current order status
            - carrier: Shipping carrier
            - delivery_date: Actual or expected delivery date
            - delay_reason: Reason for delay (if any)
            - found: Boolean indicating if the order was found

        If the order is not found, returns:
            - order_id: The queried order ID
            - found: False
            - error: Error message
    """
    try:
        df = _load_orders()

        # Normalize order_id for case-insensitive matching
        order_id_upper = order_id.strip().upper()
        match = df[df["order_id"].str.upper() == order_id_upper]

        if match.empty:
            logger.warning(f"Order not found: {order_id}")
            return {
                "order_id": order_id,
                "found": False,
                "error": f"No order found with ID '{order_id}'",
            }

        # Take the first match (order IDs should be unique)
        order = match.iloc[0]

        result = {
            "order_id": str(order["order_id"]),
            "customer_name": str(order["customer_name"]),
            "product": str(order["product"]),
            "order_date": str(order["order_date"]),
            "status": str(order["status"]),
            "carrier": str(order["carrier"]),
            "delivery_date": str(order.get("delivery_date", "")),
            "delay_reason": str(order.get("delay_reason", "")),
            "found": True,
        }

        logger.info(f"Found order {order_id}: status={result['status']}")
        return result

    except Exception as e:
        logger.error(f"Error querying order {order_id}: {e}")
        return {
            "order_id": order_id,
            "found": False,
            "error": f"Error querying order: {str(e)}",
        }


def search_orders(
    customer_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Search orders by customer name or status.

    Args:
        customer_name: Partial customer name to search for.
        status: Order status filter.
        limit: Maximum number of results to return.

    Returns:
        List of order dicts matching the search criteria.
    """
    df = _load_orders()

    if customer_name:
        df = df[df["customer_name"].str.contains(customer_name, case=False, na=False)]

    if status:
        df = df[df["status"].str.lower() == status.lower()]

    results = df.head(limit).to_dict(orient="records")
    logger.info(f"Search returned {len(results)} results")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a known order
    print("Testing get_order_status:")
    result = get_order_status("ORD-00001")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Test with unknown order
    print("\nTesting with unknown order:")
    result = get_order_status("ORD-99999")
    print(f"  Found: {result['found']}")
    print(f"  Error: {result.get('error', 'N/A')}")
