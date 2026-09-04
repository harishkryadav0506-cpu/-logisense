"""
Tracker Agent — Fetches order status and details.

Uses the order_db tool to look up order information
and adds it to the shared agent state.
"""

import logging
from typing import Any

from tools.order_db import get_order_status

logger = logging.getLogger(__name__)


def tracker_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Fetch order status and details using the order_db tool.

    Reads 'order_id' from the state and populates 'order_info'
    with the full order details.

    Args:
        state: The shared agent state dict containing at minimum:
            - order_id (str): The order ID to look up.

    Returns:
        Updated state dict with 'order_info' and 'agent_trace' appended.
    """
    order_id = state.get("order_id", "")
    logger.info(f"TrackerAgent: Looking up order {order_id}")

    # Fetch order details
    order_info = get_order_status(order_id)

    # Build trace entry
    if order_info.get("found"):
        trace_msg = (
            f"TrackerAgent: Found order {order_id} — "
            f"Status: {order_info['status']}, "
            f"Carrier: {order_info['carrier']}, "
            f"Product: {order_info['product']}"
        )
        if order_info.get("delay_reason"):
            trace_msg += f", Delay reason: {order_info['delay_reason']}"
    else:
        trace_msg = f"TrackerAgent: Order {order_id} not found — {order_info.get('error', 'Unknown error')}"

    logger.info(trace_msg)

    # Update state
    agent_trace = state.get("agent_trace", [])
    agent_trace.append(trace_msg)

    return {
        **state,
        "order_info": order_info,
        "agent_trace": agent_trace,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state = {"order_id": "ORD-00001", "complaint_text": "My order is delayed"}
    result = tracker_agent(test_state)
    print(f"\nOrder Info: {result['order_info']}")
    print(f"Trace: {result['agent_trace']}")
