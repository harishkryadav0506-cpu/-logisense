"""
Resolver Agent — Makes final resolution decisions.

Combines outputs from tracker, RAG, and sentiment agents to decide
the appropriate resolution (refund, reschedule, or escalate), then
calls the refund tool and email tool to execute the resolution.
"""

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv

from tools.email_tool import draft_email_with_metadata
from tools.refund_tool import process_refund

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def _decide_resolution_with_llm(state: dict[str, Any]) -> dict[str, str]:
    """
    Use LLM to decide the resolution based on all agent outputs.

    Args:
        state: The shared agent state with all prior agent outputs.

    Returns:
        Dict with 'action' (refund/reschedule/escalate) and 'reasoning'.
    """
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        order_info = state.get("order_info", {})
        policy_context = state.get("policy_context", "No policy information available.")
        severity = state.get("severity", "unknown")
        complaint_text = state.get("complaint_text", "")

        prompt = f"""You are a customer service resolution specialist for an e-commerce company.

Based on the following information, decide the best resolution action.

COMPLAINT: {complaint_text}

ORDER DETAILS:
- Order ID: {order_info.get('order_id', 'Unknown')}
- Status: {order_info.get('status', 'Unknown')}
- Carrier: {order_info.get('carrier', 'Unknown')}
- Delay Reason: {order_info.get('delay_reason', 'None')}
- Delivery Date: {order_info.get('delivery_date', 'N/A')}

COMPLAINT SEVERITY: {severity}

RELEVANT POLICY:
{policy_context}

Based on the above, choose ONE action and provide your reasoning:
1. REFUND - If the customer is eligible for a refund per policy
2. RESCHEDULE - If the delivery can be rescheduled or expedited
3. ESCALATE - If the case requires human intervention

Respond in this exact format:
ACTION: [REFUND/RESCHEDULE/ESCALATE]
AMOUNT: [refund amount if applicable, or N/A]
REASONING: [your reasoning in 2-3 sentences]"""

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse response
        action = "escalate"  # default
        amount = "N/A"
        reasoning = content

        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                action_text = line.split(":", 1)[1].strip().lower()
                if "refund" in action_text:
                    action = "refund"
                elif "reschedule" in action_text:
                    action = "reschedule"
                else:
                    action = "escalate"
            elif line.upper().startswith("AMOUNT:"):
                amount = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return {"action": action, "reasoning": reasoning, "amount": amount}

    except Exception as e:
        logger.warning(f"LLM resolution failed: {e}. Using rule-based fallback.")
        return _decide_resolution_rules(state)


def _decide_resolution_rules(state: dict[str, Any]) -> dict[str, str]:
    """
    Rule-based resolution decision as fallback when LLM is unavailable.

    Args:
        state: The shared agent state.

    Returns:
        Dict with 'action', 'reasoning', and 'amount'.
    """
    order_info = state.get("order_info", {})
    severity = state.get("severity", "medium")
    status = order_info.get("status", "unknown")
    delay_reason = order_info.get("delay_reason", "")

    # Decision rules
    if severity == "high":
        if status in ("delayed", "cancelled", "returned"):
            return {
                "action": "refund",
                "reasoning": (
                    f"High severity complaint with order status '{status}'. "
                    f"Per policy, immediate full refund is warranted."
                ),
                "amount": "49.99",
            }
        else:
            return {
                "action": "escalate",
                "reasoning": (
                    f"High severity complaint requires senior team review. "
                    f"Order status: {status}."
                ),
                "amount": "N/A",
            }

    elif severity == "medium":
        if status == "delayed" and delay_reason:
            return {
                "action": "reschedule",
                "reasoning": (
                    f"Medium severity with delivery delay due to '{delay_reason}'. "
                    f"Rescheduling delivery with expedited shipping."
                ),
                "amount": "N/A",
            }
        elif status in ("cancelled", "returned"):
            return {
                "action": "refund",
                "reasoning": (
                    f"Medium severity complaint for {status} order. "
                    f"Partial refund per policy guidelines."
                ),
                "amount": "25.00",
            }
        else:
            return {
                "action": "reschedule",
                "reasoning": (
                    f"Medium severity issue. Expediting delivery and providing tracking update."
                ),
                "amount": "N/A",
            }

    else:  # low severity
        return {
            "action": "reschedule",
            "reasoning": (
                f"Low severity complaint. Providing updated delivery information "
                f"and apology coupon."
            ),
            "amount": "N/A",
        }


def resolver_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Make final resolution decision and execute it.

    Combines all agent outputs, decides on refund/reschedule/escalate,
    then calls the appropriate tools (refund_tool, email_tool).

    Args:
        state: The shared agent state with outputs from all prior agents.

    Returns:
        Updated state dict with:
            - resolution_action (str): 'refund', 'reschedule', or 'escalate'
            - resolution_reasoning (str): Explanation of the decision
            - refund_result (dict): Refund transaction details (if applicable)
            - email_draft (dict): Generated email with metadata
            - agent_trace: Updated trace list
    """
    logger.info("ResolverAgent: Making resolution decision...")

    order_info = state.get("order_info", {})
    customer_name = order_info.get("customer_name", "Valued Customer")
    order_id = state.get("order_id", "Unknown")

    # Decide resolution
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key != "your_openai_api_key_here":
        decision = _decide_resolution_with_llm(state)
    else:
        logger.info("ResolverAgent: No API key found, using rule-based resolution")
        decision = _decide_resolution_rules(state)

    action = decision["action"]
    reasoning = decision["reasoning"]
    amount_str = decision.get("amount", "N/A")

    logger.info(f"ResolverAgent: Decision = {action.upper()}")

    # Execute refund if needed
    refund_result = None
    if action == "refund":
        try:
            amount = float(amount_str.replace("$", "").replace(",", ""))
        except (ValueError, AttributeError):
            amount = 49.99  # Default refund amount

        refund_result = process_refund(
            order_id=order_id,
            amount=amount,
            notes=reasoning,
        )
        logger.info(f"ResolverAgent: Refund processed — {refund_result.get('transaction_id')}")

    # Draft customer email
    email_result = draft_email_with_metadata(
        customer_name=customer_name,
        order_id=order_id,
        resolution=reasoning,
        resolution_type=action,
    )
    logger.info(f"ResolverAgent: Email drafted — Subject: {email_result['subject']}")

    # Build trace
    trace_parts = [f"ResolverAgent: Decision = {action.upper()} — {reasoning}"]
    if refund_result:
        trace_parts.append(
            f"ResolverAgent: Refund {refund_result['status']} — "
            f"Transaction: {refund_result.get('transaction_id', 'N/A')}"
        )
    trace_parts.append(f"ResolverAgent: Email drafted — {email_result['subject']}")

    agent_trace = state.get("agent_trace", [])
    agent_trace.extend(trace_parts)

    return {
        **state,
        "resolution_action": action,
        "resolution_reasoning": reasoning,
        "refund_result": refund_result,
        "email_draft": email_result,
        "agent_trace": agent_trace,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state = {
        "order_id": "ORD-00042",
        "complaint_text": "My order has been delayed for 5 days and I need a refund.",
        "order_info": {
            "order_id": "ORD-00042",
            "customer_name": "Aarav Sharma",
            "product": "Wireless Bluetooth Headphones",
            "status": "delayed",
            "carrier": "FedEx",
            "delay_reason": "weather_disruption",
            "found": True,
        },
        "policy_context": "Orders delayed by more than 3 days are eligible for partial refund.",
        "severity": "medium",
        "agent_trace": ["TrackerAgent: ...", "RAGAgent: ...", "SentimentAgent: ..."],
    }

    result = resolver_agent(test_state)
    print(f"\nAction: {result['resolution_action']}")
    print(f"Reasoning: {result['resolution_reasoning']}")
    if result.get("refund_result"):
        print(f"Refund: {result['refund_result']['message']}")
    print(f"Email Subject: {result['email_draft']['subject']}")
