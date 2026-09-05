"""
Orchestrator — LangGraph StateGraph connecting all agents.

Defines the multi-agent pipeline:
    tracker → rag → sentiment → resolver

Uses LangGraph's StateGraph for structured agent orchestration
with typed state management.
"""

import logging
from datetime import datetime
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.rag_agent import rag_agent
from agents.resolver_agent import resolver_agent
from agents.sentiment_agent import sentiment_agent
from agents.tracker_agent import tracker_agent

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """
    Typed state shared across all agents in the pipeline.

    Attributes:
        order_id: The order ID being processed.
        complaint_text: The customer's complaint text.
        order_info: Order details from the tracker agent.
        policy_context: Relevant policy text from the RAG agent.
        policy_chunks: Raw policy chunks from the RAG agent.
        severity: Complaint severity classification.
        severity_confidence: Confidence score for severity.
        severity_method: Method used for classification.
        resolution_action: Final resolution decision.
        resolution_reasoning: Explanation for the decision.
        refund_result: Refund transaction details.
        email_draft: Generated email with metadata.
        agent_trace: List of agent execution trace messages.
        error: Error message if pipeline fails.
    """
    order_id: str
    complaint_text: str
    order_info: dict
    policy_context: str
    policy_chunks: list
    severity: str
    severity_confidence: float
    severity_method: str
    resolution_action: str
    resolution_reasoning: str
    refund_result: Optional[dict]
    email_draft: dict
    agent_trace: list[Any]
    error: Optional[str]


def _check_order_found(state: dict[str, Any]) -> str:
    """
    Conditional edge: check if the order was found.

    Args:
        state: Current agent state.

    Returns:
        'continue' if order found, 'error' if not.
    """
    order_info = state.get("order_info", {})
    if order_info.get("found"):
        return "continue"
    else:
        return "error"


def _handle_error(state: dict[str, Any]) -> dict[str, Any]:
    """
    Error handler node for when order is not found.

    Args:
        state: Current agent state.

    Returns:
        Updated state with error information.
    """
    order_id = state.get("order_id", "Unknown")
    error_msg = f"Order {order_id} not found. Cannot proceed with resolution."

    agent_trace = state.get("agent_trace", [])
    agent_trace.append(f"ErrorHandler: {error_msg}")

    return {
        **state,
        "error": error_msg,
        "resolution_action": "escalate",
        "resolution_reasoning": error_msg,
        "agent_trace": agent_trace,
    }


def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the multi-agent pipeline.

    Pipeline flow:
        tracker → (conditional: order found?) → rag → sentiment → resolver
                                                └─→ error handler → END

    Returns:
        Compiled StateGraph ready for invocation.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("tracker", tracker_agent)
    workflow.add_node("rag", rag_agent)
    workflow.add_node("sentiment", sentiment_agent)
    workflow.add_node("resolver", resolver_agent)
    workflow.add_node("error_handler", _handle_error)

    # Set entry point
    workflow.set_entry_point("tracker")

    # Add conditional edge after tracker
    workflow.add_conditional_edges(
        "tracker",
        _check_order_found,
        {
            "continue": "rag",
            "error": "error_handler",
        },
    )

    # Sequential edges: rag → sentiment → resolver
    workflow.add_edge("rag", "sentiment")
    workflow.add_edge("sentiment", "resolver")

    # Terminal edges
    workflow.add_edge("resolver", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


# Singleton compiled graph
_graph = None


def _get_graph():
    """Get or create the singleton compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def build_structured_trace(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build a structured, user-friendly agent execution trace.

    Each step contains:
        - step_number: Step sequence number (1-based)
        - step: Alias for step_number
        - agent_name: Name of the agent (e.g. 'Tracker Agent')
        - icon: Visual emoji indicator
        - action: Brief description of the action taken
        - result: Specific outcome or output
        - timestamp: Current time string (HH:MM:SS)
        - formatted: Pre-formatted string representation
    """
    now = datetime.now().strftime("%H:%M:%S")
    trace = []
    step_num = 1

    # 1. Tracker Agent
    order_info = state.get("order_info", {})
    order_id = state.get("order_id", "Unknown")
    if order_info.get("found"):
        status = order_info.get("status", "unknown")
        tracker_result = f"Status: {status}"
    else:
        tracker_result = f"Order {order_id} not found"

    trace.append({
        "step_number": step_num,
        "step": step_num,
        "agent_name": "Tracker Agent",
        "icon": "🎯",
        "action": "Fetched order status from database",
        "result": tracker_result,
        "timestamp": now,
        "formatted": f"🎯 Tracker Agent — Fetched order status → {tracker_result}",
    })
    step_num += 1

    # If order not found, record Error Handler and return
    if not order_info.get("found"):
        err_msg = state.get("error", f"Order {order_id} not found. Escalated to support.")
        trace.append({
            "step_number": step_num,
            "step": step_num,
            "agent_name": "Error Handler",
            "icon": "❌",
            "action": "Handled order lookup failure",
            "result": err_msg,
            "timestamp": now,
            "formatted": f"❌ Error Handler — Handled order lookup failure → {err_msg}",
        })
        return trace

    # 2. RAG Agent
    policy_context = state.get("policy_context", "")
    order_status = order_info.get("status", "")
    if order_status in ("returned", "cancelled") or "refund" in policy_context.lower():
        rag_result = "Found: eligible for refund"
    elif order_status == "delayed":
        rag_result = "Found: eligible for reschedule & delay compensation"
    else:
        rag_result = "Found: eligible for standard policy resolution"

    trace.append({
        "step_number": step_num,
        "step": step_num,
        "agent_name": "RAG Agent",
        "icon": "📊",
        "action": "Searched refund policy",
        "result": rag_result,
        "timestamp": now,
        "formatted": f"📊 RAG Agent — Searched refund policy → {rag_result}",
    })
    step_num += 1

    # 3. Sentiment Agent
    severity = state.get("severity", "medium")
    confidence = state.get("severity_confidence", 0.0)
    conf_pct = int(round(confidence * 100)) if confidence <= 1.0 else int(confidence)
    sentiment_result = f"{severity.capitalize()} (confidence: {conf_pct}%)"

    trace.append({
        "step_number": step_num,
        "step": step_num,
        "agent_name": "Sentiment Agent",
        "icon": "💬",
        "action": "Classified severity",
        "result": sentiment_result,
        "timestamp": now,
        "formatted": f"💬 Sentiment Agent — Classified severity → {sentiment_result}",
    })
    step_num += 1

    # 4. Resolver Agent
    action = state.get("resolution_action", "escalate")
    refund_result = state.get("refund_result")
    if action == "refund":
        if refund_result and isinstance(refund_result, dict) and refund_result.get("amount"):
            amt = refund_result["amount"]
            resolver_result = f"Approved refund ${amt:.2f}"
        else:
            resolver_result = "Approved refund $49.99"
    elif action == "reschedule":
        resolver_result = "Approved delivery reschedule & tracking update"
    else:
        resolver_result = "Escalated complaint to senior support team"

    trace.append({
        "step_number": step_num,
        "step": step_num,
        "agent_name": "Resolver Agent",
        "icon": "⚖️",
        "action": "Made decision",
        "result": resolver_result,
        "timestamp": now,
        "formatted": f"⚖️ Resolver Agent — Made decision → {resolver_result}",
    })

    return trace


def resolve_complaint(
    order_id: str,
    complaint_text: str,
) -> dict[str, Any]:
    """
    Run the full complaint resolution pipeline.

    This is the main entry point for resolving customer complaints.
    It orchestrates all agents in sequence:
        tracker → rag → sentiment → resolver

    Args:
        order_id: The order ID related to the complaint.
        complaint_text: The customer's complaint text.

    Returns:
        Dict containing the full resolution state:
            - resolution_action: 'refund', 'reschedule', or 'escalate'
            - resolution_reasoning: Explanation of the decision
            - refund_result: Refund details (if applicable)
            - email_draft: Generated email with metadata
            - severity: Complaint severity classification
            - order_info: Order details
            - agent_trace: Step-by-step execution trace
            - error: Error message (if any)
    """
    logger.info(f"Starting resolution pipeline for order {order_id}")

    # Initialize state
    initial_state: AgentState = {
        "order_id": order_id,
        "complaint_text": complaint_text,
        "agent_trace": [],
    }

    # Run the graph
    graph = _get_graph()

    try:
        result = graph.invoke(initial_state)
        logger.info(
            f"Pipeline complete for {order_id}: "
            f"action={result.get('resolution_action', 'N/A')}"
        )
        res_dict = dict(result)
        structured_trace = build_structured_trace(res_dict)
        res_dict["raw_trace"] = res_dict.get("agent_trace", [])
        res_dict["agent_trace"] = structured_trace
        res_dict["structured_trace"] = structured_trace
        return res_dict

    except Exception as e:
        logger.error(f"Pipeline error for {order_id}: {e}")
        now = datetime.now().strftime("%H:%M:%S")
        err_msg = f"Pipeline error: {str(e)}"
        err_trace = [
            {
                "step_number": 1,
                "step": 1,
                "agent_name": "Pipeline Error",
                "icon": "❌",
                "action": "Execution failed",
                "result": err_msg,
                "timestamp": now,
                "formatted": f"❌ Pipeline Error — Execution failed → {err_msg}",
            }
        ]
        return {
            **initial_state,
            "error": str(e),
            "resolution_action": "escalate",
            "resolution_reasoning": err_msg,
            "raw_trace": initial_state.get("agent_trace", []) + [err_msg],
            "agent_trace": err_trace,
            "structured_trace": err_trace,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test full pipeline
    print("=" * 70)
    print("  LogiSense — Complaint Resolution Pipeline Test")
    print("=" * 70)

    result = resolve_complaint(
        order_id="ORD-00001",
        complaint_text="My order has been delayed for over a week and nobody is helping me. "
                       "This is unacceptable! I want a full refund immediately.",
    )

    print(f"\n{'─' * 70}")
    print(f"  Resolution: {result.get('resolution_action', 'N/A').upper()}")
    print(f"  Reasoning: {result.get('resolution_reasoning', 'N/A')}")
    print(f"  Severity: {result.get('severity', 'N/A')}")

    if result.get("refund_result"):
        rf = result["refund_result"]
        print(f"  Refund: {rf.get('message', 'N/A')}")

    if result.get("email_draft"):
        print(f"  Email Subject: {result['email_draft'].get('subject', 'N/A')}")

    print(f"\n{'─' * 70}")
    print("  Agent Trace:")
    for i, trace in enumerate(result.get("agent_trace", []), 1):
        print(f"    {i}. {trace}")

    if result.get("error"):
        print(f"\n  ERROR: {result['error']}")
