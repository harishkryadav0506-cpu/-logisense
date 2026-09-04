"""
Orchestrator — LangGraph StateGraph connecting all agents.

Defines the multi-agent pipeline:
    tracker → rag → sentiment → resolver

Uses LangGraph's StateGraph for structured agent orchestration
with typed state management.
"""

import logging
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
    agent_trace: list[str]
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
        return dict(result)

    except Exception as e:
        logger.error(f"Pipeline error for {order_id}: {e}")
        return {
            **initial_state,
            "error": str(e),
            "resolution_action": "escalate",
            "resolution_reasoning": f"Pipeline error: {str(e)}",
            "agent_trace": initial_state.get("agent_trace", []) + [f"Pipeline Error: {str(e)}"],
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
