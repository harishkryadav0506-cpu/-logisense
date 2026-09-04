"""
RAG Agent — Searches policy documents for relevant information.

Uses the RAG retriever to find policy excerpts relevant to the
customer complaint and order situation.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def rag_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Search policy documents for relevant information.

    Constructs a contextual query from the complaint text and order info,
    then retrieves relevant policy chunks from the vector store.

    Args:
        state: The shared agent state dict containing:
            - complaint_text (str): The customer's complaint.
            - order_info (dict): Order details from the tracker agent.

    Returns:
        Updated state dict with 'policy_context' and 'agent_trace' appended.
    """
    complaint_text = state.get("complaint_text", "")
    order_info = state.get("order_info", {})

    # Build a contextual query combining complaint and order info
    query_parts = [complaint_text]
    if order_info.get("status"):
        query_parts.append(f"Order status: {order_info['status']}")
    if order_info.get("delay_reason"):
        query_parts.append(f"Delay reason: {order_info['delay_reason']}")

    query = " ".join(query_parts)
    logger.info(f"RAGAgent: Querying policies with: '{query[:100]}...'")

    try:
        from rag.retriever import retrieve, retrieve_as_context

        # Get raw chunks for metadata
        chunks = retrieve(query, top_k=3)

        # Get formatted context for LLM
        policy_context = retrieve_as_context(query, top_k=3)

        # Build trace
        sources = list({chunk["source"] for chunk in chunks})
        trace_msg = (
            f"RAGAgent: Retrieved {len(chunks)} relevant policy chunks "
            f"from {', '.join(sources)}"
        )

        if chunks:
            top_chunk_preview = chunks[0]["text"][:150]
            trace_msg += f" | Top result: '{top_chunk_preview}...'"

    except FileNotFoundError:
        logger.warning("RAGAgent: Vector store not found. Using fallback response.")
        policy_context = (
            "Policy information unavailable. Default policy applies: "
            "Orders delayed by more than 7 days are eligible for full refund. "
            "Orders delayed by 3-7 days are eligible for partial refund (25%). "
            "Returns accepted within 30 days of delivery."
        )
        chunks = []
        trace_msg = "RAGAgent: Vector store not found — using fallback default policy"

    except Exception as e:
        logger.error(f"RAGAgent: Error querying policies: {e}")
        policy_context = "Error retrieving policy information. Applying standard resolution guidelines."
        chunks = []
        trace_msg = f"RAGAgent: Error — {str(e)}"

    logger.info(trace_msg)

    agent_trace = state.get("agent_trace", [])
    agent_trace.append(trace_msg)

    return {
        **state,
        "policy_context": policy_context,
        "policy_chunks": chunks,
        "agent_trace": agent_trace,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state = {
        "order_id": "ORD-00001",
        "complaint_text": "My order is delayed by 5 days, can I get a refund?",
        "order_info": {"status": "delayed", "delay_reason": "weather_disruption", "found": True},
        "agent_trace": [],
    }

    result = rag_agent(test_state)
    print(f"\nPolicy Context:\n{result['policy_context'][:500]}")
    print(f"\nTrace: {result['agent_trace']}")
