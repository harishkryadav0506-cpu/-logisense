"""
Sentiment Agent — Classifies complaint severity using keyword-based analysis.

Lightweight, high-speed sentiment analysis optimized for memory-constrained
environments (Render Free tier, <400MB RAM).
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Keywords for severity classification
HIGH_SEVERITY_KEYWORDS = [
    "urgent", "immediately", "unacceptable", "fraud", "scam", "legal",
    "lawsuit", "sue", "furious", "outraged", "demand", "compensation",
    "safety", "hazard", "defective", "counterfeit", "never arrived",
    "charged twice", "escalate", "manager", "consumer forum", "lost",
    "disappeared", "vanished", "stolen",
]

MEDIUM_SEVERITY_KEYWORDS = [
    "delay", "delayed", "late", "wrong item", "exchange", "replacement",
    "frustrated", "waiting", "stuck", "missing parts", "doesn't match",
    "broken", "not working", "refund", "rescheduled", "incorrect",
    "disappointed", "inconvenient", "scratch", "damaged",
]


def classify_severity(complaint_text: str) -> tuple[str, float]:
    """
    Classify complaint severity using rule/keyword heuristics.

    Args:
        complaint_text: The customer's complaint text.

    Returns:
        Tuple of (severity_label, confidence_score).
    """
    text_lower = complaint_text.lower()

    # Count keyword matches
    high_matches = sum(1 for kw in HIGH_SEVERITY_KEYWORDS if kw in text_lower)
    medium_matches = sum(1 for kw in MEDIUM_SEVERITY_KEYWORDS if kw in text_lower)

    # Check for shouting (ALL CAPS words with 3+ chars)
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", complaint_text))

    # Check for exclamation marks
    exclamation_count = complaint_text.count("!")

    high_score = high_matches * 3 + caps_words * 2 + exclamation_count
    medium_score = medium_matches * 2

    if high_score >= 3:
        confidence = min(0.70 + high_score * 0.03, 0.95)
        return "high", round(confidence, 2)
    elif medium_score >= 2 or high_score >= 1:
        confidence = min(0.65 + medium_score * 0.04, 0.90)
        return "medium", round(confidence, 2)
    else:
        return "low", 0.70


def sentiment_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify the severity of a customer complaint.

    Args:
        state: The shared agent state dict containing:
            - complaint_text (str): The customer's complaint text.

    Returns:
        Updated state dict with:
            - severity (str): 'low', 'medium', or 'high'
            - severity_confidence (float): Confidence score (0-1)
            - severity_method (str): 'keyword_classifier'
            - agent_trace: Updated trace list
    """
    complaint_text = state.get("complaint_text", "")
    logger.info("SentimentAgent: Classifying complaint severity...")

    severity, confidence = classify_severity(complaint_text)
    method = "keyword_classifier"

    trace_msg = (
        f"SentimentAgent: Classified severity as '{severity}' "
        f"(confidence: {confidence:.2f}, method: {method})"
    )
    logger.info(trace_msg)

    agent_trace = state.get("agent_trace", [])
    agent_trace.append(trace_msg)

    return {
        **state,
        "severity": severity,
        "severity_confidence": confidence,
        "severity_method": method,
        "agent_trace": agent_trace,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_cases = [
        "My order arrived a day late but everything is fine.",
        "Order is delayed by 3 days and I need it for an event. Very frustrating!",
        "URGENT: Order never arrived and I was charged twice! DEMAND immediate refund NOW!",
    ]

    for text in test_cases:
        result = sentiment_agent({"complaint_text": text, "agent_trace": []})
        print(f"Severity: {result['severity']} ({result['severity_confidence']})")
