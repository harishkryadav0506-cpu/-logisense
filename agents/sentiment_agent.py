"""
Sentiment Agent — Classifies complaint severity.

Uses the fine-tuned BERT model to classify complaint text
into low/medium/high severity categories. Falls back to
keyword-based classification if the model is not available.
"""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "finetuning" / "saved_model"

# Keywords for fallback classification
HIGH_SEVERITY_KEYWORDS = [
    "urgent", "immediately", "fraud", "scam", "legal", "lawsuit", "sue",
    "furious", "outraged", "unacceptable", "demand", "compensation",
    "safety", "hazard", "defective", "counterfeit", "never arrived",
    "charged twice", "escalate", "manager", "consumer forum", "lost",
    "disappeared", "vanished", "stolen",
]

MEDIUM_SEVERITY_KEYWORDS = [
    "wrong item", "exchange", "replacement", "frustrated", "waiting",
    "stuck", "delayed", "missing parts", "doesn't match", "broken",
    "not working", "refund", "rescheduled", "incorrect", "disappointed",
    "inconvenient", "scratch", "damaged",
]


# Singleton model cache
_MODEL = None
_TOKENIZER = None
_DEVICE = None


def reset_model_cache() -> None:
    """Reset the singleton model cache so newly saved checkpoints are reloaded."""
    global _MODEL, _TOKENIZER, _DEVICE
    _MODEL = None
    _TOKENIZER = None
    _DEVICE = None


def _get_model_and_tokenizer():
    """Load and cache the fine-tuned BERT model and tokenizer."""
    global _MODEL, _TOKENIZER, _DEVICE

    if not SAVED_MODEL_DIR.exists() or not (SAVED_MODEL_DIR / "config.json").exists():
        raise FileNotFoundError("Fine-tuned model not found in saved_model directory")

    if _MODEL is None or _TOKENIZER is None:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading fine-tuned BERT model on {_DEVICE}...")
        _MODEL = BertForSequenceClassification.from_pretrained(str(SAVED_MODEL_DIR))
        _TOKENIZER = BertTokenizer.from_pretrained(str(SAVED_MODEL_DIR))
        _MODEL.to(_DEVICE)
        _MODEL.eval()

    return _MODEL, _TOKENIZER, _DEVICE


def _classify_with_model(complaint_text: str) -> tuple[str, float]:
    """
    Classify complaint severity using the fine-tuned BERT model.

    Args:
        complaint_text: The complaint text to classify.

    Returns:
        Tuple of (severity_label, confidence_score).

    Raises:
        FileNotFoundError: If the saved model is not found.
    """
    import torch
    from finetuning.dataset import LABEL_TO_SEVERITY, MAX_LENGTH

    model, tokenizer, device = _get_model_and_tokenizer()

    encoding = tokenizer(
        complaint_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_label].item()

    severity = LABEL_TO_SEVERITY[predicted_label]
    return severity, confidence


def _classify_with_keywords(complaint_text: str) -> tuple[str, float]:
    """
    Fallback keyword-based complaint severity classification.

    Args:
        complaint_text: The complaint text to classify.

    Returns:
        Tuple of (severity_label, confidence_score).
    """
    text_lower = complaint_text.lower()

    # Count keyword matches
    high_matches = sum(1 for kw in HIGH_SEVERITY_KEYWORDS if kw in text_lower)
    medium_matches = sum(1 for kw in MEDIUM_SEVERITY_KEYWORDS if kw in text_lower)

    # Check for shouting (ALL CAPS words)
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", complaint_text))

    # Check for exclamation marks
    exclamation_count = complaint_text.count("!")

    # Score calculation
    high_score = high_matches * 3 + caps_words * 2 + exclamation_count
    medium_score = medium_matches * 2

    if high_score >= 3:
        return "high", min(0.7 + high_score * 0.03, 0.95)
    elif medium_score >= 2 or high_score >= 1:
        return "medium", min(0.6 + medium_score * 0.05, 0.90)
    else:
        return "low", 0.65


def sentiment_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify the severity of a customer complaint.

    Attempts to use the fine-tuned BERT model first, falling back
    to keyword-based classification if the model is unavailable.

    Args:
        state: The shared agent state dict containing:
            - complaint_text (str): The customer's complaint text.

    Returns:
        Updated state dict with:
            - severity (str): 'low', 'medium', or 'high'
            - severity_confidence (float): Confidence score (0-1)
            - severity_method (str): 'bert_finetuned' or 'keyword_fallback'
            - agent_trace: Updated trace list
    """
    complaint_text = state.get("complaint_text", "")
    logger.info(f"SentimentAgent: Classifying complaint severity...")

    # Try BERT model first, fall back to keywords
    try:
        severity, confidence = _classify_with_model(complaint_text)
        method = "bert_finetuned"
        logger.info(f"SentimentAgent: Using fine-tuned BERT model")
    except (FileNotFoundError, Exception) as e:
        logger.info(f"SentimentAgent: BERT model unavailable ({e}), using keyword fallback")
        severity, confidence = _classify_with_keywords(complaint_text)
        method = "keyword_fallback"

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
        state = {"complaint_text": text, "agent_trace": []}
        result = sentiment_agent(state)
        print(f"\nText: {text[:80]}...")
        print(f"Severity: {result['severity']} (confidence: {result['severity_confidence']:.2f})")
        print(f"Method: {result['severity_method']}")
