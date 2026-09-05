"""
Evaluation script for the fine-tuned BERT complaint severity classifier.

Loads the saved model and evaluates on the test set, printing:
- Accuracy, Precision, Recall, F1 scores
- Confusion matrix
- Sample predictions

Usage:
    python -m finetuning.evaluate
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import BertForSequenceClassification, BertTokenizer

from finetuning.dataset import (
    LABEL_TO_SEVERITY,
    MAX_LENGTH,
    load_and_prepare_data,
)

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "finetuning" / "saved_model"


def load_model(
    model_dir: str | None = None,
) -> tuple[BertForSequenceClassification, BertTokenizer, torch.device]:
    """
    Load the fine-tuned model and tokenizer.

    Args:
        model_dir: Directory containing the saved model.

    Returns:
        Tuple of (model, tokenizer, device).

    Raises:
        FileNotFoundError: If the saved model directory doesn't exist.
    """
    if model_dir is None:
        model_dir = str(SAVED_MODEL_DIR)

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_dir}. "
            "Run 'python -m finetuning.train' first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {model_dir} on {device}")

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict(
    texts: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[int], list[list[float]]]:
    """
    Run predictions on a list of texts.

    Args:
        texts: List of complaint texts.
        model: Fine-tuned BERT model.
        tokenizer: BERT tokenizer.
        device: Device to run inference on.
        batch_size: Inference batch size.

    Returns:
        Tuple of (predicted_labels, probabilities).
    """
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encoding = tokenizer(
            batch_texts,
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
            logits = outputs.logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()

        all_preds.extend(preds)
        all_probs.extend(probs.tolist())

    return all_preds, all_probs


def evaluate(model_dir: str | None = None) -> dict:
    """
    Full evaluation pipeline: load model, predict on test set, print metrics.

    Args:
        model_dir: Directory containing the saved model.

    Returns:
        Dict containing all evaluation metrics.
    """
    # Load model
    model, tokenizer, device = load_model(model_dir)

    # Load test data
    _, test_df = load_and_prepare_data()
    texts = test_df["complaint_text"].tolist()
    true_labels = test_df["label"].tolist()

    # Run predictions
    logger.info(f"Running predictions on {len(texts)} test samples...")
    pred_labels, pred_probs = predict(texts, model, tokenizer, device)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")

    # Calculate confidence scores
    confidences = [max(probs) * 100 for probs in pred_probs]
    avg_confidence = float(np.mean(confidences))
    high_conf_pct = float(np.mean([c >= 80.0 for c in confidences]) * 100)

    # Print results
    label_names = [LABEL_TO_SEVERITY[i] for i in range(3)]

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy:           {accuracy:.4f}")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  Avg Confidence:     {avg_confidence:.2f}%")
    print(f"  High Conf (>=80%):  {high_conf_pct:.1f}%")

    # Classification report
    print(f"\n{'-' * 60}")
    print("  CLASSIFICATION REPORT")
    print(f"{'-' * 60}")
    print(classification_report(true_labels, pred_labels, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"{'-' * 60}")
    print("  CONFUSION MATRIX")
    print(f"{'-' * 60}")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {name}" for name in label_names],
        columns=[f"Pred: {name}" for name in label_names],
    )
    print(f"\n{cm_df.to_string()}\n")

    # Sample predictions
    print(f"{'-' * 60}")
    print("  SAMPLE PREDICTIONS (10 examples)")
    print(f"{'-' * 60}")

    sample_indices = np.random.choice(len(texts), min(10, len(texts)), replace=False)
    for idx in sample_indices:
        true_sev = LABEL_TO_SEVERITY[true_labels[idx]]
        pred_sev = LABEL_TO_SEVERITY[pred_labels[idx]]
        correct = "[OK]" if true_labels[idx] == pred_labels[idx] else "[X]"
        confidence = max(pred_probs[idx]) * 100

        print(f"\n  {correct} Text: {texts[idx][:100]}...")
        print(f"    Actual: {true_sev} | Predicted: {pred_sev} | Confidence: {confidence:.1f}%")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1,
        "avg_confidence": avg_confidence,
        "high_confidence_pct": high_conf_pct,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    np.random.seed(42)
    metrics = evaluate()
