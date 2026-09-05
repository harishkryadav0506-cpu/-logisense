"""
Fine-tuning script for BERT-based complaint severity classifier.

Fine-tunes bert-base-uncased on the complaint severity classification
task (low/medium/high) using the Hugging Face Trainer API.

Colab-compatible with GPU auto-detection.

Usage:
    python -m finetuning.train
    python -m finetuning.train --epochs 5 --batch-size 16
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from finetuning.dataset import (
    LABEL_TO_SEVERITY,
    SEVERITY_TO_LABEL,
    create_datasets,
    load_and_prepare_data,
)

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "finetuning" / "saved_model"

# Training defaults
DEFAULT_EPOCHS = 6
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """
    Compute evaluation metrics for the Trainer.

    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer.

    Returns:
        Dict with accuracy, f1_weighted, and f1_macro scores.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    output_dir: str | None = None,
) -> tuple[Trainer, dict]:
    """
    Fine-tune BERT on complaint severity classification.

    Args:
        epochs: Number of training epochs.
        batch_size: Training and evaluation batch size.
        learning_rate: Learning rate for AdamW optimizer.
        output_dir: Directory to save the fine-tuned model.

    Returns:
        Tuple of (trained Trainer instance, training metrics dict).
    """
    if output_dir is None:
        output_dir = str(SAVED_MODEL_DIR)

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    if device == "cpu":
        logger.warning(
            "GPU not detected. Training on CPU with optimized sequence length (MAX_LENGTH=64). "
            f"Estimated runtime: ~3-5 minutes for {epochs} epochs."
        )
        # Reduce batch size for CPU
        batch_size = min(batch_size, 8)

    # Prepare data with train / validation / test splits
    logger.info("Preparing datasets with train/val/test splits and data augmentation...")
    train_df, val_df, test_df = load_and_prepare_data(return_val=True)
    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        train_df, test_df, val_df=val_df
    )

    # Load pretrained model
    num_labels = len(SEVERITY_TO_LABEL)
    logger.info(f"Loading bert-base-uncased with {num_labels} labels...")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        id2label=LABEL_TO_SEVERITY,
        label2id=SEVERITY_TO_LABEL,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=25,
        save_total_limit=2,
        report_to="none",  # Disable W&B / MLflow
        fp16=torch.cuda.is_available(),  # Mixed precision on GPU
        dataloader_num_workers=0,  # Safe default for Windows
    )

    # Initialize Trainer with validation set for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info(f"Starting training for {epochs} epochs...")
    train_result = trainer.train()

    # Save best model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log training metrics
    metrics = train_result.metrics
    logger.info(f"Training complete! Metrics: {metrics}")

    # Evaluate on final test set
    logger.info("Evaluating best model on held-out test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    logger.info(f"Held-out test set metrics: {test_metrics}")

    return trainer, {**metrics, **test_metrics}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune BERT for complaint severity")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    trainer, metrics = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
