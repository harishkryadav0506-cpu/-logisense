"""
Dataset preparation module for complaint severity classification.

Loads reviews.csv, tokenizes complaint text with BertTokenizer,
and creates train/test PyTorch Datasets for fine-tuning.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEWS_PATH = PROJECT_ROOT / "data" / "reviews.csv"

# Label mapping
SEVERITY_TO_LABEL = {"low": 0, "medium": 1, "high": 2}
LABEL_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_LABEL.items()}

# Tokenizer configuration
DEFAULT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


class ComplaintDataset(Dataset):
    """
    PyTorch Dataset for complaint severity classification.

    Each sample contains tokenized complaint text and its severity label.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: BertTokenizer,
        max_length: int = MAX_LENGTH,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            texts: List of complaint text strings.
            labels: List of integer labels (0=low, 1=medium, 2=high).
            tokenizer: BertTokenizer instance.
            max_length: Maximum token sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_and_prepare_data(
    reviews_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load reviews.csv and split into train/test DataFrames.

    Args:
        reviews_path: Path to reviews.csv file.
        test_size: Fraction of data for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df) DataFrames.
    """
    if reviews_path is None:
        reviews_path = REVIEWS_PATH

    logger.info(f"Loading reviews from {reviews_path}")
    df = pd.read_csv(reviews_path)

    # Validate required columns
    required_cols = ["complaint_text", "severity"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map severity to numeric labels
    df["label"] = df["severity"].map(SEVERITY_TO_LABEL)

    # Drop any rows with unknown severity values
    unknown_mask = df["label"].isna()
    if unknown_mask.any():
        logger.warning(f"Dropping {unknown_mask.sum()} rows with unknown severity")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    logger.info(f"Dataset size: {len(df)} samples")
    logger.info(f"Label distribution:\n{df['severity'].value_counts().to_string()}")

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    logger.info(f"Train set: {len(train_df)} samples | Test set: {len(test_df)} samples")
    return train_df, test_df


def create_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = MAX_LENGTH,
) -> tuple[ComplaintDataset, ComplaintDataset, BertTokenizer]:
    """
    Create PyTorch Datasets from train/test DataFrames.

    Args:
        train_df: Training DataFrame with 'complaint_text' and 'label' columns.
        test_df: Test DataFrame with same columns.
        model_name: Pretrained BERT model name for tokenizer.
        max_length: Maximum token sequence length.

    Returns:
        Tuple of (train_dataset, test_dataset, tokenizer).
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = ComplaintDataset(
        texts=train_df["complaint_text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = ComplaintDataset(
        texts=test_df["complaint_text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    logger.info(
        f"Created datasets — Train: {len(train_dataset)}, Test: {len(test_dataset)}"
    )
    return train_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    train_df, test_df = load_and_prepare_data()
    train_dataset, test_dataset, tokenizer = create_datasets(train_df, test_df)

    # Quick verification
    sample = train_dataset[0]
    print(f"\nSample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    print(f"Sample label: {sample['labels'].item()} ({LABEL_TO_SEVERITY[sample['labels'].item()]})")
