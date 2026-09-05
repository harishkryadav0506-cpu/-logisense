"""
Dataset preparation module for complaint severity classification.

Loads reviews.csv, applies data augmentation and class balancing,
tokenizes complaint text with BertTokenizer, and creates
train/val/test PyTorch Datasets for fine-tuning.
"""

import logging
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
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
MAX_LENGTH = 64  # Optimal for 1-2 sentence complaints (max ~35 tokens), 2.5x faster on CPU

# ─────────────────────────────────────────────────
# Augmented Synthetic Complaint Templates
# ─────────────────────────────────────────────────

AUGMENTED_TEMPLATES_LOW = [
    "My order arrived a day late but everything looks fine. Just wanted to let you know.",
    "The packaging was slightly creased upon delivery, but the contents are in good shape.",
    "Tracking for my parcel was not updating for 24 hours. The package arrived safely though.",
    "Received the item and the color shade is slightly different than shown online. Not a big deal.",
    "The courier left the package on the front porch without ringing the bell. Please notify drivers.",
    "Order was scheduled for yesterday afternoon and arrived this morning instead. Minor delay.",
    "There is a small spelling mistake on the printed invoice. Could you send an updated receipt?",
    "Delivery agent was somewhat abrupt, but the item itself is in perfect working order.",
    "Got my delivery a few hours outside the estimated delivery window. Just a quick heads up.",
    "Everything works great, but the quick start manual was missing from the box. Can you email a PDF?",
    "Delivery was left at my neighbor's door by mistake, but I retrieved it easily.",
    "Product is fine, just took one extra business day to get delivered. Thanks.",
    "The exterior cardboard box was slightly bent, but the product bubble wrap kept it secure.",
    "Accessory cables were placed in a separate internal pouch, initially thought they were missing.",
    "Delivery took 4 days instead of 3 days. Item works as expected.",
    "Polite feedback: please send SMS alert 30 minutes before delivery next time.",
    "Package arrived in good condition, only feedback is the driver left it near the garage.",
    "Received order today. Everything is intact, just a 1-day transit delay.",
    "Good product overall. Just noting that the outer tape was slightly loose.",
    "Delivery date moved back by one day without notification, but product is satisfactory.",
]

AUGMENTED_TEMPLATES_MEDIUM = [
    "My shipment is delayed by 3 days now and tracking hasn't updated. Need this expedited.",
    "I received the completely wrong product. Ordered headphones but received a mouse. Please exchange.",
    "The package has been stuck at the regional sorting facility for 5 days. Very frustrating.",
    "The item stopped functioning after 2 days of normal use. I need a replacement right away.",
    "Tracking says delivered yesterday but no package was left at my address. Please check with carrier.",
    "Still waiting on my refund after returning the item over a week ago. When will it process?",
    "Order arrived with missing components. The power adapter and charging cord were not in the box.",
    "The build quality does not match the product listing specifications. Requesting return and exchange.",
    "Delivery was rescheduled twice without my consent. Very inconvenient for my schedule.",
    "Paid extra for expedited 2-day delivery but it was sent via standard ground shipping. Need shipping refund.",
    "There is a visible scratch across the screen of this supposedly brand-new device.",
    "My order was split into two consignments without notice and the main item is still missing.",
    "Product arrived with the factory security seal broken. I suspect this is a returned unit.",
    "Need to modify or cancel my order before dispatch, but customer service has not replied.",
    "The dimensions of the product delivered do not match the size chart published on the website.",
    "Carrier attempted delivery while I was away after giving the wrong time estimate. Need redelivery.",
    "Item arrived partially defective - only one side of the earbuds is working. Need a swap.",
    "Package was delivered 4 days late and missed the birthday event I bought it for.",
    "Customer service rep promised a replacement 3 days ago but I received no confirmation email.",
    "Delivered to the wrong apartment block. Thankfully my neighbor brought it over. Need better courier notes.",
]

AUGMENTED_TEMPLATES_HIGH = [
    "URGENT: Order never arrived and I was charged twice on my credit card! Immediate refund demanded!",
    "Order is 12 days overdue and customer support refuses to give answers. This is unacceptable! Escalate NOW!",
    "Received completely shattered and destroyed product! Box was crushed. I DEMAND an immediate full refund!",
    "I have contacted support 5 times regarding my lost package. No updates. I want my money back IMMEDIATELY!",
    "FRAUD ALERT: Order was marked delivered to wrong address and signed by an unknown person. Investigate!",
    "Product is dangerously DEFECTIVE and started smoking when plugged in! Severe safety hazard!",
    "Been given the runaround for 3 weeks regarding my return refund. Filing a formal consumer forum complaint!",
    "SCAM WARNING: Item delivered is an obvious counterfeit knockoff, not authentic! Full refund and compensation now!",
    "EXTREMELY ANGRY: Wrong item, damaged box, 10 days late! I demand to speak to a senior manager right now!",
    "Carrier officially lost my $600 package and support is doing nothing. Taking immediate legal action!",
    "Sold me a refurbished and heavily scratched unit as brand new! This is outright fraud! Refund now!",
    "High-value order vanished in transit. Tracking dead for 2 weeks. Customer service hung up on me!",
    "URGENT: Ordered critical medical supplies that are 10 days late! Inexcusable negligence! Full refund now!",
    "Unauthorized transaction and fraudulent duplicate billing on my statement for this order! Reversal needed!",
    "Absolute nightmare experience. Package stolen, zero response from carrier, support ghosted me. Escalate!",
    "Product caused an electrical short circuit and damaged my wall outlet! Immediate supervisor review needed!",
    "You have held my money for a month without delivering the order! I demand an instant full refund!",
    "Furious! Sent me a broken return item with someone else's return label inside! Disgraceful service!",
    "Package lost, customer service was rude and hung up on me. Reporting to consumer protection authorities!",
    "Item never delivered, no refund issued after 14 business days. Immediate resolution or credit card chargeback!",
]


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


def augment_and_balance_data(
    df: pd.DataFrame,
    target_samples_per_class: int = 220,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Augment and balance the complaints dataset so each class has at least target_samples_per_class.

    Args:
        df: Input DataFrame with 'complaint_text' and 'severity' columns.
        target_samples_per_class: Desired number of samples for each severity class.
        random_state: Random seed for reproducibility.

    Returns:
        Balanced and augmented DataFrame.
    """
    rng = random.Random(random_state)
    templates = {
        "low": AUGMENTED_TEMPLATES_LOW,
        "medium": AUGMENTED_TEMPLATES_MEDIUM,
        "high": AUGMENTED_TEMPLATES_HIGH,
    }

    balanced_dfs = []

    for severity, label in SEVERITY_TO_LABEL.items():
        subset = df[df["severity"] == severity].copy()
        current_count = len(subset)

        if current_count < target_samples_per_class:
            needed = target_samples_per_class - current_count
            synthetic_rows = []
            class_templates = templates[severity]

            for i in range(needed):
                base_text = rng.choice(class_templates)
                # Apply light linguistic variations
                oid = f"ORD-{rng.randint(1000, 9999)}"
                if "{oid}" in base_text:
                    aug_text = base_text.replace("{oid}", oid)
                elif rng.random() < 0.4:
                    aug_text = f"Regarding {oid}: {base_text}"
                else:
                    aug_text = base_text

                synthetic_rows.append({
                    "complaint_text": aug_text,
                    "severity": severity,
                    "label": label,
                })

            aug_df = pd.DataFrame(synthetic_rows)
            subset = pd.concat([subset, aug_df], ignore_index=True)
        elif current_count > target_samples_per_class:
            subset = subset.sample(n=target_samples_per_class, random_state=random_state)

        subset["label"] = label
        balanced_dfs.append(subset)

    result_df = pd.concat(balanced_dfs, ignore_index=True)
    result_df = result_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    logger.info(
        f"Augmented & balanced dataset: {len(result_df)} samples "
        f"({result_df['severity'].value_counts().to_dict()})"
    )
    return result_df


def load_and_prepare_data(
    reviews_path: Optional[Path] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    return_val: bool = False,
    target_samples_per_class: int = 220,
) -> Union[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load reviews.csv, balance and augment data, and split into train/val/test DataFrames.

    Args:
        reviews_path: Path to reviews.csv file.
        test_size: Fraction of data for test set.
        val_size: Fraction of data for validation set (used when return_val=True).
        random_state: Random seed for reproducibility.
        return_val: If True, returns (train_df, val_df, test_df); else (train_df, test_df).
        target_samples_per_class: Minimum balanced samples per class (default 220).

    Returns:
        Tuple of DataFrames (train_df, test_df) or (train_df, val_df, test_df).
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
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Augment and balance classes
    df = augment_and_balance_data(
        df,
        target_samples_per_class=target_samples_per_class,
        random_state=random_state,
    )

    if return_val and val_size > 0:
        # First split off test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
        # Next split train into train and val
        adjusted_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=train_val_df["label"],
        )

        logger.info(
            f"Dataset split — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        return train_df, val_df, test_df
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
        logger.info(f"Dataset split — Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df


def create_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = MAX_LENGTH,
) -> Union[
    tuple[ComplaintDataset, ComplaintDataset, BertTokenizer],
    tuple[ComplaintDataset, ComplaintDataset, ComplaintDataset, BertTokenizer],
]:
    """
    Create PyTorch Datasets from train/(val)/test DataFrames.

    Args:
        train_df: Training DataFrame with 'complaint_text' and 'label' columns.
        test_df: Test DataFrame with same columns.
        val_df: Optional validation DataFrame.
        model_name: Pretrained BERT model name for tokenizer.
        max_length: Maximum token sequence length.

    Returns:
        (train_dataset, test_dataset, tokenizer) if val_df is None,
        (train_dataset, val_dataset, test_dataset, tokenizer) if val_df is provided.
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

    if val_df is not None:
        val_dataset = ComplaintDataset(
            texts=val_df["complaint_text"].tolist(),
            labels=val_df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        logger.info(
            f"Created datasets — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset, tokenizer

    logger.info(
        f"Created datasets — Train: {len(train_dataset)}, Test: {len(test_dataset)}"
    )
    return train_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    train_df, val_df, test_df = load_and_prepare_data(return_val=True)
    train_ds, val_ds, test_ds, tok = create_datasets(train_df, test_df, val_df=val_df)

    sample = train_ds[0]
    print(f"\nSample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    print(f"Sample label: {sample['labels'].item()} ({LABEL_TO_SEVERITY[sample['labels'].item()]})")
