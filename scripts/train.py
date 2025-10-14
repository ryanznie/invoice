"""
Training script for LayoutLMv3 Invoice Number Extraction with LoRA
Uses Focal Loss to handle extreme class imbalance
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict
from dataclasses import dataclass, field
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.data.data_collator import default_data_collator
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ============================================================================
# FOCAL LOSS
# ============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where N = batch_size * seq_length, C = num_classes
            targets: (N,)
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Cross entropy loss
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Get class probabilities for targets
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Mask for valid (non-ignored) targets
        mask = targets != self.ignore_index

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if isinstance(self.alpha, (list, tuple)):
            # Per-class alpha
            alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            # Single alpha value
            focal_loss = self.alpha * focal_weight * ce_loss

        # Apply mask and reduction
        focal_loss = focal_loss * mask.float()

        if self.reduction == "mean":
            return focal_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    # Paths
    train_data_file: str = "data/SROIE2019/train/train.json"
    test_data_file: str = "data/SROIE2019/test/test.json"
    output_dir: str = "./layoutlmv3-invoice-lora"

    # Model
    model_name: str = "microsoft/layoutlmv3-base"
    max_length: int = 512

    # Labels
    label2id: Dict[str, int] = None
    id2label: Dict[int, str] = None

    # Training
    num_epochs: int = 20
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "query",
            "value",
            "output.dense",
            "intermediate.dense",
            "output.dense",
        ]
    )

    # Focal Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Data split
    val_ratio: float = 0.15
    random_seed: int = 42

    def __post_init__(self):
        if self.label2id is None:
            self.label2id = {"O": 0, "B-INVOICE_ID": 1, "I-INVOICE_ID": 2}
        if self.id2label is None:
            self.id2label = {v: k for k, v in self.label2id.items()}


# ============================================================================
# DATASET
# ============================================================================


class InvoiceDataset(Dataset):
    """Dataset for invoice NER with LayoutLMv3"""

    def __init__(
        self,
        data: List[Dict],
        processor: LayoutLMv3Processor,
        max_length: int = 512,
        label2id: Dict[str, int] = None,
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.label2id = label2id or {"O": 0, "B-INVOICE_ID": 1, "I-INVOICE_ID": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = item.get("image_path") or item.get("file", "").replace(".jpg", "")
        if not image_path.endswith(".jpg"):
            image_path = f"data/SROIE2019/train/img/{item['file']}"

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Error loading image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")

        # Get words, boxes, and labels
        words = item["words"]
        boxes = item.get("bboxes") or item.get("boxes")
        word_labels = item.get("labels", [])

        # Convert string labels to integers if needed
        if word_labels and isinstance(word_labels[0], str):
            word_labels = [self.label2id[label] for label in word_labels]

        # Process with LayoutLMv3Processor
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels if word_labels else None,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding


# ============================================================================
# CUSTOM TRAINER WITH FOCAL LOSS
# ============================================================================


class FocalLossTrainer(Trainer):
    """Custom Trainer that uses Focal Loss"""

    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, ignore_index=-100
        )
        print(f"✓ Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Flatten for loss computation
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# METRICS
# ============================================================================


def compute_metrics(eval_pred):
    """Compute entity-level metrics using seqeval"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = []
    true_labels = []

    label_list = ["O", "B-INVOICE_ID", "I-INVOICE_ID"]

    for prediction, label in zip(predictions, labels):
        true_preds = []
        true_labs = []
        for pred, lab in zip(prediction, label):
            if lab != -100:
                true_preds.append(label_list[pred])
                true_labs.append(label_list[lab])
        true_predictions.append(true_preds)
        true_labels.append(true_labs)

    # Use seqeval for entity-level metrics
    try:
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )

        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

        print("\n" + classification_report(true_labels, true_predictions))

    except ImportError:
        print("⚠️  seqeval not installed. Install with: pip install seqeval")
        correct = sum(
            1
            for pred_seq, lab_seq in zip(true_predictions, true_labels)
            for pred, lab in zip(pred_seq, lab_seq)
            if pred == lab
        )
        total = sum(len(lab_seq) for lab_seq in true_labels)
        results = {"accuracy": correct / total if total > 0 else 0}

    return results


# ============================================================================
# DATA LOADING
# ============================================================================


def load_data(config: Config):
    """Load train and test data"""

    print(f"📂 Loading training data from {config.train_data_file}...")
    with open(config.train_data_file, "r") as f:
        train_data = json.load(f)

    print(f"📂 Loading test data from {config.test_data_file}...")
    with open(config.test_data_file, "r") as f:
        test_data = json.load(f)

    print(f"✓ Loaded {len(train_data)} training invoices")
    print(f"✓ Loaded {len(test_data)} test invoices")

    # Split train into train/val
    train_data, val_data = train_test_split(
        train_data,
        test_size=config.val_ratio,
        random_state=config.random_seed,
        shuffle=True,
    )

    print("\n📊 Data splits:")
    print(f"  Train:      {len(train_data)} invoices")
    print(f"  Validation: {len(val_data)} invoices")
    print(f"  Test:       {len(test_data)} invoices")

    return train_data, val_data, test_data


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def train_model(config: Config):
    """Main training function"""

    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_default_device(device)
    print(f"🚀 Using device: {device}")

    # Load data
    train_data, val_data, test_data = load_data(config)

    # Initialize processor and model
    print(f"\n🤖 Loading model: {config.model_name}")
    processor = LayoutLMv3Processor.from_pretrained(config.model_name, apply_ocr=False)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.label2id),
        id2label=config.id2label,
        label2id=config.label2id,
    )

    print(f"✓ Base model loaded with {len(config.label2id)} labels")

    # Apply LoRA
    print(f"\n🔧 Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = InvoiceDataset(
        train_data, processor, config.max_length, config.label2id
    )
    val_dataset = InvoiceDataset(
        val_data, processor, config.max_length, config.label2id
    )
    test_dataset = InvoiceDataset(
        test_data, processor, config.max_length, config.label2id
    )
    print("✓ Datasets created")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # report_to="tensorboard",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Initialize trainer
    print("\n⚙️  Initializing trainer...")
    trainer = FocalLossTrainer(
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "=" * 70)
    print("🚀 Starting training...")
    print("=" * 70 + "\n")

    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("📊 Evaluating on test set...")
    print("=" * 70 + "\n")

    test_results = trainer.evaluate(test_dataset)
    print("\n📈 Test Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    # Save model
    print(f"\n💾 Saving LoRA adapter to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)

    # Save config
    config_path = Path(config.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {k: v for k, v in vars(config).items() if not k.startswith("_")},
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)

    return trainer, test_results


# ============================================================================
# INFERENCE
# ============================================================================


def predict_invoice(
    image_path: str,
    words: List[str],
    boxes: List[List[int]],
    model_path: str = "models/layoutlmv3-lora-invoice-number",
    base_model: str = "microsoft/layoutlmv3-base",
):
    """
    Run inference on a single invoice

    Args:
        image_path: Path to invoice image
        words: List of words from OCR
        boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000
        model_path: Path to LoRA adapter
        base_model: Base model name

    Returns:
        predicted_labels: List of predicted label strings
        invoice_number: Extracted invoice number string
    """
    # Load processor
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)

    # Load base model + LoRA adapter
    print("Loading LoRA model...")
    base = LayoutLMv3ForTokenClassification.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Process
    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert to labels
    predicted_labels = []
    invoice_tokens = []

    token_boxes = encoding.bbox[0].tolist()
    word_ids = encoding.word_ids(0)

    prev_word_idx = None
    for idx, (pred, box, word_idx) in enumerate(
        zip(predictions[0].tolist(), token_boxes, word_ids)
    ):
        if box != [0, 0, 0, 0] and word_idx is not None:
            label = model.config.id2label[pred]

            # Only take first subtoken prediction for each word
            if word_idx != prev_word_idx:
                predicted_labels.append(label)

                if label.startswith("B-INVOICE") or label.startswith("I-INVOICE"):
                    invoice_tokens.append(words[word_idx])

                prev_word_idx = word_idx

    invoice_number = " ".join(invoice_tokens) if invoice_tokens else None

    return predicted_labels, invoice_number


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train LayoutLMv3 for Invoice NER with LoRA"
    )
    parser.add_argument("--train_data", default="data/SROIE2019/train/train.json")
    parser.add_argument("--test_data", default="data/SROIE2019/test/test.json")
    parser.add_argument("--output_dir", default="models/layoutlmv3-lora-invoice-number")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    args = parser.parse_args()

    # Initialize config
    config = Config(
        train_data_file=args.train_data,
        test_data_file=args.test_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    # Train
    trainer, test_results = train_model(config)
