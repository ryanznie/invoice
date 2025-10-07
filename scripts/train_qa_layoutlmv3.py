import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
)


@dataclass
class TrainingConfig:
    """Configuration for training arguments."""

    model_checkpoint: str = "microsoft/layoutlmv3-base"
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    output_dir: str = "models/layoutlmv3-finetuned-qa"
    logging_dir: str = "logs"
    dataset_path: str = "data/SROIE2019/train/qa_dataset.json"
    images_dir: str = "data/SROIE2019/train/img"


class SROIEDataset(Dataset):
    """Dataset for SROIE invoice data."""

    def __init__(self, config, processor, split="train"):
        self.config = config
        self.processor = processor
        self.dataset = load_dataset("json", data_files={split: config.dataset_path})[
            split
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_path = os.path.join(self.config.images_dir, sample["image_path"])
        image = Image.open(image_path).convert("RGB")

        words = sample["words"]
        boxes = sample["bboxes"]

        # The processor is a feature extractor and a tokenizer. We need to handle them separately.
        features = self.processor.feature_extractor(image, return_tensors="pt")
        encoding = self.processor.tokenizer(
            text=sample["question"],
            text_pair=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        encoding["pixel_values"] = features.pixel_values

        # Find start and end token indices for the answer
        start_positions, end_positions = self.get_answer_token_indices(
            encoding, sample["answer_text"], sample["words"]
        )

        return {
            **{k: v.squeeze() for k, v in encoding.items()},
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
        }

    def get_answer_token_indices(self, encoding, answer_text, words):
        """Find start/end token indices for the answer in the encoded sequence."""
        sequence_ids = encoding.sequence_ids()
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(encoding.input_ids[0]) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Find the start and end of the answer in the tokenized sequence
        # This is a simplified approach; a more robust solution would handle tokenization edge cases
        start_char = answer_text.split()[0]
        end_char = answer_text.split()[-1]

        start_token = -1
        end_token = -1

        for i, word in enumerate(words):
            if start_char in word and start_token == -1:
                start_token = i
            if end_char in word:
                end_token = i

        if start_token != -1 and end_token != -1:
            return token_start_index + start_token, token_start_index + end_token
        else:
            return 0, 0  # Default to CLS token if answer not found


def main():
    config = TrainingConfig()

    # Initialize processor and model
    processor = LayoutLMv3Processor.from_pretrained(
        config.model_checkpoint, apply_ocr=False
    )
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(config.model_checkpoint)

    # Create datasets
    train_dataset = SROIEDataset(config, processor, split="train")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_dir=config.logging_dir,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        gradient_accumulation_steps=4,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    print("🚀 Starting model training...")
    trainer.train(
        resume_from_checkpoint="models/layoutlmv3-finetuned-qa/checkpoint-278"
    )
    print("✅ Training complete!")

    # Save the model and processor
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    print(f"💾 Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
