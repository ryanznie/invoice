import argparse
import json
import os
from PIL import Image

from datasets import Dataset, Features, Sequence, Value
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)


def main(args):
    # 1. Load dataset
    with open(args.train_file, "r") as f:
        data = json.load(f)

    # 2. Create Hugging Face Dataset
    features = Features(
        {
            "words": Sequence(Value("string")),
            "bboxes": Sequence(Sequence(Value("int64"))),
            "labels": Sequence(Value("string")),
            "file": Value("string"),
        }
    )

    dataset = Dataset.from_list(data, features=features)

    # 3. Define labels
    labels = list(set([label for sublist in dataset["labels"] for label in sublist]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # 4. Initialize processor
    # Use apply_ocr=False because we have pre-tokenized words and boxes
    processor = AutoProcessor.from_pretrained(args.model_name, apply_ocr=False)

    # 5. Preprocessing function
    def preprocess_data(examples):
        images = [
            Image.open(os.path.join(args.image_dir, f)).convert("RGB")
            for f in examples["file"]
        ]
        words = examples["words"]
        boxes = examples["bboxes"]
        word_labels = examples["labels"]

        # The processor needs the label mapping to convert string labels to IDs
        def get_label_ids(labels):
            return [label2id[label] for label in labels]

        word_labels_ids = [get_label_ids(label) for label in word_labels]

        encoded_inputs = processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels_ids,
            truncation=True,
            padding="max_length",
        )

        return encoded_inputs

    # 6. Map to new features with updated labels
    features = Features(
        {
            "pixel_values": Sequence(Sequence(Sequence(Value("float32")))),
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "bbox": Sequence(Sequence(Value("int64"))),
            "labels": Sequence(Value("int64")),
        }
    )

    processed_dataset = dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=dataset.column_names,
        features=features,
    ).train_test_split(test_size=0.1, seed=42)

    processed_dataset["train"].set_format(type="torch")
    processed_dataset["test"].set_format(type="torch")

    # 7. Define model
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    # 8. Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # 9. Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=1e-4,
        do_train=True,
        do_eval=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
    )

    # 10. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
    )

    # 11. Train
    trainer.train()

    # 12. Save model
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train.json",
        help="Path to the training data.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/SROIE2019/train/img",
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./layoutlmv3-finetuned-lora",
        help="Output directory for trained model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/layoutlmv3-base",
        help="Pretrained model name.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training and evaluation.",
    )
    args = parser.parse_args()
    main(args)
