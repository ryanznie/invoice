import os
import json
import logging
import re
from glob import glob
from PIL import Image


def estimate_word_boxes(line_text, tokens, line_bbox):
    """
    Estimate individual word bounding boxes from line bbox.
    Distributes bbox proportionally based on character positions.
    """
    if len(tokens) == 1:
        return [line_bbox]

    x_min, y_min, x_max, y_max = line_bbox
    line_width = x_max - x_min

    word_boxes = []
    char_position = 0
    total_chars = len(line_text)

    for token in tokens:
        # Find token in remaining text
        token_start = line_text.find(token, char_position)

        if token_start == -1:
            # Fallback: equal distribution
            token_width = line_width / len(tokens)
            token_x_min = x_min + (len(word_boxes) * token_width)
            token_x_max = token_x_min + token_width
        else:
            # Proportional position
            token_end = token_start + len(token)

            start_ratio = token_start / max(total_chars, 1)
            end_ratio = token_end / max(total_chars, 1)

            token_x_min = x_min + (start_ratio * line_width)
            token_x_max = x_min + (end_ratio * line_width)

            char_position = token_end

        word_boxes.append([int(token_x_min), y_min, int(token_x_max), y_max])

    return word_boxes


def normalize_bbox(bbox, image_w, image_h):
    """Normalize OCR bbox to LayoutLM format [0,1000]."""
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * (x0 / image_w)),
        int(1000 * (y0 / image_h)),
        int(1000 * (x1 / image_w)),
        int(1000 * (y1 / image_h)),
    ]


def parse_ocr_file(path):
    """Parse a single OCR .txt file into [{'text': str, 'bbox': [x0,y0,x1,y1]}, ...]."""
    words = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:]).strip()
            if not text:
                continue

            # bbox = min_x, min_y, max_x, max_y
            xs, ys = coords[::2], coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            words.append({"text": text, "bbox": bbox})
    return words


def split_invoice_string(invoice_str):
    """Split invoice string into tokens."""
    # Keep delimiters by capturing them
    tokens_with_delimiters = re.split(r"([/:.#\(\)\[\]-])", invoice_str.strip())

    # Split tokens by space and filter out empty strings
    final_tokens = []
    for token in tokens_with_delimiters:
        if token in ("/", "-", ":", ".", "#", "[", "]", "(", ")"):
            final_tokens.append(token)
        else:
            final_tokens.extend(token.split())

    return [t for t in final_tokens if t]


def preprocess(ocr_dir, output_path, labels_path=None, mode="train"):
    labels_data = {}
    if mode == "train":
        if not labels_path:
            raise ValueError("Labels path must be provided in train mode")
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

    processed = []
    ambiguous_count = 0

    for box_path in glob(os.path.join(ocr_dir, "*.txt")):
        file_id = os.path.basename(box_path).replace(".txt", ".jpg")

        if mode == "train":
            if file_id not in labels_data:
                logging.warning(f"No label found for {file_id}, skipping")
                continue

            label = labels_data[file_id]
            if label.strip().lower() == "ambiguous":
                logging.warning(f"Skipping ambiguous label in {file_id}")
                ambiguous_count += 1
                continue
            invoice_tokens = split_invoice_string(label)
        else:  # test mode
            label = ""
            invoice_tokens = []

        ocr_entries = parse_ocr_file(box_path)
        # Sort by y then x coordinate to ensure reading order
        ocr_entries.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))

        words, bboxes = [], []
        img_path = os.path.join(ocr_dir.replace("box", "img"), file_id)
        img_w, img_h = Image.open(img_path).size

        # FIX: Estimate word-level boxes
        for entry in ocr_entries:
            text = entry["text"]
            line_bbox = entry["bbox"]
            tokens = split_invoice_string(text)

            # Estimate individual word boxes from line box
            token_boxes = estimate_word_boxes(text, tokens, line_bbox)

            # Add words with their estimated boxes
            for token, token_bbox in zip(tokens, token_boxes):
                words.append(token)
                bboxes.append(normalize_bbox(token_bbox, image_w=img_w, image_h=img_h))

        labels = ["O"] * len(words)

        # FIX: Better invoice matching
        if mode == "train" and invoice_tokens:
            words_lower = [w.lower() for w in words]
            invoice_lower = [t.lower() for t in invoice_tokens]

            best_match = None
            min_dist = float("inf")

            # Try to find match allowing small gaps
            for start_idx in range(len(words) - len(invoice_tokens) + 1):
                match_indices = []
                inv_idx = 0

                for i in range(
                    start_idx, min(start_idx + len(invoice_tokens) * 2, len(words))
                ):
                    if inv_idx >= len(invoice_lower):
                        break
                    if words_lower[i] == invoice_lower[inv_idx]:
                        match_indices.append(i)
                        inv_idx += 1

                if len(match_indices) == len(invoice_tokens):
                    dist = match_indices[-1] - match_indices[0]
                    if dist < min_dist:
                        min_dist = dist
                        best_match = match_indices

            if best_match:
                labels[best_match[0]] = "B-INVOICE_ID"
                for i in best_match[1:]:
                    labels[i] = "I-INVOICE_ID"
            else:
                logging.warning(f"⚠️  Could not find invoice '{label}' in {file_id}")

        record = {
            "file": file_id,
            "words": words,
            "bboxes": bboxes,
        }
        if mode == "train":
            record["labels"] = labels

        processed.append(record)

    if mode == "train":
        from collections import Counter

        all_labels = [label for record in processed for label in record["labels"]]
        label_counts = Counter(all_labels)
        logging.info("Class distributions:")
        for label, count in sorted(label_counts.items()):
            logging.info(f"  - {label}: {count}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Preprocessing complete. Saved to {output_path}")
    logging.info(f"- Mode: {mode}")
    logging.info(f"- Processed: {len(processed)} files")
    if mode == "train":
        logging.info(f"- Ambiguous: {ambiguous_count} files skipped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocr_dir",
        required=True,
        help="OCR txt dir. train=data/SROIE2019/train/box, test=data/SROIE2019/test/box",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Ground truth: data/labels.json or data/test_labels.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output json for LayoutLMv3 train=data/SROIE2019/train/train.json, test=data/SROIE2019/test/test.json",
    )
    parser.add_argument("--log", default="INFO", help="Logging level")
    parser.add_argument(
        "--mode", choices=["train", "test"], required=True, help="Run mode"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    preprocess(args.ocr_dir, args.output, labels_path=args.labels, mode=args.mode)

# python scripts/preprocess.py --mode test --labels data/SROIE2019/test/test_labels.json --ocr_dir data/SROIE2019/test/box/ --output data/SROIE2019/test/test.json
# python scripts/preprocess.py --mode train --labels data/SROIE2019/train/labels.json --ocr_dir data/SROIE2019/train/box/ --output data/SROIE2019/train/train.json
