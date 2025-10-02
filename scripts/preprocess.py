import os
import json
import logging
import re
from glob import glob


def normalize_bbox(bbox, width=1000, height=1000, image_w=1000, image_h=1000):
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
    with open(path, "r", encoding="utf-8") as f:
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
    tokens_with_delimiters = re.split(r"([/-])", invoice_str.strip())

    # Split tokens by space and filter out empty strings
    final_tokens = []
    for token in tokens_with_delimiters:
        if token in ("/", "-"):
            final_tokens.append(token)
        else:
            final_tokens.extend(token.split())

    return [t for t in final_tokens if t]


def preprocess(ocr_dir, labels_path, output_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    processed = []

    for box_path in glob(os.path.join(ocr_dir, "*.txt")):
        file_id = os.path.basename(box_path).replace(".txt", ".jpg")

        if file_id not in labels_data:
            logging.warning(f"No label found for {file_id}, skipping")
            continue

        label = labels_data[file_id]
        if label.strip().lower() == "ambiguous":
            logging.warning(f"Skipping ambiguous label in {file_id}")
            continue

        ocr_entries = parse_ocr_file(box_path)
        # Sort by y then x coordinate to ensure reading order
        ocr_entries.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
        invoice_tokens = split_invoice_string(label)

        words, bboxes = [], []
        for entry in ocr_entries:
            text = entry["text"]
            bbox = entry["bbox"]
            tokens = split_invoice_string(text)
            for token in tokens:
                words.append(token)
                bboxes.append(normalize_bbox(bbox))

        labels = ["O"] * len(words)
        if not invoice_tokens:
            pass  # No invoice tokens to match
        else:
            best_match = None
            min_dist = float("inf")

            # Find all possible start indices
            starts = [i for i, w in enumerate(words) if w == invoice_tokens[0]]

            for start in starts:
                indices = [start]
                curr = start + 1
                for t in invoice_tokens[1:]:
                    found = False
                    for i in range(curr, len(words)):
                        if words[i] == t:
                            indices.append(i)
                            curr = i + 1
                            found = True
                            break
                    if not found:
                        indices = []
                        break

                if indices:
                    dist = indices[-1] - indices[0]
                    if dist < min_dist:
                        min_dist = dist
                        best_match = indices

            if best_match:
                labels[best_match[0]] = "B-INVOICE_ID"
                for i in best_match[1:]:
                    labels[i] = "I-INVOICE_ID"

        processed.append(
            {
                "file": file_id,
                "words": words,
                "bboxes": bboxes,
                "labels": labels,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Preprocessing complete. Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocr_dir", default="data/SROIE2019/train/box", help="OCR txt dir"
    )
    parser.add_argument(
        "--labels", default="data/labels.json", help="Ground truth labels json"
    )
    parser.add_argument(
        "--output", default="data/train.json", help="Output LayoutLMv3 json"
    )
    parser.add_argument("--log", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    preprocess(args.ocr_dir, args.labels, args.output)
