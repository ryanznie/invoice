import os
import json

# Paths
OCR_DIR = "data/SROIE2019/train/box/"
GT_PATH = "data/SROIE2019/train/labels.json"
OUTPUT_PATH = "data/SROIE2019/train/qa_dataset.json"


def parse_ocr_file(path):
    """Parse OCR file -> lists of words and bounding boxes."""
    words, bboxes = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue

            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:]).strip()
            # Convert quadrilateral → rectangular [x0, y0, x2, y2]
            x0, y0, x2, y2 = coords[0], coords[1], coords[4], coords[5]
            words.append(text)
            bboxes.append([x0, y0, x2, y2])
    return words, bboxes


def build_qa_sample(image_file, words, bboxes, invoice_number):
    """Build a single QA example entry."""
    # Find matching boxes for invoice number tokens
    answer_bboxes = [
        bbox
        for w, bbox in zip(words, bboxes)
        if invoice_number.strip().lower() in w.strip().lower()
    ]

    if not answer_bboxes:
        print(f"⚠️ Warning: no bbox match for {invoice_number} in {image_file}")

    return {
        "image_path": image_file,
        "question": "What is the invoice number?",
        "words": words,
        "bboxes": bboxes,
        "answer_text": invoice_number,
        "answer_bboxes": answer_bboxes,
    }


def main():
    # 1️⃣ Load ground truth
    with open(GT_PATH, "r") as f:
        ground_truth = json.load(f)

    qa_samples = []

    # 2️⃣ Iterate over ground-truth entries
    for img_name, invoice_number in ground_truth.items():
        if invoice_number.lower() == "ambiguous" or not invoice_number.strip():
            continue  # skip ambiguous or empty labels

        txt_name = img_name.replace(".jpg", ".txt")
        ocr_path = os.path.join(OCR_DIR, txt_name)

        if not os.path.exists(ocr_path):
            print(f"❌ Missing OCR file for {img_name}")
            continue

        words, bboxes = parse_ocr_file(ocr_path)
        sample = build_qa_sample(img_name, words, bboxes, invoice_number)
        qa_samples.append(sample)

    # 3️⃣ Save to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(qa_samples, f, indent=2)

    print(f"\n✅ Saved {len(qa_samples)} clean QA samples → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
