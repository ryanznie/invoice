"""
Locust load profile for the Invoice NER API.

Run with:
    locust -f locustfile.py --host=http://localhost:7860 --headless --users=10 --spawn-rate=2 --run-time=60s

By default this samples local labeled training invoices:
    data/train/qa_dataset.json
    data/SROIE2019/train/img

Override with:
    INVOICE_NER_LOCUST_DATASET=/path/to/qa_dataset.json
    INVOICE_NER_LOCUST_IMAGE_ROOT=/path/to/images
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

DATASET_PATH = Path(
    os.getenv("INVOICE_NER_LOCUST_DATASET", "data/train/qa_dataset.json")
)
IMAGE_ROOT = Path(
    os.getenv("INVOICE_NER_LOCUST_IMAGE_ROOT", "data/SROIE2019/train/img")
)
SAMPLE_LIMIT = int(os.getenv("INVOICE_NER_LOCUST_SAMPLE_LIMIT", "0"))


def _load_examples() -> list[dict]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if SAMPLE_LIMIT > 0:
        dataset = dataset[:SAMPLE_LIMIT]

    examples = []
    for item in dataset:
        image_name = item.get("image_path") or item.get("file")
        image_path = IMAGE_ROOT / image_name
        boxes = item.get("bboxes") or item.get("boxes")
        if image_path.exists() and item.get("words") and boxes:
            examples.append(
                {
                    "image_path": image_path,
                    "ocr": {"words": item["words"], "bboxes": boxes},
                    "expected": item.get("answer_text") or item.get("invoice_number"),
                }
            )

    if not examples:
        raise RuntimeError(
            f"No usable Locust examples found in {DATASET_PATH} with images in {IMAGE_ROOT}"
        )
    return examples


EXAMPLES = _load_examples()


def _sample_invoice_payload() -> tuple[str, bytes, bytes, str | None]:
    example = random.choice(EXAMPLES)
    image_path = example["image_path"]
    return (
        image_path.name,
        image_path.read_bytes(),
        json.dumps(example["ocr"]).encode("utf-8"),
        example["expected"],
    )


class InvoiceNerUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="GET /metrics")

    @task(8)
    def predict(self):
        image_name, image_bytes, ocr_bytes, _expected = _sample_invoice_payload()
        files = {
            "image": (image_name, image_bytes, "image/jpeg"),
            "ocr_file": ("ocr.json", ocr_bytes, "application/json"),
        }
        with self.client.post(
            "/predict",
            files=files,
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}")
                return

            data = response.json()
            if "invoice_number" not in data or "extraction_method" not in data:
                response.failure("response missing extraction fields")
