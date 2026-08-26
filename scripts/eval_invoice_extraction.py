"""
Offline eval for invoice-number extraction against a labeled dataset.

This intentionally uses deterministic extraction metrics. LLM-as-judge and
online eval hooks can be layered on later once production traces exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

INVOICE_PATTERN = re.compile(r"^(?=.*[A-Za-z0-9])[A-Za-z0-9][A-Za-z0-9._/-]{2,}$")


def normalize_invoice_number(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value.lower() == "not found":
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def build_multipart(
    fields: dict[str, str], files: dict[str, Path]
) -> tuple[bytes, str]:
    boundary = f"----invoice-ner-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def predict(
    api_url: str, image_path: Path, ocr_payload: dict[str, Any]
) -> dict[str, Any]:
    tmp_ocr = image_path.with_suffix(".eval-ocr.json")
    tmp_ocr.write_text(json.dumps(ocr_payload), encoding="utf-8")
    try:
        body, boundary = build_multipart({}, {"image": image_path, "ocr_file": tmp_ocr})
        request = Request(
            f"{api_url}/predict",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    finally:
        tmp_ocr.unlink(missing_ok=True)


def evaluate_example(
    api_url: str, image_root: Path, example: dict[str, Any]
) -> dict[str, Any]:
    image_name = example.get("image_path") or example.get("file")
    image_path = image_root / image_name
    expected = example.get("answer_text") or example.get("invoice_number")

    ocr_payload = {
        "words": example["words"],
        "bboxes": example.get("bboxes") or example.get("boxes"),
    }
    start = time.perf_counter()
    response = predict(api_url, image_path, ocr_payload)
    latency_ms = (time.perf_counter() - start) * 1000

    actual = response.get("invoice_number")
    expected_norm = normalize_invoice_number(expected)
    actual_norm = normalize_invoice_number(actual)

    return {
        "file": image_name,
        "expected": expected,
        "actual": actual,
        "expected_norm": expected_norm,
        "actual_norm": actual_norm,
        "exact_match": actual == expected,
        "normalized_exact_match": actual_norm == expected_norm,
        "edit_distance": edit_distance(actual_norm, expected_norm),
        "valid_format": bool(INVOICE_PATTERN.match(actual or "")),
        "not_found": actual_norm == "",
        "extraction_method": response.get("extraction_method", "unknown"),
        "latency_ms": latency_ms,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    latencies = [row["latency_ms"] for row in rows]
    method_counts = Counter(row["extraction_method"] for row in rows)
    return {
        "total": total,
        "exact_match_rate": sum(row["exact_match"] for row in rows) / total,
        "normalized_exact_match_rate": sum(
            row["normalized_exact_match"] for row in rows
        )
        / total,
        "valid_format_rate": sum(row["valid_format"] for row in rows) / total,
        "not_found_rate": sum(row["not_found"] for row in rows) / total,
        "mean_edit_distance": statistics.mean(row["edit_distance"] for row in rows),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": (
            statistics.quantiles(latencies, n=20)[18] if total >= 20 else max(latencies)
        ),
        "method_counts": dict(method_counts),
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate invoice extraction API.")
    parser.add_argument("--api-url", default="http://localhost:7860")
    parser.add_argument("--dataset", default="data/train/qa_dataset.json")
    parser.add_argument("--image-root", default="data/SROIE2019/train/img")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output-dir", default="monitoring/evals/latest")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    examples = dataset[: args.limit] if args.limit else dataset
    image_root = Path(args.image_root)

    rows: list[dict[str, Any]] = []
    try:
        for example in examples:
            rows.append(evaluate_example(api_url, image_root, example))
    except (
        FileNotFoundError,
        HTTPError,
        URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        print(f"eval failed: {exc}")
        return 1

    write_outputs(rows, Path(args.output_dir))
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
