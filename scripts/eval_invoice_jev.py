"""Evaluate invoice extraction predictions with the TypeSafe Jev API.

Usage:
    JEV_API_KEY=... uv run python scripts/eval_invoice_jev.py \
        --input data/test/predictions_with_labels.csv \
        --output-dir monitoring/evals/jev
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JEV_URL = "https://api.typesafe.ai/v1/systemone"


def load_predictions(path: Path) -> list[dict[str, str]]:
    """Reconstruct one predicted invoice number from token-level CSV output."""
    grouped: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            item = grouped.setdefault(
                row["file"],
                {"file": row["file"], "truth": row["true_invoice_number"], "words": []},
            )
            if row["prediction"] in {"LABEL_1", "LABEL_2"}:
                item["words"].append(row["word"])

    return [
        {
            "file": item["file"],
            "prediction": " ".join(item["words"]),
            "truth": item["truth"],
        }
        for item in grouped.values()
    ]


def evaluate(row: dict[str, str], api_key: str) -> dict[str, Any]:
    payload = {
        "model": "jev-latest",
        "state": {
            "task": "Invoice number extraction evaluation",
            "ground_truth": row["truth"],
            "prediction": row["prediction"],
        },
        "questions": {
            "correct": {
                "type": "noul",
                "instructions": (
                    "Is the predicted invoice number exactly correct compared with the "
                    "ground-truth invoice number? Ignore capitalization and surrounding "
                    "whitespace, but do not forgive missing, extra, or changed characters."
                ),
            },
            "quality": {
                "type": "score",
                "instructions": "How reliable is this extraction?",
                "criteria": [
                    "The prediction is missing or materially wrong.",
                    "The prediction is close but has a minor formatting or character issue.",
                    "The prediction exactly matches the ground truth.",
                ],
            },
        },
    }
    request = Request(
        JEV_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504, 529} or attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    latency_ms = (time.perf_counter() - started) * 1000

    correct = result["answers"]["correct"]
    quality = result["answers"]["quality"]
    return {
        **row,
        "noul": correct["noul"],
        "quality_score": quality["score"],
        "quality_confidence": quality.get("confidence"),
        "model": result.get("model"),
        "usage": result.get("usage"),
        "latency_ms": latency_ms,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, int] = {}
    for row in rows:
        key = str(round(row["quality_score"]))
        scores[key] = scores.get(key, 0) + 1
    latencies = sorted(row["latency_ms"] for row in rows)
    p95_index = min(len(latencies) - 1, round(0.95 * len(latencies)) - 1)
    return {
        "total": len(rows),
        "exact_or_likely_exact": sum(row["noul"] >= 0.5 for row in rows),
        "high_confidence_exact": sum(row["noul"] >= 0.9 for row in rows),
        "quality_scores": scores,
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[p95_index],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/test/predictions_with_labels.csv")
    parser.add_argument("--output-dir", default="monitoring/evals/jev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("JEV_API_KEY")
    if not api_key:
        print("eval failed: set JEV_API_KEY in the environment", file=sys.stderr)
        return 1

    rows = load_predictions(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]

    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate, row, api_key) for row in rows]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(f"Evaluated {index}/{len(rows)}", end="\r", file=sys.stderr)
    except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        print(f"\neval failed: {exc}", file=sys.stderr)
        return 1

    results.sort(key=lambda row: row["file"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summarize(results), indent=2), encoding="utf-8"
    )
    (output_dir / "rows.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(summarize(results), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
