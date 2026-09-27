#!/usr/bin/env python3
"""Verify processor startup branches in the production worker image offline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "ghcr.io/ryanznie/invoice-ner-backend:v0.3.0-cpu.2"
DEFAULT_ARTIFACT = REPO_ROOT / "docs" / "artifacts" / "processor-startup-e2e.json"
RESULT_PREFIX = "PROCESSOR_E2E_RESULT="
CONTAINER_PROBE = """
import json
from src import inference

inference.load_model()
print(
    "PROCESSOR_E2E_RESULT="
    + json.dumps(
        {
            "backend": type(inference.backend).__name__,
            "processor": type(inference.processor).__name__,
            "tokenizer_source": inference.processor.tokenizer.name_or_path,
            "base_model": inference.BASE_MODEL,
            "configured_processor_path": inference.PROCESSOR_PATH,
        },
        sort_keys=True,
    )
)
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def run_case(
    name: str,
    image: str,
    *,
    processor_path: str | None = None,
    base_model: str = "/app/models/artifacts",
) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        f"BASE_MODEL={base_model}",
    ]
    if processor_path is not None:
        command.extend(["--env", f"PROCESSOR_PATH={processor_path}"])
    command.extend(["--entrypoint", "python", image, "-c", CONTAINER_PROBE])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    marker = next(
        (
            line.removeprefix(RESULT_PREFIX)
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    details = json.loads(marker) if marker is not None else None
    return {
        "name": name,
        "passed": completed.returncode == 0 and details is not None,
        "exit_code": completed.returncode,
        "details": details,
        "startup_log": [
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(RESULT_PREFIX)
        ],
        "stderr": completed.stderr.splitlines(),
    }


def run_contract(image: str, output: Path) -> int:
    cases = [
        run_case("bundled_processor", image),
        run_case(
            "invalid_processor_path_falls_back_to_base_model",
            image,
            processor_path="/missing/processor",
        ),
    ]
    passed = all(case["passed"] for case in cases)
    artifact = {
        "schema_version": 1,
        "suite": "processor-startup-e2e",
        "result": "passed" if passed else "failed",
        "container_image": image,
        "network_mode": "Hugging Face offline",
        "cases": cases,
        "repeat": (
            "RUN_PROCESSOR_CONTAINER_E2E=1 uv run pytest --no-cov "
            "tests/e2e/test_processor_startup.py -q"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    print(f"Artifact: {output}")
    return 0 if passed else 1


def test_processor_startup_e2e() -> None:
    """Load the real processor and ONNX runtime from the production image."""
    import pytest

    if os.getenv("RUN_PROCESSOR_CONTAINER_E2E") != "1":
        pytest.skip("set RUN_PROCESSOR_CONTAINER_E2E=1 to run the Docker E2E")
    assert run_contract(DEFAULT_IMAGE, DEFAULT_ARTIFACT) == 0


def main() -> int:
    args = parse_args()
    return run_contract(args.image, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
