"""
Production-quality benchmarking script for invoice extraction models.

This script provides comprehensive benchmarking with Weights & Biases integration:
- Per-invoice logging (prediction, ground truth, latency, method used)
- Run-level summary metrics (accuracy, P50/P95 latency, fallback rate)
- Histograms and visualizations
- Model comparison across multiple runs
- Modular design for easy model integration

Usage:
    python benchmarks/benchmark.py --model hybrid --data-dir data/test --run-name "hybrid-v1"
    python benchmarks/benchmark.py --model layoutlmv3 --data-dir data/test --run-name "layoutlmv3-baseline"
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
from tqdm import tqdm

from benchmarks.models.base import BaseInvoiceModel, InferenceResult
from benchmarks.models.layoutlmv3_model import LayoutLMv3Model
from benchmarks.models.hybrid_model import HybridModel
from benchmarks.models.onnx_model import OnnxModel
from benchmarks.models.openrouter_model import OpenRouterModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class InvoiceBenchmarkResult:
    """Single invoice benchmark result."""

    file_name: str
    ground_truth: Optional[str]
    prediction: Optional[str]
    is_correct: bool
    latency_ms: float
    method_used: str  # 'heuristic', 'model', 'model_fallback', 'failed'
    confidence: Optional[float]
    fallback_used: bool
    needs_human_review: bool  # True if prediction has multiple words
    metadata: Dict[str, Any]


class InvoiceBenchmark:
    """
    Comprehensive benchmarking system for invoice extraction models.

    Features:
    - Loads test data from JSON or image directory
    - Runs inference with timing
    - Computes accuracy and latency metrics
    - Logs to Weights & Biases with rich visualizations
    - Supports multiple model comparisons
    """

    def __init__(
        self,
        model: BaseInvoiceModel,
        data_dir: str,
        wandb_project: str = "invoice-extraction-benchmark",
        wandb_entity: Optional[str] = None,
        run_name: Optional[str] = None,
        run_tags: Optional[List[str]] = None,
        dataset_version: str = "v1.0",
        offline: bool = False,
        split: str = "test",
    ):
        """
        Initialize benchmark.

        Args:
            model: Model to benchmark (must implement BaseInvoiceModel)
            data_dir: Directory containing test data
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team)
            run_name: Name for this benchmark run
            run_tags: Tags for organizing runs
            dataset_version: Version identifier for the dataset
            offline: Run in offline mode (no W&B sync)
            split: Dataset split to use ('test' or 'train')
        """
        self.model = model
        self.data_dir = Path(data_dir)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.run_name = run_name or f"{model.model_name}-{int(time.time())}"
        self.run_tags = run_tags or []
        self.dataset_version = dataset_version
        self.offline = offline
        self.split = split

        self.results: List[InvoiceBenchmarkResult] = []
        self.wandb_run = None

    def load_test_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Load test data and ground truth labels.

        Expected structure:
        - Test split:
          - Data: data/test/test.json with [{"file": "...", "words": [...], "bboxes": [...]}]
          - Labels: data/SROIE2019/test/test_labels.json with {"filename": "invoice_number"}
        - Train split:
          - Data: data/train/train.json with [{"file": "...", "words": [...], "bboxes": [...]}]
          - Labels: data/SROIE2019/train/labels.json with {"filename": "invoice_number"}

        Returns:
            Tuple of (test_data, ground_truth_labels)
        """
        logger.info(f"Loading {self.split} data from {self.data_dir}")

        # Determine file paths based on split
        if self.split == "train":
            data_json_path = self.data_dir / "train.json"
            labels_path = self.data_dir.parent / "SROIE2019" / "train" / "labels.json"
        else:  # test
            data_json_path = self.data_dir / "test.json"
            labels_path = (
                self.data_dir.parent / "SROIE2019" / "test" / "test_labels.json"
            )

        # Load data (words and bboxes)
        if not data_json_path.exists():
            raise FileNotFoundError(
                f"{self.split.capitalize()} data not found at {data_json_path}"
            )

        with open(data_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        logger.info(f"✓ Loaded {len(test_data)} {self.split} examples")

        # Load ground truth labels
        ground_truth = {}

        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            logger.info(f"✓ Loaded {len(ground_truth)} ground truth labels")
        else:
            logger.warning(f"Ground truth labels not found at {labels_path}")

        return test_data, ground_truth

    def run_single_inference(
        self,
        example: Dict[str, Any],
        ground_truth: Optional[str] = None,
        image_dir: Optional[Path] = None,
        box_dir: Optional[Path] = None,
    ) -> InvoiceBenchmarkResult:
        """
        Run inference on a single invoice and measure metrics.

        Args:
            example: Test example dictionary with "file" field
            ground_truth: Ground truth invoice number
            image_dir: Optional directory containing invoice images
            box_dir: Optional directory containing OCR .txt files

        Returns:
            InvoiceBenchmarkResult with all metrics
        """
        file_name = example.get("file", "unknown")

        # Load image if available
        image = None
        if image_dir and file_name != "unknown":
            image_path = image_dir / file_name
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")

        # Load OCR data from .txt file if available
        words = []
        boxes = []
        ocr_lines = None

        if box_dir and file_name != "unknown":
            ocr_file_path = (box_dir / file_name).with_suffix(".txt")
            if ocr_file_path.exists():
                try:
                    from src.utils import parse_ocr_text_file, normalize_boxes

                    # Parse OCR txt file
                    ocr_data = parse_ocr_text_file(str(ocr_file_path))
                    words = ocr_data["words"]
                    boxes = ocr_data["bboxes"]
                    ocr_lines = ocr_data.get(
                        "ocr_lines"
                    )  # IMPORTANT: Needed for heuristics!

                    # Normalize boxes if we have image dimensions
                    if image is not None:
                        img_width, img_height = image.size
                        boxes = normalize_boxes(boxes, img_width, img_height)

                    logger.debug(
                        f"Loaded OCR from {ocr_file_path}: {len(words)} words, {len(ocr_lines) if ocr_lines else 0} lines"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load OCR file {ocr_file_path}: {e}")
                    # Fall back to JSON data if available
                    words = example.get("words", [])
                    boxes = example.get("bboxes", example.get("boxes", []))
                    ocr_lines = example.get("ocr_lines")
            else:
                logger.warning(f"OCR file not found: {ocr_file_path}")
                # Fall back to JSON data if available
                words = example.get("words", [])
                boxes = example.get("bboxes", example.get("boxes", []))
                ocr_lines = example.get("ocr_lines")
        else:
            # No box_dir specified, use data from JSON
            words = example.get("words", [])
            boxes = example.get("bboxes", example.get("boxes", []))
            ocr_lines = example.get("ocr_lines")

        # Run inference with timing
        start_time = time.time()

        try:
            result: InferenceResult = self.model.predict(
                image=image,
                words=words,
                boxes=boxes,
                text=" ".join(words) if words else None,
                ocr_lines=ocr_lines,  # Pass OCR lines for heuristics
            )

            latency_ms = (time.time() - start_time) * 1000
            prediction = result.invoice_number
            confidence = result.confidence
            method_used = result.method
            metadata = result.metadata or {}

        except Exception as e:
            logger.error(f"Inference failed for {file_name}: {e}")
            latency_ms = (time.time() - start_time) * 1000
            prediction = None
            confidence = None
            method_used = "error"
            metadata = {"error": str(e)}

        # Determine correctness
        is_correct = self._check_correctness(prediction, ground_truth)

        # Check if fallback was used
        fallback_used = metadata.get("fallback_used", False)

        # Check if needs human review
        # Triggers when:
        # 1. Prediction is None or empty (no extraction)
        # 2. Prediction has multiple words (ambiguous extraction)
        needs_human_review = False
        if prediction is None or not prediction.strip():
            # No prediction - needs human review
            needs_human_review = True
        else:
            # Check if prediction has multiple words
            word_count = len(prediction.split())
            needs_human_review = word_count > 1

        return InvoiceBenchmarkResult(
            file_name=file_name,
            ground_truth=ground_truth,
            prediction=prediction,
            is_correct=is_correct,
            latency_ms=latency_ms,
            method_used=method_used,
            confidence=confidence,
            fallback_used=fallback_used,
            needs_human_review=needs_human_review,
            metadata=metadata,
        )

    def _check_correctness(
        self, prediction: Optional[str], ground_truth: Optional[str]
    ) -> bool:
        """
        Check if prediction matches ground truth.

        Args:
            prediction: Predicted invoice number
            ground_truth: Ground truth invoice number

        Returns:
            True if correct, False otherwise
        """
        if prediction is None or ground_truth is None:
            return False

        # Normalize for comparison (remove whitespace, lowercase)
        pred_norm = "".join(prediction.split()).lower()
        gt_norm = "".join(ground_truth.split()).lower()

        return pred_norm == gt_norm

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics from all results.

        Returns:
            Dictionary of metrics including accuracy, latency stats, fallback rate
        """
        if not self.results:
            return {}

        # Accuracy metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0

        # Latency metrics
        latencies = [r.latency_ms for r in self.results]
        latency_mean = np.mean(latencies)
        latency_median = np.median(latencies)
        latency_std = np.std(latencies)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        latency_min = np.min(latencies)
        latency_max = np.max(latencies)

        # Method breakdown
        method_counts = {}
        for r in self.results:
            method_counts[r.method_used] = method_counts.get(r.method_used, 0) + 1

        # Fallback metrics
        fallback_count = sum(1 for r in self.results if r.fallback_used)
        fallback_rate = fallback_count / total if total > 0 else 0.0

        # Human review metrics
        human_review_count = sum(1 for r in self.results if r.needs_human_review)
        human_review_rate = human_review_count / total if total > 0 else 0.0

        # Confidence metrics (if available)
        confidences = [r.confidence for r in self.results if r.confidence is not None]
        confidence_mean = np.mean(confidences) if confidences else None
        confidence_std = np.std(confidences) if confidences else None

        # Accuracy by method
        accuracy_by_method = {}
        for method in method_counts.keys():
            method_results = [r for r in self.results if r.method_used == method]
            if method_results:
                method_correct = sum(1 for r in method_results if r.is_correct)
                accuracy_by_method[method] = method_correct / len(method_results)

        return {
            # Overall metrics
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            # Latency metrics
            "latency_mean_ms": latency_mean,
            "latency_median_ms": latency_median,
            "latency_std_ms": latency_std,
            "latency_p95_ms": latency_p95,
            "latency_p99_ms": latency_p99,
            "latency_min_ms": latency_min,
            "latency_max_ms": latency_max,
            # Method breakdown
            "method_counts": method_counts,
            "accuracy_by_method": accuracy_by_method,
            # Fallback metrics
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
            # Human review metrics
            "human_review_count": human_review_count,
            "human_review_rate": human_review_rate,
            # Confidence metrics
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark pipeline.

        Steps:
        1. Initialize W&B run
        2. Load model
        3. Load test data
        4. Run inference on all examples
        5. Compute metrics
        6. Log to W&B
        7. Generate visualizations

        Returns:
            Dictionary of final metrics
        """
        try:
            # Step 1: Initialize W&B
            logger.info("Initializing Weights & Biases...")
            self._init_wandb()

            # Step 2: Load model
            logger.info("Loading model...")
            self.model.load()

            # Step 3: Load test data and ground truth labels
            test_data, ground_truth_labels = self.load_test_data()
            if not test_data:
                raise ValueError("No test data found")

            logger.info(f"Running benchmark on {len(test_data)} examples...")

            # Step 4: Run inference on all examples
            image_dir = self.data_dir.parent / "SROIE2019" / self.split / "img"
            box_dir = self.data_dir.parent / "SROIE2019" / self.split / "box"

            if not image_dir.exists():
                image_dir = None
                logger.warning("Image directory not found, running without images")

            if not box_dir.exists():
                box_dir = None
                logger.warning(
                    "Box directory not found, will use JSON data if available"
                )

            skipped_count = 0
            for example in tqdm(test_data, desc="Benchmarking"):
                file_name = example.get("file", "unknown")

                # Get ground truth for this file
                ground_truth = ground_truth_labels.get(file_name)

                # Skip if no ground truth or ambiguous
                if ground_truth is None:
                    logger.debug(f"Skipping {file_name}: no ground truth label")
                    skipped_count += 1
                    continue

                if ground_truth == "ambiguous":
                    logger.debug(f"Skipping {file_name}: ambiguous label")
                    skipped_count += 1
                    continue

                # Run inference
                result = self.run_single_inference(
                    example, ground_truth, image_dir, box_dir
                )
                self.results.append(result)

                # Log to W&B in real-time (optional, can be batched)
                if self.wandb_run:
                    self._log_single_result(result)

            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} samples (no label or ambiguous)")

            # Step 5: Compute aggregate metrics
            logger.info("Computing metrics...")
            metrics = self.compute_metrics()

            # Step 6: Log summary to W&B
            if self.wandb_run:
                self._log_summary(metrics)
                self._create_visualizations(metrics)

            # Step 7: Print summary
            self._print_summary(metrics)

            return metrics

        finally:
            # Cleanup
            self.model.cleanup()
            if self.wandb_run:
                self.wandb_run.finish()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        # Log dataset metadata
        model_config = self.model.get_config()

        self.wandb_run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.run_name,
            tags=self.run_tags,
            config={
                **model_config,
                "dataset_version": self.dataset_version,
                "dataset_split": self.split,
                "data_dir": str(self.data_dir),
            },
            mode="offline" if self.offline else "online",
        )

        logger.info(f"✓ W&B run initialized: {self.run_name}")

    def _log_single_result(self, result: InvoiceBenchmarkResult) -> None:
        """Log a single inference result to W&B."""
        wandb.log(
            {
                "per_invoice/latency_ms": result.latency_ms,
                "per_invoice/is_correct": 1 if result.is_correct else 0,
                "per_invoice/fallback_used": 1 if result.fallback_used else 0,
                "per_invoice/needs_human_review": 1 if result.needs_human_review else 0,
                "per_invoice/has_prediction": 1 if result.prediction else 0,
                "per_invoice/confidence": result.confidence if result.confidence else 0,
            }
        )

    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log summary metrics to W&B."""
        # Log all metrics
        wandb.summary.update(
            {
                "accuracy": metrics["accuracy"],
                "latency_mean_ms": metrics["latency_mean_ms"],
                "latency_median_ms": metrics["latency_median_ms"],
                "latency_p95_ms": metrics["latency_p95_ms"],
                "latency_p99_ms": metrics["latency_p99_ms"],
                "fallback_rate": metrics["fallback_rate"],
                "human_review_rate": metrics["human_review_rate"],
                "human_review_count": metrics["human_review_count"],
                "total_samples": metrics["total_samples"],
            }
        )

        # Log method-specific accuracies
        for method, acc in metrics["accuracy_by_method"].items():
            wandb.summary[f"accuracy_{method}"] = acc

        logger.info("✓ Summary metrics logged to W&B")

    def _create_visualizations(self, metrics: Dict[str, Any]) -> None:
        """Create and log visualizations to W&B."""
        # Histogram of latencies
        latencies = [r.latency_ms for r in self.results]
        wandb.log({"visualizations/latency_histogram": wandb.Histogram(latencies)})

        # Confidence histogram (if available)
        confidences = [r.confidence for r in self.results if r.confidence is not None]
        if confidences:
            wandb.log(
                {"visualizations/confidence_histogram": wandb.Histogram(confidences)}
            )

        # Method distribution pie chart (using table for simplicity)
        method_data = []
        for method, count in metrics["method_counts"].items():
            method_data.append(
                [method, count, metrics["accuracy_by_method"].get(method, 0)]
            )

        wandb.log(
            {
                "visualizations/method_breakdown": wandb.Table(
                    columns=["Method", "Count", "Accuracy"], data=method_data
                )
            }
        )

        # Predictions table (all results)
        pred_data = []
        for r in self.results:
            pred_data.append(
                [
                    r.file_name,
                    r.ground_truth or "N/A",
                    r.prediction or "N/A",
                    "Yes" if r.is_correct else "No",
                    f"{r.latency_ms:.1f}",
                    r.method_used,
                    "Yes" if r.needs_human_review else "No",
                ]
            )

        wandb.log(
            {
                "results/predictions_table": wandb.Table(
                    columns=[
                        "File",
                        "Ground Truth",
                        "Prediction",
                        "Correct",
                        "Latency (ms)",
                        "Method",
                        "Review",
                    ],
                    data=pred_data,
                )
            }
        )

        # Human review cases table
        human_review_cases = [r for r in self.results if r.needs_human_review]
        if human_review_cases:
            review_data = []
            for r in human_review_cases[:50]:  # First 50 cases
                review_data.append(
                    [
                        r.file_name,
                        r.ground_truth or "N/A",
                        r.prediction or "N/A",
                        len(r.prediction.split()) if r.prediction else 0,
                        "Yes" if r.is_correct else "No",
                        r.method_used,
                    ]
                )

            wandb.log(
                {
                    "results/human_review_cases": wandb.Table(
                        columns=[
                            "File",
                            "Ground Truth",
                            "Prediction",
                            "Word Count",
                            "Correct",
                            "Method",
                        ],
                        data=review_data,
                    )
                }
            )

        logger.info("✓ Visualizations created and logged")

    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print summary to console."""
        print("\n" + "=" * 70)
        print(f"BENCHMARK RESULTS: {self.run_name}")
        print("=" * 70)
        print("\n📊 Overall Metrics:")
        print(f"  Total Samples:     {metrics['total_samples']}")
        print(f"  Accuracy:          {metrics['accuracy']:.2%}")
        print(
            f"  Correct:           {metrics['correct_predictions']}/{metrics['total_samples']}"
        )

        print("\n⏱️  Latency Metrics:")
        print(f"  Mean:              {metrics['latency_mean_ms']:.2f} ms")
        print(f"  Median:            {metrics['latency_median_ms']:.2f} ms")
        print(f"  Std Dev:           {metrics['latency_std_ms']:.2f} ms")
        print(f"  P95:               {metrics['latency_p95_ms']:.2f} ms")
        print(f"  P99:               {metrics['latency_p99_ms']:.2f} ms")
        print(f"  Min:               {metrics['latency_min_ms']:.2f} ms")
        print(f"  Max:               {metrics['latency_max_ms']:.2f} ms")

        print("\n🔀 Method Breakdown:")
        for method, count in metrics["method_counts"].items():
            acc = metrics["accuracy_by_method"].get(method, 0)
            print(f"  {method:20s}: {count:4d} samples ({acc:.2%} accuracy)")

        print("\n🔄 Fallback Metrics:")
        print(f"  Fallback Count:    {metrics['fallback_count']}")
        print(f"  Fallback Rate:     {metrics['fallback_rate']:.2%}")

        print("\n🔍 Human Review Metrics:")
        print(f"  Review Count:      {metrics['human_review_count']}")
        print(f"  Review Rate:       {metrics['human_review_rate']:.2%}")
        print("  (Cases needing review: no prediction OR multiple words)")

        if metrics["confidence_mean"]:
            print("\n🎯 Confidence Metrics:")
            print(f"  Mean Confidence:   {metrics['confidence_mean']:.3f}")
            print(f"  Std Dev:           {metrics['confidence_std']:.3f}")

        print("\n" + "=" * 70)

        if self.wandb_run:
            print(f"\n🔗 View full results at: {self.wandb_run.url}")

        print()


def get_model(model_name: str, config: Optional[Dict] = None) -> BaseInvoiceModel:
    """
    Factory function to instantiate models by name.

    Args:
        model_name: Name of the model ('hybrid', 'layoutlmv3', etc.)
        config: Optional configuration dictionary

    Returns:
        Instantiated model
    """
    model_registry = {
        "hybrid": HybridModel,
        "layoutlmv3": LayoutLMv3Model,
        "onnx": OnnxModel,
        "openrouter": OpenRouterModel,
    }

    if model_name.lower() not in model_registry:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(model_registry.keys())}"
        )

    model_class = model_registry[model_name.lower()]
    return model_class(config)


def main():
    """Main entry point for benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Benchmark invoice extraction models with W&B logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark hybrid model (production setup)
  python benchmarks/benchmark.py --model hybrid --data-dir data/test --run-name "hybrid-baseline"
  
  # Benchmark LayoutLMv3 alone
  python benchmarks/benchmark.py --model layoutlmv3 --data-dir data/test --run-name "layoutlmv3-only"
  
  # Compare multiple models (run separately and compare in W&B)
  python benchmarks/benchmark.py --model hybrid --data-dir data/test --run-name "run-1" --tags v1 baseline
  python benchmarks/benchmark.py --model layoutlmv3 --data-dir data/test --run-name "run-2" --tags v1 model-only
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hybrid", "layoutlmv3", "onnx", "openrouter"],
        help="Model to benchmark",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing test data (JSON file)",
    )

    # Optional arguments
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this benchmark run (default: auto-generated)",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="invoice-extraction-benchmark",
        help="W&B project name",
    )

    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity (team/username)"
    )

    parser.add_argument(
        "--tags", nargs="+", default=[], help="Tags for organizing runs"
    )

    parser.add_argument(
        "--dataset-version", type=str, default="v2.1", help="Dataset version identifier"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Dataset split to use (test or train)",
    )

    parser.add_argument(
        "--offline", action="store_true", help="Run W&B in offline mode"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for model inference",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file or hosted model ID (required for ONNX, optional for others)",
    )

    args = parser.parse_args()

    # Create model config
    model_config = {
        "device": args.device,
        "model_path": args.model_path,
    }

    # Instantiate model
    logger.info(f"Initializing {args.model} model...")
    model = get_model(args.model, model_config)

    # Create benchmark
    benchmark = InvoiceBenchmark(
        model=model,
        data_dir=args.data_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        run_tags=args.tags,
        dataset_version=args.dataset_version,
        split=args.split,
        offline=args.offline,
    )

    # Run benchmark
    logger.info("Starting benchmark...")
    metrics = benchmark.run_benchmark()

    logger.info("✅ Benchmark complete!")

    return metrics


if __name__ == "__main__":
    main()
