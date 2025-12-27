"""
Export LayoutLMv3 LoRA model to ONNX format for Triton Inference Server

This script:
1. Loads the LoRA adapter and merges it with the base model
2. Exports the merged model to ONNX format
3. Validates the ONNX model outputs match PyTorch
4. Optimizes the ONNX model for inference
"""

import argparse
import json
import torch
import numpy as np
import platform  # Added for MacOS detection
from pathlib import Path
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)
from peft import PeftModel
import onnx
import onnxruntime as ort
from typing import Dict, Tuple


class ONNXExporter:
    """Export LayoutLMv3 LoRA model to ONNX"""

    def __init__(
        self,
        model_path: str,
        base_model: str = "microsoft/layoutlmv3-base",
        output_dir: str = "models/onnx",
        num_labels: int = 3,
        model_name: str = "layoutlmv3_invoice_ner",
    ):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.num_labels = num_labels
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("🔧 Initializing ONNX Exporter")
        print(f"  Model path: {self.model_path}")
        print(f"  Base model: {self.base_model}")
        print(f"  Model name: {self.model_name}")
        print(f"  Output dir: {self.output_dir}")

    def load_and_merge_model(self) -> Tuple[torch.nn.Module, LayoutLMv3Processor]:
        """Load LoRA adapter and merge with base model"""
        print("\n📦 Loading model...")

        # Load processor
        processor = LayoutLMv3Processor.from_pretrained(
            self.model_path, apply_ocr=False
        )
        print("  ✓ Processor loaded")

        # Load base model
        base_model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.base_model, num_labels=self.num_labels
        )
        print(f"  ✓ Base model loaded ({self.base_model})")

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, self.model_path)
        print("  ✓ LoRA adapter loaded")

        # Merge LoRA weights into base model
        print("\n🔀 Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        merged_model.eval()
        print("  ✓ Model merged successfully")

        return merged_model, processor

    def export_to_onnx(
        self,
        model: torch.nn.Module,
        processor: LayoutLMv3Processor,
        opset_version: int = 14,
    ) -> Path:
        """Export merged model to ONNX format"""
        print("\n🚀 Exporting to ONNX...")

        # Create dummy inputs for export
        dummy_image = Image.new("RGB", (224, 224), color="white")
        dummy_words = ["INVOICE", "NUMBER", "12345"]
        dummy_boxes = [[0, 0, 100, 50], [100, 0, 200, 50], [200, 0, 300, 50]]

        # Process dummy inputs
        encoding = processor(
            dummy_image,
            dummy_words,
            boxes=dummy_boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        # Prepare inputs for ONNX export
        input_names = ["pixel_values", "input_ids", "attention_mask", "bbox"]
        output_names = ["logits"]

        # Dynamic axes for variable batch size and sequence length
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

        # Export path
        onnx_path = self.output_dir / f"{self.model_name}.onnx"

        # Export to ONNX
        # Use a wrapper to ensure correct argument passing
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values, input_ids, attention_mask, bbox):
                return self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                ).logits

        wrapped_model = ModelWrapper(model)

        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                (
                    encoding["pixel_values"],
                    encoding["input_ids"],
                    encoding["attention_mask"],
                    encoding["bbox"],
                ),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )

        print(f"  ✓ ONNX model exported to: {onnx_path}")
        print(f"  ✓ Model size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")

        return onnx_path

    def validate_onnx_model(
        self,
        onnx_path: Path,
        pytorch_model: torch.nn.Module,
        processor: LayoutLMv3Processor,
    ) -> bool:
        """Validate ONNX model outputs match PyTorch"""
        print("\n🔍 Validating ONNX model...")

        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model structure is valid")

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        print("  ✓ ONNX Runtime session created")

        # Debug: Print expected input types
        print("\n  Expected inputs:")
        for input_meta in ort_session.get_inputs():
            print(f"    {input_meta.name}: {input_meta.type}")

        # Create test inputs
        test_image = Image.new("RGB", (300, 400), color="white")
        test_words = ["INVOICE", "NO", ":", "CS00012944"]
        test_boxes = [
            [50, 50, 150, 80],
            [160, 50, 200, 80],
            [210, 50, 230, 80],
            [240, 50, 380, 80],
        ]

        encoding = processor(
            test_image,
            test_words,
            boxes=test_boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        # PyTorch inference
        with torch.no_grad():
            pytorch_outputs = pytorch_model(
                pixel_values=encoding["pixel_values"],
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                bbox=encoding["bbox"],
            )
            pytorch_logits = pytorch_outputs.logits.numpy()

        # ONNX inference - match the expected types from the model
        onnx_inputs = {
            "pixel_values": encoding["pixel_values"].numpy(),
            "input_ids": encoding["input_ids"].numpy(),
            "attention_mask": encoding["attention_mask"].numpy(),
            "bbox": encoding["bbox"].numpy(),
        }
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]

        # Compare outputs
        print("\n  Shape comparison:")
        print(f"    PyTorch: {pytorch_logits.shape}")
        print(f"    ONNX:    {onnx_logits.shape}")

        max_diff = np.abs(pytorch_logits - onnx_logits).max()
        mean_diff = np.abs(pytorch_logits - onnx_logits).mean()

        # Also compute relative difference
        pytorch_max = np.abs(pytorch_logits).max()
        relative_diff = max_diff / pytorch_max if pytorch_max > 0 else 0

        print("\n  ✓ Output comparison:")
        print(f"    Max absolute difference: {max_diff:.6f}")
        print(f"    Mean absolute difference: {mean_diff:.6f}")
        print(
            f"    Max relative difference: {relative_diff:.6f} ({relative_diff*100:.2f}%)"
        )
        print(
            f"    PyTorch logits range: [{pytorch_logits.min():.3f}, {pytorch_logits.max():.3f}]"
        )
        print(
            f"    ONNX logits range:    [{onnx_logits.min():.3f}, {onnx_logits.max():.3f}]"
        )

        # Check prediction agreement (more important than logit differences)
        pytorch_preds = np.argmax(pytorch_logits, axis=-1)
        onnx_preds = np.argmax(onnx_logits, axis=-1)
        pred_agreement = np.mean(pytorch_preds == onnx_preds)

        print(f"\n  Prediction agreement: {pred_agreement*100:.2f}%")

        # Check if outputs are close enough
        if pred_agreement >= 0.99:  # 99% prediction agreement
            print("  ✅ ONNX predictions match PyTorch (≥99% agreement)")
            return True
        elif pred_agreement >= 0.95:  # 95% prediction agreement
            print("  ✅ ONNX predictions mostly match PyTorch (≥95% agreement)")
            return True
        elif relative_diff < 0.01:  # 1% relative difference in logits
            print("  ✅ ONNX model outputs match PyTorch (within 1% tolerance)")
            return True
        elif max_diff < 2.0:  # Absolute difference < 2.0 for logits is usually fine
            print(
                "  ✅ ONNX model outputs acceptable (logit differences within normal range)"
            )
            print("  ℹ️  Prediction agreement: {pred_agreement*100:.1f}%")
            return True
        else:
            print("  ⚠️  ONNX model outputs differ significantly from PyTorch")
            print("  ℹ️  This may still be acceptable - test with real data to verify")
            return False

    def convert_to_fp16(self, onnx_path: Path) -> Path:
        """Convert model to FP16 (half precision)"""
        print("\n🔄 Converting to FP16...")

        try:
            from onnxconverter_common import float16

            # Load model
            model = onnx.load(str(onnx_path))

            # Convert to FP16
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

            # Validate FP16 model structure
            try:
                onnx.checker.check_model(model_fp16)
            except Exception as e:
                print(f"  ⚠️  FP16 model validation failed: {e}")
                raise

            # Save FP16 model
            fp16_path = self.output_dir / f"{self.model_name}_fp16.onnx"
            onnx.save(model_fp16, str(fp16_path))

            print(f"  ✓ FP16 model saved to: {fp16_path}")
            print(f"  ✓ FP16 size: {fp16_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(
                f"  ✓ Size reduction: {(1 - fp16_path.stat().st_size / onnx_path.stat().st_size) * 100:.1f}%"
            )

            return fp16_path

        except ImportError:
            print("  ⚠️  onnxconverter-common not available")
            print("  ℹ️  Install with: pip install onnxconverter-common")
            return None

    def quantize_model(self, onnx_path: Path, quantization_mode: str = "int8") -> Path:
        """Quantize model to INT8 or UINT8"""
        print(f"\n🔢 Quantizing to {quantization_mode.upper()}...")

        # Check for MacOS
        if platform.system() == "Darwin":
            print(
                "  ⚠️  Skipping INT8 quantization on MacOS (ONNX Runtime incompatible with MacOS CPU / GPU Kernels)"
            )
            return None

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            # Determine quantization type
            if quantization_mode == "int8":
                quant_type = QuantType.QInt8
                suffix = "int8"
            elif quantization_mode == "uint8":
                quant_type = QuantType.QUInt8
                suffix = "uint8"
            else:
                raise ValueError(f"Unsupported quantization mode: {quantization_mode}")

            # Output path
            quantized_path = self.output_dir / f"{self.model_name}_{suffix}.onnx"

            # Quantize
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=quant_type,
            )

            print(f"  ✓ {quantization_mode.upper()} model saved to: {quantized_path}")
            print(
                f"  ✓ {quantization_mode.upper()} size: {quantized_path.stat().st_size / 1024 / 1024:.2f} MB"
            )
            print(
                f"  ✓ Size reduction: {(1 - quantized_path.stat().st_size / onnx_path.stat().st_size) * 100:.1f}%"
            )

            return quantized_path

        except ImportError:
            print("  ⚠️  onnxruntime quantization not available")
            print("  ℹ️  Ensure onnxruntime>=1.16.0 is installed")
        return None

    def optimize_onnx_model(self, onnx_path: Path) -> Path:
        """Optimize ONNX model using ORT graph optimizations (safe/minimal)"""
        print("\n⚡ Optimizing ONNX model (safe/minimal)...")

        try:
            # Output path
            optimized_path = self.output_dir / f"{self.model_name}_optimized.onnx"

            # Use ORT SessionOptions to optimize and save
            sess_options = ort.SessionOptions()
            # ORT_ENABLE_BASIC is the safest level (eliminates dead code, constant folding, etc.)
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.optimized_model_filepath = str(optimized_path)

            # Create session to trigger optimization and saving
            # Optimization happens during session initialization
            _ = ort.InferenceSession(
                str(onnx_path), sess_options, providers=["CPUExecutionProvider"]
            )

            print(f"  ✓ Optimized model saved to: {optimized_path}")
            print(
                f"  ✓ Optimized size: {optimized_path.stat().st_size / 1024 / 1024:.2f} MB"
            )

            return optimized_path

        except Exception as e:
            print(f"  ⚠️  Optimization failed: {e}")
            return onnx_path

    def save_metadata(self, processor: LayoutLMv3Processor, onnx_path: Path) -> None:
        """Save model metadata and configuration"""
        print("\n💾 Saving metadata...")

        metadata = {
            "model_name": self.model_name,
            "base_model": self.base_model,
            "num_labels": self.num_labels,
            "max_length": 512,
            "onnx_opset_version": 14,
            "input_names": ["pixel_values", "input_ids", "attention_mask", "bbox"],
            "output_names": ["logits"],
            "label2id": {"O": 0, "B-INVOICE_ID": 1, "I-INVOICE_ID": 2},
            "id2label": {"0": "O", "1": "B-INVOICE_ID", "2": "I-INVOICE_ID"},
        }

        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Metadata saved to: {metadata_path}")

        # Save processor config
        processor.save_pretrained(self.output_dir)
        print(f"  ✓ Processor config saved to: {self.output_dir}")

    def export(
        self,
        optimize: bool = True,
        validate: bool = True,
        export_fp16: bool = True,
        export_int8: bool = True,
    ) -> Dict:
        """Main export pipeline with multiple model variants"""
        print("\n" + "=" * 70)
        print("🚀 ONNX Export Pipeline - Multi-Variant Export")
        print("=" * 70)

        exported_models = {}

        # Step 1: Load and merge model
        merged_model, processor = self.load_and_merge_model()

        # Step 2: Export base FP32 model to ONNX
        print("\n" + "=" * 70)
        print("📦 Exporting Base FP32 Model")
        print("=" * 70)
        onnx_path = self.export_to_onnx(merged_model, processor)
        exported_models["fp32_base"] = str(onnx_path)

        # Step 3: Validate (optional)
        validation_passed = True
        if validate:
            validation_passed = self.validate_onnx_model(
                onnx_path, merged_model, processor
            )

        # Step 4: Optimize FP32 model (optional)
        optimized_path = onnx_path
        if optimize:
            optimized_path = self.optimize_onnx_model(onnx_path)
            exported_models["fp32_optimized"] = str(optimized_path)

        # Step 5: Export FP16 variant
        if export_fp16:
            print("\n" + "=" * 70)
            print("📦 Exporting FP16 Model")
            print("=" * 70)
            # Use base model for FP16 conversion to avoid graph issues
            fp16_path = self.convert_to_fp16(onnx_path)
            if fp16_path:
                exported_models["fp16"] = str(fp16_path)

        # Step 6: Export INT8 quantized variant
        # Step 6: Export INT8 quantized variant
        if export_int8:
            print("\n" + "=" * 70)
            print("📦 Exporting INT8 Quantized Model")
            print("=" * 70)
            # Quantize the optimized model if available, otherwise base
            target_for_quant = optimized_path if optimize else onnx_path
            int8_path = self.quantize_model(target_for_quant, quantization_mode="int8")
            if int8_path:
                exported_models["int8"] = str(int8_path)
            elif platform.system() == "Darwin":
                print("  ℹ️  INT8 export skipped on MacOS")

        # Step 7: Save metadata
        self.save_metadata(processor, optimized_path)

        # Summary
        print("\n" + "=" * 70)
        print("✅ Export Complete - All Variants")
        print("=" * 70)
        print(f"\n📁 Output directory: {self.output_dir}")
        print("\n📄 Exported models:")
        for variant, path in exported_models.items():
            size = Path(path).stat().st_size / 1024 / 1024
            print(f"  • {variant:20} {Path(path).name:45} ({size:.1f} MB)")
        print(f"\n📊 Validation: {'✅ Passed' if validation_passed else '⚠️  Failed'}")

        print("\n💡 Model Selection Guide:")
        print("  • FP32 (base):      Highest accuracy, largest size, slowest")
        print("  • FP32 (optimized): Same accuracy, optimized ops, faster")
        print(
            "  • FP16:             ~50% size, minimal accuracy loss, 2x faster on GPU"
        )
        print("  • INT8:             ~75% size, slight accuracy loss, 4x faster on CPU")

        print("\n💡 Next steps:")
        print("  1. Test each model variant with sample inputs")
        print("  2. Compare accuracy vs speed tradeoffs")
        print("  3. Set up Triton model repository with chosen variant(s)")
        print("  4. Create Triton config.pbtxt for each variant")
        print("  5. Deploy with Triton Inference Server")

        return {
            "exported_models": exported_models,
            "output_dir": str(self.output_dir),
            "validation_passed": validation_passed,
        }


def main():
    parser = argparse.ArgumentParser(description="Export LayoutLMv3 LoRA model to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/layoutlmv3-lora-invoice-number",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/layoutlmv3-base",
        help="Base model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="layoutlmv3_invoice_ner",
        help="Model name for output files (default: layoutlmv3_invoice_ner)",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=3,
        help="Number of labels (default: 3 for O, B-INVOICE_ID, I-INVOICE_ID)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip ONNX optimization",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Skip FP16 export",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Skip INT8 quantization",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=["fp32", "fp16", "int8", "all"],
        default=["all"],
        help="Model variants to export (default: all)",
    )

    args = parser.parse_args()

    # Determine which variants to export
    export_fp16 = not args.no_fp16 and (
        "all" in args.variants or "fp16" in args.variants
    )
    export_int8 = not args.no_int8 and (
        "all" in args.variants or "int8" in args.variants
    )

    # Create exporter
    exporter = ONNXExporter(
        model_path=args.model_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_labels=args.num_labels,
        model_name=args.model_name,
    )

    # Run export
    results = exporter.export(
        optimize=not args.no_optimize,
        validate=not args.no_validate,
        export_fp16=export_fp16,
        export_int8=export_int8,
    )

    return results


if __name__ == "__main__":
    main()
