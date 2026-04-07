#!/usr/bin/env python3
"""
Setup Triton Inference Server Model Repository for LayoutLMv3

This script creates the necessary directory structure and configuration files
for serving the LayoutLMv3 ONNX model with Triton Inference Server.
"""

import argparse
import shutil
from pathlib import Path
import onnx
import sys


def create_model_repo(
    model_path: str, repo_path: str, model_name: str, config_name: str = "config.pbtxt"
):
    """Create Triton model repository structure"""
    print(f"🔧 Setting up Triton Model Repository for: {model_name}")

    source_path = Path(model_path)
    if not source_path.exists():
        print(f"❌ Error: Model file not found at {source_path}")
        sys.exit(1)

    repo_dir = Path(repo_path)
    model_dir = repo_dir / model_name
    version_dir = model_dir / "1"

    # Create directories
    version_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created directory structure: {version_dir}")

    # Copy model
    dest_path = version_dir / "model.onnx"
    shutil.copy2(source_path, dest_path)
    print(f"  ✓ Copied model to: {dest_path}")

    # Generate config.pbtxt
    generate_config(source_path, model_dir / config_name, model_name)


def generate_config(model_path: Path, config_path: Path, model_name: str):
    """Generate config.pbtxt based on ONNX model properties"""
    print("📝 Generating Triton configuration...")

    # Load ONNX model to inspect inputs/outputs
    model = onnx.load(str(model_path))

    # Basic configuration
    config_lines = [
        f'name: "{model_name}"',
        'platform: "onnxruntime_onnx"',
        "max_batch_size: 8",  # Adjust as needed
        "",
        "dynamic_batching {",
        "  preferred_batch_size: [ 4, 8 ]",
        "  max_queue_delay_microseconds: 100",
        "}",
        "",
    ]

    # Inputs
    for input_tensor in model.graph.input:
        name = input_tensor.name
        # Skip batch dim for Triton config if max_batch_size > 0
        dims = [
            d.dim_value if d.dim_value > 0 else -1
            for d in input_tensor.type.tensor_type.shape.dim
        ]

        # Handle dynamic batch dimension (usually the first one)
        # In Triton with max_batch_size > 0, we exclude the batch dimension from the config shape
        if len(dims) > 0:
            dims = dims[1:]

        data_type = mapping_onnx_type_to_triton(input_tensor.type.tensor_type.elem_type)

        config_lines.append("input {")
        config_lines.append(f'  name: "{name}"')
        config_lines.append(f"  data_type: {data_type}")
        config_lines.append(f"  dims: {dims}")
        config_lines.append("}")
    config_lines.append("")

    # Outputs
    for output_tensor in model.graph.output:
        name = output_tensor.name
        dims = [
            d.dim_value if d.dim_value > 0 else -1
            for d in output_tensor.type.tensor_type.shape.dim
        ]

        if len(dims) > 0:
            dims = dims[1:]

        data_type = mapping_onnx_type_to_triton(
            output_tensor.type.tensor_type.elem_type
        )

        config_lines.append("output {")
        config_lines.append(f'  name: "{name}"')
        config_lines.append(f"  data_type: {data_type}")
        config_lines.append(f"  dims: {dims}")
        config_lines.append("}")
    # config_lines.append("]") <-- Removed list syntax

    # Write config
    with open(config_path, "w") as f:
        f.write("\n".join(config_lines))

    print(f"  ✓ Configuration saved to: {config_path}")


def mapping_onnx_type_to_triton(onnx_type):
    """Map ONNX data types to Triton data types"""
    # https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes
    type_map = {
        1: "TYPE_FP32",  # FLOAT
        2: "TYPE_UINT8",  # UINT8
        3: "TYPE_INT8",  # INT8
        4: "TYPE_UINT16",  # UINT16
        5: "TYPE_INT16",  # INT16
        6: "TYPE_INT32",  # INT32
        7: "TYPE_INT64",  # INT64
        9: "TYPE_BOOL",  # BOOL
        10: "TYPE_FP16",  # FLOAT16
        11: "TYPE_FP64",  # DOUBLE
    }
    return type_map.get(onnx_type, "TYPE_FP32")  # Default to FP32


def main():
    parser = argparse.ArgumentParser(description="Setup Triton Model Repository")
    parser.add_argument("--model_path", required=True, help="Path to source ONNX model")
    parser.add_argument(
        "--repo_dir",
        default="triton_model_repo",
        help="Path to Triton model repository",
    )
    parser.add_argument(
        "--model_name",
        default="layoutlmv3-lora-invoice-number",
        help="Name of the model in Triton",
    )

    args = parser.parse_args()

    create_model_repo(args.model_path, args.repo_dir, args.model_name)

    print("\n✅ Triton repository setup complete!")
    print(f"  Repository path: {Path(args.repo_dir).absolute()}")
    print("  To start Triton:")
    print(
        f"  docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v {Path(args.repo_dir).absolute()}:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models"
    )


if __name__ == "__main__":
    main()
