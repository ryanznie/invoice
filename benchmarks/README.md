# Invoice Extraction Benchmarking

Benchmark invoice number extraction models with W&B tracking for accuracy, latency, fallback rate, and human review metrics.

## Installation

```bash
# Dependencies already in pyproject.toml
uv pip install -e .

# Login to W&B (or use --offline flag)
wandb login
```

## Usage

```bash
# Benchmark hybrid model (heuristics + LayoutLMv3 fallback)
python benchmarks/benchmark.py \
  --model hybrid \
  --data-dir data/train \
  --run-name "layoutlmv3-lora-heuristics-train-mps" \
  --device mps \
  --split train

# Benchmark LayoutLMv3 only
python benchmarks/benchmark.py \
  --model layoutlmv3 \
  --data-dir data/test \ 
  --run-name "layoutlmv3-lora-invoice-number-mps" \      
  --device mps

# Run offline (no W&B)
python benchmarks/benchmark.py \
  --model hybrid \
  --data-dir data/test \
  --offline
```

## Available Models

- **`hybrid`** - Heuristics first, LayoutLMv3 fallback (recommended)
- **`layoutlmv3`** - LayoutLMv3 model only

## Tracked Metrics

**Per-Invoice:**
- Prediction vs ground truth (correctness)
- Latency (ms)
- Method used (heuristic/model/fallback)
- Confidence score
- Human review flag (no prediction OR multiple words)

**Aggregate:**
- Accuracy, latency (mean/P95/P99)
- Fallback rate, human review rate
- Method breakdown

## Command-Line Arguments

**Required:**
- `--model` - Model to benchmark (`hybrid`, `layoutlmv3`)
- `--data-dir` - Path to data directory

**Optional:**
- `--split` - Dataset split (`test` or `train`, default: `test`)
- `--run-name` - Name for this run
- `--wandb-project` - W&B project name
- `--tags` - Tags for organizing runs (space-separated)
- `--device` - Device (`cpu`, `cuda`, `mps`)
- `--offline` - Run without W&B sync


## Data Format

Expected directory structure:

```
data/SROIE2019/
├── test/  (or train/)
    ├── img/               # Invoice images (.jpg)
    ├── box/               # OCR txt files (x1,y1,x2,y2,x3,y3,x4,y4,text format)
    └── test_labels.json   # Ground truth {"file.jpg": "invoice_number"}
```

**Note:** Benchmark reads OCR data from `.txt` files in `box/` directory, matching the API/notebook approach.

Ground truth format:
```json
{
  "X001.jpg": "INV-123456",
  "X002.jpg": "ambiguous"  // automatically skipped
}
```
