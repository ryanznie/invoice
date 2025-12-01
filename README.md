# Invoice NER

Named Entity Recognition (NER) for invoice processing using LayoutLMv3 with LoRA fine-tuning. Extract invoice numbers and key information from invoice images.

## 📂 Repository Structure

```
invoice-ner/
├── app.py                      # Main FastAPI application
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker image definition
├── pyproject.toml              # Python project configuration & dependencies
├── setup.sh                    # Development environment setup script
├── .env.example                # Environment variables template
│
├── data/                       # Dataset and labeling tools
│   ├── app.py                  # Streamlit labeling application
│   ├── SROIE2019/              # Invoice dataset (train/test images & OCR)
│   ├── labels.json             # Training data labels
│   └── test_labels.json        # Test data labels
│
├── models/                     # Model files and checkpoints
│   └── layoutlmv3-lora-invoice-number/  # Fine-tuned LoRA adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── 01_heuristics.ipynb     # Heuristic-based extraction
│   ├── 02_labeling.ipynb       # Data labeling analysis
│   ├── 03_inference.ipynb      # Model inference testing
│   └── 04_postprocess.ipynb    # Post-processing experiments
│
├── scripts/                    # Utility scripts
│   ├── preprocess.py           # Data preprocessing utilities
│   └── train.py
│
├── docs/                       # Additional documentation
│   └── DEV_SETUP.md            # Developer setup guide
│
├── LICENSE                     # MIT License
└── README.md                   # This file                   
```

### Key Directories

- **`data/`** - Contains the SROIE2019 dataset and Streamlit labeling tool for annotating invoice images
- **`models/`** - Stores fine-tuned LoRA adapters and exported ONNX models for deployment
- **`notebooks/`** - Jupyter notebooks for experimentation, analysis, and prototyping
- **`scripts/`** - Utility scripts for data preprocessing, model export, and deployment preparation
- **`docs/`** - Additional documentation for ONNX export and model variants

## 🚀 Quick Start

### Run with Docker (Recommended)

```bash
# 1. Copy environment file (optional)
cp .env.example .env
# Edit .env to customize settings (port, log level, etc.)

# 2. Build and start
docker-compose up -d --build

# 3. Check logs
docker-compose logs -f

# 4. Open browser
open http://localhost:7860

# 5. Stop when done
docker-compose down
```

### Run Locally

```bash
# 1. Set up virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Copy environment file
cp .env.example .env

# 3. Install dependencies
uv pip install -e .

# 4. Run the app (automatically loads .env)
python app.py

# 5. Open browser
open http://localhost:7860
```

## 📋 Prerequisites

- **Docker** (>= 20.10) and Docker Compose (>= 2.0) - for containerized deployment
- **Python** (>= 3.10) - for local development
- **uv** - fast Python package installer ([installation guide](https://github.com/astral-sh/uv))
- **8GB RAM** minimum (16GB recommended)
- **Model files** in `models/layoutlmv3-lora-invoice-number/`

## 📁 Required Files

Ensure these exist before running:
```
models/
└── layoutlmv3-lora-invoice-number/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ... (other config files)
```

## ✅ Verify Installation

```bash
# Check health endpoint
curl http://localhost:7860/health

# Expected response:
# {"status": "healthy", "model_loaded": true, "device": "cpu"}
```

## 🔧 Configuration

### Using .env File (Recommended)

The easiest way to configure the application:

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to customize settings:
   ```bash
   # Example: Enable debug logging
   LOG_LEVEL=DEBUG
   
   # Example: Change port
   PORT=8080
   
   # Example: Use Apple MPS
   DEVICE=mps
   ```

3. Start the application (automatically loads `.env`):
   ```bash
   docker-compose up -d
   ```

### Available Environment Variables

Key variables (see `.env.example` for all options):

- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default: `INFO`
- `DEVICE`: Device to run on (`cpu`, `cuda`, or `mps`). Default: `cpu`
- `PORT`: Port to expose. Default: `7860`
- `MODEL_PATH`: Path to model directory. Default: `models/layoutlmv3-lora-invoice-number`
- `DOCKER_CPU_LIMIT`: CPU cores limit. Default: `4`
- `DOCKER_MEMORY_LIMIT`: Memory limit. Default: `8G`

### Command Line Override

Override `.env` values from the command line:

```bash
# Override port
PORT=9000 python app.py

# Override multiple variables
LOG_LEVEL=DEBUG DEVICE=cpu PORT=8080 python app.py

# Docker Compose
PORT=9000 docker-compose up
```

## 🐳 Docker Deployment

### Basic Commands

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Resource Configuration

Adjust resource limits in `docker-compose.yml` or `.env`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

Or in `.env`:
```bash
DOCKER_CPU_LIMIT=4
DOCKER_MEMORY_LIMIT=8G
```

### Port Configuration

Change the exposed port in `docker-compose.yml`:

```yaml
ports:
  - "8080:7860"  # Map host port 8080 to container port 7860
```

Or in `.env`:
```bash
PORT=8080
```

## 📚 API Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:7860/docs
- **Health check**: http://localhost:7860/health

## 🛠️ Development

For development setup, data labeling, and model training, see [docs/DEV_SETUP.md](docs/DEV_SETUP.md).

## 📄 License

MIT License

Copyright (c) 2025 Ryan Nie