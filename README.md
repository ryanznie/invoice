# Invoice NER

Named Entity Recognition (NER) for invoice processing using LayoutLMv3 with LoRA fine-tuning. Extract invoice numbers and key information from invoice images.

## ✨ Features

- 🤖 **Hybrid Extraction Pipeline** - Combines fast heuristic pattern matching with deep learning fallback
- 🎯 **LayoutLMv3 with LoRA** - Efficient fine-tuning on multimodal document understanding
- 🌐 **Dual Interface** - REST API for programmatic access + Gradio UI for interactive use
- 🚀 **Production Ready** - Comprehensive test suite (107 tests), Docker support, health checks
- 📊 **Multi-Format Support** - Accepts TXT and JSON OCR data formats
- 🔧 **Device Flexible** - Runs on CPU, CUDA (NVIDIA), or MPS (Apple Silicon)
- 📝 **Interactive Docs** - Auto-generated Swagger/ReDoc API documentation

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
│   ├── scripts/                # Data processing utilities
│   │   ├── create_dataframe.py # Creates DataFrame from labeled data
│   │   └── validate_labels.py  # Validates label quality
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
├── src/                        # Core application modules
│   ├── __init__.py              # Package initialization
│   ├── api.py                   # FastAPI endpoints
│   ├── gradio_ui.py             # Gradio interface
│   ├── inference.py             # Model inference logic
│   ├── heuristics.py            # Pattern-based extraction
│   ├── postprocessing.py        # Result postprocessing
│   ├── validation.py            # Input validation
│   └── utils.py                 # Utility functions
│
├── docs/                       # Additional documentation
│   ├── API_USAGE.md             # Complete API documentation and examples
│   ├── DEV_SETUP.md             # Developer setup guide
│   └── TESTING.md               # Testing guide and validation
│
├── tests/                      # Test suite
│   ├── conftest.py             # Shared test fixtures
│   ├── test_app.py             # Application tests
│   ├── test_scripts.py         # Script tests
│   ├── test_api.py             # API endpoint tests
│   └── README.md               # Testing documentation
│
├── LICENSE                     # MIT License
└── README.md                   # This file                   
```

### Key Directories

- **`src/`** - Core application modules (API endpoints, inference, UI, validation, utilities)
- **`data/`** - Contains the SROIE2019 dataset and Streamlit labeling tool for annotating invoice images
- **`models/`** - Stores fine-tuned LoRA adapters and exported ONNX models for deployment
- **`notebooks/`** - Jupyter notebooks for experimentation, analysis, and prototyping
- **`scripts/`** - Utility scripts for data preprocessing, model export, and deployment preparation
- **`tests/`** - Comprehensive test suite with 107 tests for production validation
- **`docs/`** - Documentation for API usage, development setup, testing, and deployment

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

### Quick API Test

```bash
# Extract invoice number from an invoice
curl -X POST http://localhost:7860/predict \
  -F "image=@path/to/invoice.jpg" \
  -F "ocr_file=@path/to/ocr_data.json"

# Response:
# {
#   "invoice_number": "INV-2023-001234",
#   "extraction_method": "heuristic",
#   "total_words": 127,
#   "model_device": "cpu"
# }
```

For detailed API documentation with code examples in Python, JavaScript, and more, see **[docs/API_USAGE.md](docs/API_USAGE.md)**.

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

The application provides both a **Gradio web interface** and a **REST API**:

### Web Interface (Gradio)
- **URL**: http://localhost:7860/
- **Features**: Drag-and-drop upload, visual preview, no coding required
- **Best for**: Manual testing, demos, non-technical users

### REST API
- **Interactive docs**: http://localhost:7860/docs (Swagger UI)
- **Alternative docs**: http://localhost:7860/redoc (ReDoc)
- **Health check**: http://localhost:7860/health

**Detailed API Guide**: See [docs/API_USAGE.md](docs/API_USAGE.md) for:
- Complete endpoint documentation
- Request/response formats
- Code examples in Python, JavaScript, cURL
- Error handling and best practices

## 🛠️ Development

For development setup, data labeling, and model training, see [docs/DEV_SETUP.md](docs/DEV_SETUP.md). For detailed testing documentation, see [docs/TESTING.md](docs/TESTING.md).

## 📄 License

MIT License

Copyright (c) 2025 Ryan Nie