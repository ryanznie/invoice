# Developer Setup Guide

This guide covers setting up the development environment, data labeling, model training, and advanced configuration.

## 🛠️ Development Environment Setup

### Prerequisites

- Python >= 3.10
- Git
- `uv` package manager (recommended) or `pip`

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:ryanznie/invoice.git
   cd invoice
   ```

2. **Run the setup script:**
   
   This will create a virtual environment, install all necessary dependencies using `uv`, and set up pre-commit hooks.
   ```bash
   bash setup.sh
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install uv
pip install uv

# Install dependencies
uv pip install -e .

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

### Development Dependencies

```bash
# Install with development extras
uv pip install -e ".[dev]"

# Or install specific tools
uv pip install pytest black ruff mypy
```

## Code Review Automation

Greptile is configured through the root `.greptile/` directory:

- `.greptile/config.json` defines review triggers, status checks, ignored generated artifacts, and structured project rules.
- `.greptile/rules.md` gives Greptile review guidance for API stability, invoice data handling, ML behavior, and deployment consistency.
- `.greptile/files.json` points Greptile at repo docs it should use as review context.

To activate reviews, install/connect the Greptile GitHub App for this GitHub account or organization, grant it access to this repository, and enable the repository for review in the Greptile dashboard. Once enabled, new pull requests are reviewed automatically; add the `no-greptile`, `skip-greptile`, or `wip-*` label to skip automated review.

## 📊 Data Labeling

### Dataset Setup

1. **Download the SROIE2019 dataset:**
   
   Download from [Kaggle](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2?resource=download) and unpack in `data/`.

2. **Verify directory structure:**
   ```
   data/
   └── SROIE2019/
       ├── train/
       │   ├── img/      # Contains .jpg images
       │   │   ├── X00016469612.jpg
       │   │   └── ...
       │   └── box/      # Contains .txt files with extracted text
       │       ├── X00016469612.txt
       │       └── ...
       └── test/
           ├── img/
           └── box/
   ```

### Running the Labeling Tool

The Streamlit labeling application helps you label and correct text extracted from invoice images.

1. **Install Streamlit dependencies:**
   ```bash
   uv pip install streamlit pillow
   ```

2. **Run the labeling app:**
   ```bash
   cd data
   streamlit run app.py
   ```

### Using the Labeling Tool

#### Basic Navigation

- The application displays an image on the left and a text box on the right
- Current file name and position (e.g., "X00016469612.jpg (1/626)") shown at top

#### Editing Labels

1. **Review the image**: Look at the invoice/receipt image
2. **Edit the text**: Enter the invoice number or ID from the image
   - Simple invoice numbers: `7030F715`
   - Complex multi-part IDs: `CS-SA-0096677` or `18124/102/T0146`
   - Unclear/ambiguous: Enter `ambiguous` (automatically logged to `ambiguous_edits.log`)
3. **Save your work**: Click **💾 Save and Next**

#### Navigation Controls

- **⬅️ Previous**: Go to previous image
- **Next ➡️**: Go to next image without saving
- **💾 Save and Next**: Save current label and move to next
- **Go to page**: Jump directly to a page number (1-based)
- **Go to file**: Jump to specific filename (e.g., `X00016469612.jpg`)

#### Mode Selection (Sidebar)

1. **Dataset Mode**:
   - **Train**: Work with training data (`SROIE2019/train/`)
   - **Test**: Work with test data (`SROIE2019/test/`)

2. **Filter by Label**:
   - **All**: Show all images
   - **Ambiguous**: Show only images labeled as "ambiguous"

### Output Files

Labels are saved in JSON files in the `data/` directory:

- **`labels.json`**: Training data labels
- **`test_labels.json`**: Test data labels
- **`ambiguous_edits.log`**: Log of ambiguous label edits

**Example `labels.json`:**
```json
{
    "X00016469612.jpg": "7030F715",
    "X00016469613.jpg": "CS-SA-0096677",
    "X00016469614.jpg": "ambiguous"
}
```

### Dataset Documentation

- **Dataset documentation**: [Notion document](https://www.notion.so/Dataset-Documentation-Notes-1609faffd568479dbaf1c072b23c472d)
- **Labeled data releases**:
  - [Kaggle](https://www.kaggle.com/datasets/ryanznie/sroie-datasetv2-with-labels)
  - [HuggingFace](https://huggingface.co/datasets/ryanznie/SROIE_2019_with_labels)
- **Labeling heuristics**: [Notion documentation](https://www.notion.so/Heuristics-Details-53af761344b7402fac834031244e032a#27ac697d927c8087ab97ebfbb0d23a38)

## 🔧 Advanced Configuration

### Environment Variables Reference

All configuration is managed through environment variables. Copy `.env.example` to `.env` and customize as needed.

#### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

#### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cpu` | Inference device (cpu, cuda, mps) |
| `MODEL_PATH` | `models/layoutlmv3-lora-invoice-number` | Model directory path |
| `BASE_MODEL` | `microsoft/layoutlmv3-base` | Base model identifier |
| `MAX_LENGTH` | `512` | Maximum sequence length for model input |
| `NUM_LABELS` | `3` | Number of NER labels (O, B-INVOICE_NUMBER, I-INVOICE_NUMBER) |
| `GOOGLE_API_KEY` | `""` | API Key for Gemini fallback (uses `gemini-2.5-flash`) |

#### Server Configuration

> [!NOTE]
> **Gemini Fallback**: If the primary local model fails, the system automatically attempts to extract the invoice number using Google's `gemini-2.5-flash` model. This requires the `GOOGLE_API_KEY` environment variable to be set.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Port to run the application on |
| `HOST` | `0.0.0.0` | Host to bind to (0.0.0.0 = all interfaces, 127.0.0.1 = localhost only) |
| `DEBUG` | `false` | Enable debug mode (auto-reload on code changes) |

#### Docker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_CPU_LIMIT` | `4` | CPU cores limit |
| `DOCKER_CPU_RESERVATION` | `2` | CPU cores reservation |
| `DOCKER_MEMORY_LIMIT` | `8G` | Memory limit |
| `DOCKER_MEMORY_RESERVATION` | `4G` | Memory reservation |

#### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMEOUT` | `60` | Request timeout (in seconds) |

#### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Data directory for test files |
| `MODELS_DIR` | `./models` | Models directory |

### Multiple Environments

#### Development
```bash
# .env.dev
LOG_LEVEL=DEBUG
DEVICE=cpu
PORT=7860
DEBUG=true
```

#### Production
```bash
# .env.prod
LOG_LEVEL=INFO
DEVICE=cuda
PORT=80
DEBUG=false
DOCKER_MEMORY_LIMIT=16G
```

**Usage:**
```bash
# Development
cp .env.dev .env
docker-compose up

# Production
cp .env.prod .env
docker-compose up -d
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## 📦 Building and Deployment

### Docker Build

```bash
# Build image
docker build -t invoice-ner:latest .

# Build with specific tag
docker build -t invoice-ner:v1.0.0 .

# Build with no cache
docker build --no-cache -t invoice-ner:latest .
```

### Running with Docker Compose

Docker Compose is the recommended way to run the application as it handles the model server (Triton) and application services together.

```bash
# Start all services (detached mode)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Production Deployment Considerations

1. **Use a reverse proxy** (nginx, Traefik) for SSL/TLS termination
2. **Set up monitoring** with Prometheus/Grafana
3. **Configure log aggregation** (ELK stack, Loki)
4. **Use Docker secrets** for sensitive configuration
5. **Set up automatic restarts** with proper health checks
6. **Use multi-stage builds** to reduce image size

### Multi-Stage Build (Advanced)

For smaller production images:

```dockerfile
# Builder stage
FROM python:3.10-slim as builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && \
    uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache -e .

# Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
CMD ["python", "app.py"]
```

## 🐛 Troubleshooting

### Development Issues

#### Import errors
```bash
# Reinstall in editable mode
uv pip install -e .
```

#### Pre-commit hook failures
```bash
# Fix formatting
black .
ruff check --fix .

# Skip hooks (not recommended)
git commit --no-verify
```

### Docker Issues

#### Container won't start
```bash
# Check logs
docker-compose logs invoice-ner

# Check container status
docker ps -a
```

#### Volume mount issues
```bash
# Verify paths exist
ls -la models/
ls -la data/

# Check permissions
chmod -R 755 models/
chmod -R 755 data/
```

### Environment Variable Issues

#### Variable not recognized
1. Check spelling in `.env`
2. Ensure no spaces around `=`: `PORT=7860` not `PORT = 7860`
3. Restart application
4. Check if variable is defined in code

#### Checking current configuration
```bash
# View environment variables
docker-compose config

# Check what the app sees
docker-compose exec invoice-ner env | grep -E "LOG_LEVEL|DEVICE|PORT"
```

## 📚 Additional Resources

- **Main README**: [README.md](../README.md) - User guide and quick start
- **Dataset documentation**: [Notion](https://www.notion.so/Dataset-Documentation-Notes-1609faffd568479dbaf1c072b23c472d)
- **Labeling heuristics**: [Notion](https://www.notion.so/Heuristics-Details-53af761344b7402fac834031244e032a)
- **Labeled datasets**:
  - [Kaggle](https://www.kaggle.com/datasets/ryanznie/sroie-datasetv2-with-labels)
  - [HuggingFace](https://huggingface.co/datasets/ryanznie/SROIE_2019_with_labels)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run pre-commit hooks: `pre-commit run --all-files`
6. Commit your changes: `git commit -m "Add my feature"`
7. Push to the branch: `git push origin feature/my-feature`
8. Create a Pull Request
