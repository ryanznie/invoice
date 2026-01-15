# Use Python 3.10 slim image as base
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies directly (not editable to avoid README requirement)
# Note: We'll use CPU-only PyTorch for Docker to reduce image size
# Split install to avoid huge layer commit fails
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache \
    transformers \
    fastapi \
    uvicorn[standard] \
    gradio \
    python-dotenv \
    pillow \
    pandas \
    tqdm \
    onnx \
    onnxruntime \
    tritonclient[http]

# Copy application code
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p models

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

# Run the application
CMD ["python", "app.py", "--port", "7860"]
