#!/usr/bin/env bash

VENV_DIR=".venv"

echo "=== Setting up project from existing pyproject.toml and uv.lock ==="

# Step 1: Ensure uv is installed globally
if ! command -v uv &> /dev/null; then
    echo "Installing uv globally..."
    curl -LsSf https://astral.sh/uv/install.sh || echo "Failed to install uv globally, continuing..."
else
    echo "uv is already installed globally."
fi

# Step 2: Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || echo "Failed to create venv, continuing..."
fi

# Step 3: Activate virtual environment
source "$VENV_DIR/bin/activate" || echo "Failed to activate venv, continuing..."

# Step 4: Upgrade pip in the venv (optional, uv will manage it anyway)
python3 -m ensurepip --upgrade || echo "Failed to bootstrap pip, continuing..."
python3 -m pip install --upgrade pip setuptools wheel || echo "Failed to upgrade pip/setuptools/wheel, continuing..."

# Step 5: Sync dependencies from uv.lock using global uv
echo "Syncing dependencies..."
uv sync || echo "Failed to sync dependencies, continuing..."

# Step 6: Verify installations
echo "Installed packages:"
uv pip list || echo "Failed to list installed packages, continuing..."

# Step 7: Optionally initialize pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install || echo "Failed to install pre-commit hooks, continuing..."
fi

echo "Setup script finished! Virtual environment is in $VENV_DIR"
echo "Activate it with: source $VENV_DIR/bin/activate"

