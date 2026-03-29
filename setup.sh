#!/usr/bin/env bash
# setup.sh - Create a reproducible Python virtual environment for the mla-proj

set -e

ENV_DIR=".venv"

echo "==== 🚀 Setting up Whiteboard Digitizer Environment ===="

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 could not be found. Please install Python 3.9+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $PYTHON_VERSION"

# 2. Setup Virtual Environment
if [ -d "$ENV_DIR" ]; then
    echo "✅ Virtual environment already exists at $ENV_DIR/"
else
    echo "📦 Creating new virtual environment at $ENV_DIR/..."
    python3 -m venv "$ENV_DIR"
fi

# 3. Activate the environment
echo "🔄 Activating environment..."
source "$ENV_DIR/bin/activate"

# 4. Install dependencies
echo "⬇️ Installing Python dependencies (this might take a few minutes)..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Success
echo ""
echo "==== 🎉 Setup Complete! ===="
echo "To start working on the project, activate the environment by running:"
echo "👉  source .venv/bin/activate"
echo "To run the layout detector locally as a test:"
echo "👉  python src/detect_layout.py"
