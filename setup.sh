#!/bin/bash
# Quick setup script for new machines
# Usage: bash setup.sh

set -e

echo "=================================================="
echo "BirdNET Custom Classifier Suite - Environment Setup"
echo "=================================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

if [[ ! $PYTHON_VERSION == 3.10.* ]]; then
    echo "WARNING: This project requires Python 3.10.x, you have $PYTHON_VERSION"
    echo "Continuing anyway, but you may encounter issues..."
fi

echo ""
echo "Step 1: Installing main dependencies..."
pip install -r requirements-frozen.txt

echo ""
echo "Step 2: Installing this package in editable mode..."
pip install -e .

echo ""
echo "Step 3: Patching BirdNET-Analyzer Python requirement..."
BIRDNET_TOML="external/BirdNET-Analyzer/pyproject.toml"

if [ ! -f "$BIRDNET_TOML" ]; then
    echo "ERROR: BirdNET-Analyzer not found at $BIRDNET_TOML"
    echo "Make sure you've cloned the submodule: git submodule update --init --recursive"
    exit 1
fi

# Backup original
cp "$BIRDNET_TOML" "$BIRDNET_TOML.backup"

# Patch the Python version requirement
sed -i 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' "$BIRDNET_TOML"
echo "✓ Patched $BIRDNET_TOML (backup saved as .backup)"

echo ""
echo "Step 4: Installing BirdNET-Analyzer in editable mode..."
pip install -e external/BirdNET-Analyzer[train]

echo ""
echo "Step 5: Verifying installation..."
python -c "import tensorflow; print('✓ TensorFlow:', tensorflow.__version__)"
python -c "import birdnet_analyzer; print('✓ BirdNET-Analyzer:', birdnet_analyzer.__version__)"
python -c "import birdnet_custom_classifier_suite; print('✓ Custom Classifier Suite: OK')"

echo ""
echo "=================================================="
echo "Setup complete! 🎉"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  - Review ENVIRONMENT_SETUP.md for details"
echo "  - Run tests: pytest"
echo "  - Start UI: streamlit run scripts/streamlit_app.py"
echo ""
