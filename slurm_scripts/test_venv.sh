#!/bin/bash
# Test script to verify venv activation works on the cluster

echo "=== Testing Virtual Environment Setup ==="
echo ""

# Load Python module (same as in SLURM script)
echo "1. Loading Python module..."
module load python/3.10.13-fasrc01 2>&1 || module load python/3.9.12-fasrc01 2>&1 || echo "Warning: Could not load Python module, using system Python"
echo "  System Python: $(which python)"
echo "  Python version: $(python --version 2>&1)"
echo ""

# Change to project directory
PROJECT_DIR="$HOME/edge-of-stochastic-stability-1"
echo "2. Changing to project directory..."
cd "$PROJECT_DIR" || { echo "ERROR: Could not change to $PROJECT_DIR"; exit 1; }
echo "  Working directory: $(pwd)"
echo ""

# Check for virtual environment
echo "3. Checking for virtual environment..."
VENV_PATH="eoss/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $(pwd)/$VENV_PATH"
    if [ -d "eoss" ]; then
        echo "Contents of eoss directory:"
        ls -la eoss/ 2>&1 | head -20
    else
        echo "eoss/ directory does not exist"
    fi
    exit 1
fi
echo "  ✓ Activate script found at $VENV_PATH"
echo ""

# Activate virtual environment
echo "4. Activating virtual environment..."
source "$VENV_PATH" || {
    echo "ERROR: Failed to activate virtual environment"
    echo "Activate script exists but failed to source"
    exit 1
}
echo "  ✓ Sourced activate script"
echo ""

# Verify activation worked
echo "5. Verifying activation..."
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: VIRTUAL_ENV not set after activation"
    exit 1
fi
echo "  ✓ VIRTUAL_ENV is set: $VIRTUAL_ENV"
echo ""

# Verify we're using the venv's Python
PYTHON_FROM_VENV="$VIRTUAL_ENV/bin/python"
if [ ! -f "$PYTHON_FROM_VENV" ]; then
    echo "ERROR: Python not found in venv at $PYTHON_FROM_VENV"
    exit 1
fi
echo "  ✓ Python executable found in venv"
echo ""

# Show Python info
echo "6. Python information:"
echo "  which python: $(which python)"
echo "  Python version: $(python --version 2>&1)"
echo "  Python path: $(python -c 'import sys; print(sys.executable)' 2>&1)"
echo ""

# Test importing key packages
echo "7. Testing imports..."
python -c "import torch; print(f'  ✓ torch {torch.__version__}')" 2>&1 || echo "  ✗ torch import failed"
python -c "import numpy; print(f'  ✓ numpy {numpy.__version__}')" 2>&1 || echo "  ✗ numpy import failed"
python -c "import wandb; print(f'  ✓ wandb available')" 2>&1 || echo "  ✗ wandb import failed"
echo ""

# Test importing training.py dependencies
echo "8. Testing training.py dependencies..."
python -c "from utils.data import prepare_dataset; print('  ✓ utils.data')" 2>&1 || echo "  ✗ utils.data import failed"
python -c "from utils.nets import MLP; print('  ✓ utils.nets')" 2>&1 || echo "  ✗ utils.nets import failed"
echo ""

echo "=== Test Complete ==="
echo "If all checks passed, the venv setup is working correctly!"

