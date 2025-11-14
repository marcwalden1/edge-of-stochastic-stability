#!/bin/bash
#SBATCH --job-name=dist_bs32_lr0p002_m09
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/distance_bs32_lr0p002_m09_%j.out
#SBATCH --error=slurm/distance_bs32_lr0p002_m09_%j.err
#SBATCH --requeue

# Distance tracking run: batch size 32, lr 0.002, momentum 0.9
BATCH_SIZE=32
LEARNING_RATE=0.002
MOMENTUM=0.9

echo "Job ID: $SLURM_JOB_ID"
echo "Configuration: batch=$BATCH_SIZE, lr=$LEARNING_RATE, momentum=$MOMENTUM"

# Load Python module (required for FASRC)
module load python/3.10.13-fasrc01 2>&1 \
  || module load python/3.9.12-fasrc01 2>&1 \
  || echo "Warning: Could not load cluster Python module, using system Python"

# Change to the project root directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    if [[ "$SLURM_SUBMIT_DIR" == *"/slurm_scripts"* ]]; then
        PROJECT_DIR=$(dirname "$SLURM_SUBMIT_DIR")
    else
        PROJECT_DIR="$SLURM_SUBMIT_DIR"
    fi
    cd "$PROJECT_DIR" || { echo "Error: Could not change to project directory $PROJECT_DIR"; exit 1; }
else
    cd "$HOME/edge-of-stochastic-stability-1" || { echo "Error: Could not change to project directory"; exit 1; }
fi
echo "Working directory: $(pwd)"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Check for virtual environment
echo "Checking for virtual environment..."
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

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH" || {
    echo "ERROR: Failed to activate virtual environment"
    echo "Activate script exists but failed to source"
    exit 1
}

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: VIRTUAL_ENV not set after activation"
    exit 1
fi

PYTHON_FROM_VENV="$VIRTUAL_ENV/bin/python"
if [ ! -f "$PYTHON_FROM_VENV" ]; then
    echo "ERROR: Python not found in venv at $PYTHON_FROM_VENV"
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Ensure training.py exists
echo "Confirming training.py exists in $(pwd)"
if [ ! -f "training.py" ]; then
  echo "ERROR: training.py not found in $(pwd)"
  ls -la
  exit 1
fi

export DATASETS="$HOME/datasets"
export RESULTS="$HOME/results"
export WANDB_MODE=offline
export WANDB_PROJECT=eoss2
export WANDB_DIR="$RESULTS"

python training.py \
  --dataset cifar10 --model mlp --batch $BATCH_SIZE --lr $LEARNING_RATE --momentum $MOMENTUM \
  --steps 150000 --num-data 8192 --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 --lambdamax --batch-sharpness \
  --track-trajectory --projection-seed 888 \
  --wandb-tag distance-test-b32-lr0p002-m0p9

