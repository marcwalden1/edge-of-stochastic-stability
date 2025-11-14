#!/bin/bash
#SBATCH --job-name=gf_lr0p001
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/full_batch_lr0p001_%j.out
#SBATCH --error=slurm/full_batch_lr0p001_%j.err
#SBATCH --requeue

# Gradient flow run: full batch (8192), lr 0.001, momentum 0.0
BATCH_SIZE=8192
NUM_DATA=8192
LEARNING_RATE=0.001
MOMENTUM=0.0

echo "Job ID: $SLURM_JOB_ID"
echo "Configuration: batch=$BATCH_SIZE, num_data=$NUM_DATA, lr=$LEARNING_RATE, momentum=$MOMENTUM"

# Load Python module (required for FASRC)
module load python/3.10.13-fasrc01 2>&1 || module load python/3.9.12-fasrc01 2>&1 || echo "Warning: Could not load Python module, using system Python"

# Change to the project root directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
    if [[ "$PROJECT_DIR" == *"/slurm_scripts" ]]; then
        PROJECT_DIR=$(dirname "$PROJECT_DIR")
    elif [[ "$PROJECT_DIR" == *"/slurm_scripts/"* ]]; then
        PROJECT_DIR=$(dirname "$PROJECT_DIR")
        PROJECT_DIR=$(dirname "$PROJECT_DIR")
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

# Verify activation worked by checking Python path
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: VIRTUAL_ENV not set after activation"
    exit 1
fi

# Verify we're using the venv's Python
PYTHON_FROM_VENV="$VIRTUAL_ENV/bin/python"
if [ ! -f "$PYTHON_FROM_VENV" ]; then
    echo "ERROR: Python not found in venv at $PYTHON_FROM_VENV"
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Sanity check: ensure training.py exists in working directory
echo "Confirming training.py exists in $(pwd)"
if [ ! -f "training.py" ]; then
  echo "ERROR: training.py not found in $(pwd)"
  echo "Listing current directory:"
  ls -la
  echo "PROJECT_DIR: $PROJECT_DIR"
  exit 1
fi

export DATASETS="$HOME/datasets"
export RESULTS="$HOME/results"
export WANDB_MODE=offline
export WANDB_PROJECT=eoss2
export WANDB_DIR="$RESULTS"
export WANDB_API_KEY=71c8fb423d7047b7be7481652b13776f0dd6584d

python training.py \
  --dataset cifar10 --model mlp --batch $BATCH_SIZE --num-data $NUM_DATA \
  --lr $LEARNING_RATE --momentum $MOMENTUM \
  --steps 150000 --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 --lambdamax --batch-sharpness \
  --track-trajectory --projection-seed 888 \
  --wandb-tag gradient-flow-comparison --wandb-name full_batch_lr${LEARNING_RATE}

