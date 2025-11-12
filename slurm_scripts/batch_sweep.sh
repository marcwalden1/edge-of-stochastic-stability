#!/bin/bash
# Generate batch_size_*.sbatch files for batch size sweep

BATCH_SIZES=( \
2 3 4 5 6 7 8 9 \
10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 \
120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 660 680 700 720 740 760 780 800 820 840 860 880 900 920 940 960 980 \
1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 \
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Generating batch_size_*.sbatch files in $SCRIPT_DIR"
echo "Total batch sizes: ${#BATCH_SIZES[@]}"
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    script_path="${SCRIPT_DIR}/batch_size_${batch_size}.sbatch"
    cat > "$script_path" <<EOF
    
#!/bin/bash
#SBATCH --job-name=bs${batch_size}
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/batch_size_${batch_size}_%j.out
#SBATCH --error=slurm/batch_size_${batch_size}_%j.err
#SBATCH --requeue

# Single job for batch size ${batch_size}
BATCH_SIZE=${batch_size}

echo "Job ID: \$SLURM_JOB_ID, Batch Size: \$BATCH_SIZE"

# Load Python module (required for FASRC)
module load python/3.10.13-fasrc01 2>&1 || module load python/3.9.12-fasrc01 2>&1 || echo "Warning: Could not load Python module, using system Python"

# Change to the project root directory
# SLURM_SUBMIT_DIR is the directory where sbatch was run from
if [ -n "\$SLURM_SUBMIT_DIR" ]; then
    # If submitted from slurm_scripts, go up one level; otherwise use submit dir
    if [[ "\$SLURM_SUBMIT_DIR" == *"/slurm_scripts" ]]; then
        PROJECT_DIR=\$(dirname "\$SLURM_SUBMIT_DIR")
    else
        PROJECT_DIR="\$SLURM_SUBMIT_DIR"
    fi
    cd "\$PROJECT_DIR" || { echo "Error: Could not change to project directory \$PROJECT_DIR"; exit 1; }
else
    # Fallback: use explicit home directory path
    cd "\$HOME/edge-of-stochastic-stability-1" || { echo "Error: Could not change to project directory"; exit 1; }
fi
echo "Working directory: \$(pwd)"
echo "SLURM_SUBMIT_DIR: \$SLURM_SUBMIT_DIR"

# Check for virtual environment
echo "Checking for virtual environment..."
VENV_PATH="eoss/bin/activate"
if [ ! -f "\$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at \$(pwd)/\$VENV_PATH"
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
source "\$VENV_PATH" || {
    echo "ERROR: Failed to activate virtual environment"
    echo "Activate script exists but failed to source"
    exit 1
}

# Verify activation worked by checking Python path
if [ -z "\$VIRTUAL_ENV" ]; then
    echo "ERROR: VIRTUAL_ENV not set after activation"
    exit 1
fi

# Verify we're using the venv's Python
PYTHON_FROM_VENV="\$VIRTUAL_ENV/bin/python"
if [ ! -f "\$PYTHON_FROM_VENV" ]; then
    echo "ERROR: Python not found in venv at \$PYTHON_FROM_VENV"
    exit 1
fi

echo "Virtual environment activated: \$VIRTUAL_ENV"
echo "Python: \$(which python)"
echo "Python version: \$(python --version 2>&1)"

# Sanity check: ensure training.py exists in working directory
echo "Confirming training.py exists in \$(pwd)"
if [ ! -f "training.py" ]; then
  echo "ERROR: training.py not found in \$(pwd)"
  echo "Listing current directory:"
  ls -la
  echo "PROJECT_DIR: \$PROJECT_DIR"
  exit 1
fi

export DATASETS="\$HOME/datasets"
export RESULTS="\$HOME/results"
export WANDB_MODE=offline
export WANDB_PROJECT=eoss2
export WANDB_DIR="\$RESULTS"
export WANDB_API_KEY=71c8fb423d7047b7be7481652b13776f0dd6584d

python training.py \\
  --dataset cifar10 --model mlp --batch \$BATCH_SIZE --lr 0.01 --momentum 0.9 \\
  --steps 150000 --num-data 8192 --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --stop-loss 0.00001 --lambdamax --batch-sharpness --track-trajectory --projection-seed 888 --wandb-tag batch-size-sweep
EOF

    chmod +x "$script_path"
    echo "Generated: batch_size_${batch_size}.sbatch"
done

echo ""
echo "Generated ${#BATCH_SIZES[@]} batch_size_*.sbatch files"