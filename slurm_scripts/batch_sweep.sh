#!/bin/bash
# Generate batch_size_*.sbatch files for batch size sweep

BATCH_SIZES=( \
2 3 4 5 6 7 8 9 \
10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 \
120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 660 680 700 \
800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 \
)

SCRIPT_DIR="$HOME/edge-of-stochastic-stability/slurm_scripts"

echo "Generating batch_size_*.sbatch files in $SCRIPT_DIR"
echo "Total batch sizes: ${#BATCH_SIZES[@]}"
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    script_path="${SCRIPT_DIR}/batch_size_${batch_size}.sbatch"
    cat > "$script_path" <<EOF
#!/bin/bash
#SBATCH --job-name=bs${batch_size}
#SBATCH --partition=mit_normal_gpu
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
module load python/3.10.13-fasrc01 2>&1 || module load python/3.9.12-fasrc01 2>&1 || echo "Warning: Could not load Python"

# Change to the project root directory
if [ -n "\$SLURM_SUBMIT_DIR" ]; then
    cd "\$SLURM_SUBMIT_DIR" || exit 1
else
    cd "\$HOME/edge-of-stochastic-stability" || exit 1
fi
echo "Working directory: \$(pwd)"

# Activate your virtual environment
source ~/eoss/bin/activate 2>/dev/null || echo "Warning: could not activate venv"

# Export necessary environment variables
export DATASETS="\$HOME/datasets"
export RESULTS="\$HOME/results"
export WANDB_MODE=offline
export WANDB_PROJECT=eoss
export WANDB_DIR="\$RESULTS"

# Run training
python training.py \\
  --dataset cifar10 --model mlp --batch \$BATCH_SIZE --lr 0.01 \\
  --steps 150000 --num-data 8192 \\
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --stop-loss 0.00001 --lambdamax --batch-sharpness
EOF

    chmod +x "$script_path"
    echo "Generated: batch_size_${batch_size}.sbatch"
done

echo ""
echo "Generated ${#BATCH_SIZES[@]} batch_size_*.sbatch files."
