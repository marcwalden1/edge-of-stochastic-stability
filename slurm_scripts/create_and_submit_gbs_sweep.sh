#!/bin/bash
# create_and_submit_gbs_sweep.sh
#
# Generates and submits 20 SLURM jobs for the GBS suite sweep:
#   4 batch sizes × 5 optimizers
#
# Usage: bash slurm_scripts/create_and_submit_gbs_sweep.sh
#        (run from the repository root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SWEEP_DIR="$SCRIPT_DIR/gbs_sweep"
LOG_DIR="$SWEEP_DIR/logs"
mkdir -p "$SWEEP_DIR" "$LOG_DIR"

# Validate required environment variables
if [ -z "${DATASETS:-}" ]; then
    echo "ERROR: DATASETS environment variable is not set." >&2
    echo "       Set it in ~/.bashrc, e.g.:  export DATASETS=/n/holyscratch01/mwalden/eoss/datasets" >&2
    exit 1
fi

if [ -z "${RESULTS:-}" ]; then
    echo "ERROR: RESULTS environment variable is not set." >&2
    echo "       Set it in ~/.bashrc, e.g.:  export RESULTS=/n/holyscratch01/mwalden/eoss/results" >&2
    exit 1
fi

# Warn if RESULTS is inside $HOME (quota risk)
if [[ "$RESULTS" == "$HOME"* ]]; then
    echo "WARNING: RESULTS='$RESULTS' appears to be inside \$HOME." >&2
    echo "         Home directory has a 100 GB quota. Consider using scratch instead." >&2
    echo "         Continuing in 5 seconds (Ctrl-C to abort)..."
    sleep 5
fi

echo "Results will be written to: $RESULTS"
df -h "$RESULTS" 2>/dev/null || true

# Remove any stale wandb directories in the repo
rm -rf "$REPO_ROOT/wandb/" || true

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------
BATCH_SIZES=(1024 256 64 8)

declare -A OPT_FLAGS
OPT_FLAGS[SGD]="--lr 0.01"
OPT_FLAGS[SGD_m]="--lr 0.001 --momentum 0.9"
OPT_FLAGS[SGD_nest]="--lr 0.001 --momentum 0.9 --nesterov"
OPT_FLAGS[Adam]="--lr 0.00003 --adam 0.9 0.99"
OPT_FLAGS[Muon]="--lr 0.001 --muon --muon-momentum 0.95"

OPTIMIZERS=(SGD SGD_m SGD_nest Adam Muon)

# Common flags for every run
COMMON_FLAGS="--dataset cifar10 --model mlp --steps 150000 --num-data 8192 --activation silu \
--init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
--stop-loss 0.00001 \
--gbs-suite --batch-sharpness --lambdamax \
--disable-wandb \
--checkpoint-every 5000 \
--experiment-subdir gbs_sweep"

# ---------------------------------------------------------------------------
# Generate + submit one sbatch file per (optimizer, batch_size) combination
# ---------------------------------------------------------------------------
N_SUBMITTED=0

for OPT in "${OPTIMIZERS[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        TAG="${OPT}_${BATCH}"
        SBATCH_FILE="$SWEEP_DIR/${TAG}.sbatch"

        cat > "$SBATCH_FILE" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=eoss_gbs_${TAG}
#SBATCH --output=${LOG_DIR}/${TAG}.out
#SBATCH --error=${LOG_DIR}/${TAG}.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate eoss

# Results go to SCRATCH, not home directory
export RESULTS=${RESULTS}
# DATASETS inherited from ~/.bashrc

set -euo pipefail

echo "Starting GBS sweep run: OPT=${OPT}  BATCH=${BATCH}"
echo "Results dir: \$RESULTS"
echo "Node: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

cd ${REPO_ROOT}

python training.py \\
    ${COMMON_FLAGS} \\
    --batch ${BATCH} \\
    ${OPT_FLAGS[$OPT]} \\
    --experiment-tag ${TAG}

echo "Done: ${TAG}"
date
SBATCH_EOF

        echo "Submitting: $SBATCH_FILE"
        sbatch "$SBATCH_FILE"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Submitted $N_SUBMITTED jobs."
echo "Monitor with:  squeue -u \$USER"
echo "Results will appear in: \$RESULTS/plaintext/cifar10_mlp/gbs_sweep/"
