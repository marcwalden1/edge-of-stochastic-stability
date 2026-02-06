#!/bin/bash
#SBATCH --job-name=traj_compare
#SBATCH --output=logs/traj_compare_%A_%a.out
#SBATCH --error=logs/traj_compare_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1

# Trajectory Comparison Experiment
# ================================
# Runs 2 training configurations to compare SGD trajectories with different
# hyperparameters but equal effective learning rate.
#
# Experiment matrix (indexed by $SLURM_ARRAY_TASK_ID):
#   ID | Config
#   ---|------------------------------------------
#   0  | Run 1: eta=eta1, beta=beta1 (e.g., standard SGD)
#   1  | Run 2: eta=eta2, beta=beta2 (e.g., SGD with momentum)
#
# Arguments (passed after -- to sbatch):
#   --eta1 LR             - Learning rate for Run 1 (default: 0.005)
#   --beta1 M             - Momentum for Run 1 (default: 0.0)
#   --eta2 LR             - Learning rate for Run 2 (default: 0.0025)
#   --beta2 M             - Momentum for Run 2 (default: 0.5)
#   --batch B             - Batch size for both runs (default: 8)
#   --num-data N          - Number of training data points (default: 8192)
#   --steps STEPS         - Training steps (REQUIRED)
#   --model MODEL         - Model architecture (default: mlp)
#   --experiment-name NAME - Base name for outputs (default: traj_compare)
#   --experiment-subdir DIR - Subdirectory for grouping experiments
#
# Usage:
#   sbatch distance_plots/run_trajectory_comparison.sbatch \
#     --eta1 0.005 --beta1 0.0 \
#     --eta2 0.0025 --beta2 0.5 \
#     --batch 8 --steps 50000 \
#     --experiment-name sgd_vs_momentum \
#     --experiment-subdir trajectory_exp_1

# Default hyperparameters
ETA1="0.005"
BETA1="0.0"
ETA2="0.0025"
BETA2="0.5"
BATCH="8"
NUM_DATA="8192"

# Required arguments
STEPS=""

# Default values
MODEL="mlp"
EXPERIMENT_NAME="traj_compare"
EXPERIMENT_SUBDIR=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eta1)
            ETA1="$2"
            shift 2
            ;;
        --beta1)
            BETA1="$2"
            shift 2
            ;;
        --eta2)
            ETA2="$2"
            shift 2
            ;;
        --beta2)
            BETA2="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --num-data)
            NUM_DATA="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --experiment-subdir)
            EXPERIMENT_SUBDIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$STEPS" ]]; then
    echo "Error: --steps is required"
    exit 1
fi

# Create logs directory if needed
mkdir -p logs

echo "========================================"
echo "Trajectory Comparison Experiment"
echo "========================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Num Data: ${NUM_DATA}"
echo "Batch Size: ${BATCH}"
echo "Total Steps: ${STEPS}"
echo "Run 1: eta=${ETA1}, beta=${BETA1}"
echo "Run 2: eta=${ETA2}, beta=${BETA2}"
echo "========================================"

# Set offline mode for wandb
export WANDB_MODE=offline

# Map task ID to experiment configuration
case ${SLURM_ARRAY_TASK_ID} in
    0)
        # Run 1
        LR=${ETA1}
        MOMENTUM=${BETA1}
        EXPERIMENT_TAG="${EXPERIMENT_NAME}_run1_eta${ETA1}_beta${BETA1}"
        ;;
    1)
        # Run 2
        LR=${ETA2}
        MOMENTUM=${BETA2}
        EXPERIMENT_TAG="${EXPERIMENT_NAME}_run2_eta${ETA2}_beta${BETA2}"
        ;;
    *)
        echo "Error: Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "Experiment Tag: ${EXPERIMENT_TAG}"
echo "LR: ${LR}, Momentum: ${MOMENTUM}"
echo "========================================"

# Build momentum argument (only add if non-zero)
MOMENTUM_ARG=""
if [[ $(echo "$MOMENTUM != 0" | bc -l) -eq 1 ]]; then
    MOMENTUM_ARG="--momentum ${MOMENTUM}"
fi

# Run training with trajectory tracking
python training.py \
    --dataset cifar10 \
    --model ${MODEL} \
    --lr ${LR} \
    ${MOMENTUM_ARG} \
    --batch ${BATCH} \
    --steps ${STEPS} \
    --num-data ${NUM_DATA} \
    --activation silu \
    --init-scale 0.2 \
    --dataset-seed 111 \
    --init-seed 8312 \
    --stop-loss 0.00001 \
    --batch-sharpness \
    --lambdamax \
    --track-trajectory \
    --measure-test-distance \
    --experiment-tag ${EXPERIMENT_TAG} \
    ${EXPERIMENT_SUBDIR:+--experiment-subdir ${EXPERIMENT_SUBDIR}}

# Get the output folder path (from RESULTS env var)
RESULTS_BASE="${RESULTS:-$HOME/results}"
if [[ -n "$EXPERIMENT_SUBDIR" ]]; then
    SEARCH_PATH="${RESULTS_BASE}/plaintext/cifar10_${MODEL}/${EXPERIMENT_SUBDIR}"
else
    SEARCH_PATH="${RESULTS_BASE}/plaintext/cifar10_${MODEL}"
fi

# Find the folder that was just created (most recent matching the experiment tag)
RUN_FOLDER=$(find "${SEARCH_PATH}" -maxdepth 1 -type d -name "${EXPERIMENT_TAG}_*" | sort -r | head -1)

echo "========================================"
echo "Training completed for ${EXPERIMENT_TAG}"
echo "Output folder: ${RUN_FOLDER}"
echo "========================================"

# Write to manifest file (append mode, one line per run)
MANIFEST_DIR="${SEARCH_PATH}"
MANIFEST_FILE="${MANIFEST_DIR}/manifest_${EXPERIMENT_NAME}.txt"

# Use flock for safe concurrent writes
{
    flock -x 200
    echo "${RUN_FOLDER}" >> "${MANIFEST_FILE}"
} 200>"${MANIFEST_FILE}.lock"

echo "Added to manifest: ${MANIFEST_FILE}"

# Print next step command
echo ""
echo "========================================"
echo "NEXT STEP (after BOTH array tasks complete):"
echo "========================================"
echo ""
echo "# Read run paths from manifest:"
echo "cat ${MANIFEST_FILE}"
echo ""
echo "# Then run distance computation:"
echo "python distance_plots/compute_comparison_distances.py \\"
echo "    <RUN1_PATH> \\"
echo "    <RUN2_PATH> \\"
echo "    --output-dir output/${EXPERIMENT_NAME} \\"
echo "    --lr1 ${ETA1} --lr2 ${ETA2} \\"
echo "    --offline"
echo ""
echo "========================================"
