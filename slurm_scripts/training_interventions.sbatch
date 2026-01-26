#!/bin/bash
#SBATCH --job-name=intervention
#SBATCH --output=logs/intervention_%A_%a.out
#SBATCH --error=logs/intervention_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-8

# Training Interventions Experiment
# =================================
# Runs 3 training configurations (A, B, C) to study how mid-training
# hyperparameter changes affect batch sharpness.
#
# Experiment matrix (indexed by $SLURM_ARRAY_TASK_ID):
#   ID | Type     | Variant | Config
#   ---|----------|---------|------------------------------------------
#   0  | lr       | A       | lr=0.004, mom=0.9, batch=16 (baseline)
#   1  | lr       | B       | lr=0.003, mom=0.9, batch=16
#   2  | lr       | C       | baseline -> lr=0.003 at intervention step
#   3  | momentum | A       | lr=0.004, mom=0.9, batch=16 (baseline)
#   4  | momentum | B       | lr=0.004, mom=0.8, batch=16
#   5  | momentum | C       | baseline -> mom=0.8 at intervention step
#   6  | batch    | A       | lr=0.004, mom=0.9, batch=16 (baseline)
#   7  | batch    | B       | lr=0.004, mom=0.9, batch=32
#   8  | batch    | C       | baseline -> batch=32 at intervention step
#
# Arguments (passed after -- to sbatch):
#   --baseline-lr LR         - Baseline learning rate for Run A (REQUIRED)
#   --baseline-momentum M    - Baseline momentum for Run A (REQUIRED)
#   --baseline-batch B       - Baseline batch size for Run A (REQUIRED)
#   --steps STEPS            - Total training steps (REQUIRED)
#   --intervention-step N    - Step at which Run C switches hyperparameters (REQUIRED)
#   --num-data N             - Number of CIFAR-10 data points for training (REQUIRED)
#   --intervention-lr LR     - LR for Run B/C intervention (default: same as baseline)
#   --intervention-momentum M - Momentum for Run B/C (default: same as baseline)
#   --intervention-batch B   - Batch size for Run B/C (default: same as baseline)
#   --model MODEL            - Model architecture (default: mlp)
#   --dataset DATASET        - Dataset name (default: cifar10)
#   --experiment-name NAME   - Prefix for result folders (default: intervention)
#   --experiment-subdir DIR  - Subdirectory for grouping experiments (e.g., experiment_1)
#
# Usage:
#   # Full experiment with all required parameters
#   sbatch slurm_scripts/training_interventions.sbatch \
#     --baseline-lr 0.004 --baseline-momentum 0.9 --baseline-batch 16 \
#     --steps 200000 --intervention-step 100000 --num-data 32000 \
#     --intervention-lr 0.003 --intervention-momentum 0.8 --intervention-batch 8
#
#   # LR intervention only
#   sbatch --array=0-2 slurm_scripts/training_interventions.sbatch \
#     --baseline-lr 0.004 --baseline-momentum 0.9 --baseline-batch 16 \
#     --steps 200000 --intervention-step 100000 --num-data 32000 \
#     --intervention-lr 0.003
#
#   # Momentum intervention only
#   sbatch --array=3-5 slurm_scripts/training_interventions.sbatch \
#     --baseline-lr 0.004 --baseline-momentum 0.9 --baseline-batch 16 \
#     --steps 200000 --intervention-step 100000 --num-data 32000 \
#     --intervention-momentum 0.8
#
#   # Batch size intervention only
#   sbatch --array=6-8 slurm_scripts/training_interventions.sbatch \
#     --baseline-lr 0.004 --baseline-momentum 0.9 --baseline-batch 16 \
#     --steps 200000 --intervention-step 100000 --num-data 32000 \
#     --intervention-batch 8

# Baseline hyperparameters (Run A) - must be specified via arguments
BASELINE_LR=""
BASELINE_MOMENTUM=""
BASELINE_BATCH=""

# Required arguments
STEPS=""
INTERVENTION_STEP=""
NUM_DATA=""

# Default values
MODEL="mlp"
EXPERIMENT_NAME="intervention"
DATASET="cifar10"
EXPERIMENT_SUBDIR=""

# Intervention values (default to baseline, set after parsing)
INTERVENTION_LR_VALUE=""
INTERVENTION_MOMENTUM_VALUE=""
INTERVENTION_BATCH_VALUE=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-lr)
            BASELINE_LR="$2"
            shift 2
            ;;
        --baseline-momentum)
            BASELINE_MOMENTUM="$2"
            shift 2
            ;;
        --baseline-batch)
            BASELINE_BATCH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --intervention-step)
            INTERVENTION_STEP="$2"
            shift 2
            ;;
        --num-data)
            NUM_DATA="$2"
            shift 2
            ;;
        --intervention-lr)
            INTERVENTION_LR_VALUE="$2"
            shift 2
            ;;
        --intervention-momentum)
            INTERVENTION_MOMENTUM_VALUE="$2"
            shift 2
            ;;
        --intervention-batch)
            INTERVENTION_BATCH_VALUE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
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
if [[ -z "$BASELINE_LR" ]]; then
    echo "Error: --baseline-lr is required"
    exit 1
fi
if [[ -z "$BASELINE_MOMENTUM" ]]; then
    echo "Error: --baseline-momentum is required"
    exit 1
fi
if [[ -z "$BASELINE_BATCH" ]]; then
    echo "Error: --baseline-batch is required"
    exit 1
fi
if [[ -z "$STEPS" ]]; then
    echo "Error: --steps is required"
    exit 1
fi
if [[ -z "$INTERVENTION_STEP" ]]; then
    echo "Error: --intervention-step is required"
    exit 1
fi
if [[ -z "$NUM_DATA" ]]; then
    echo "Error: --num-data is required"
    exit 1
fi

# Set intervention values to baseline if not specified
if [[ -z "$INTERVENTION_LR_VALUE" ]]; then
    INTERVENTION_LR_VALUE=${BASELINE_LR}
fi
if [[ -z "$INTERVENTION_MOMENTUM_VALUE" ]]; then
    INTERVENTION_MOMENTUM_VALUE=${BASELINE_MOMENTUM}
fi
if [[ -z "$INTERVENTION_BATCH_VALUE" ]]; then
    INTERVENTION_BATCH_VALUE=${BASELINE_BATCH}
fi

# Create logs directory if needed
mkdir -p logs

echo "========================================"
echo "Training Interventions Experiment"
echo "========================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Num Data: ${NUM_DATA}"
echo "Total Steps: ${STEPS}"
echo "Intervention Step: ${INTERVENTION_STEP}"
echo "Baseline (A): LR=${BASELINE_LR}, Mom=${BASELINE_MOMENTUM}, Batch=${BASELINE_BATCH}"
echo "Intervention (B): LR=${INTERVENTION_LR_VALUE}, Mom=${INTERVENTION_MOMENTUM_VALUE}, Batch=${INTERVENTION_BATCH_VALUE}"
echo "========================================"

# Map task ID to experiment configuration
case ${SLURM_ARRAY_TASK_ID} in
    0)
        # LR experiment - Run A (baseline)
        EXP_TYPE="lr"
        VARIANT="A"
        LR=${BASELINE_LR}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS=""
        ;;
    1)
        # LR experiment - Run B (changed LR)
        EXP_TYPE="lr"
        VARIANT="B"
        LR=${INTERVENTION_LR_VALUE}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS=""
        ;;
    2)
        # LR experiment - Run C (intervention)
        EXP_TYPE="lr"
        VARIANT="C"
        LR=${BASELINE_LR}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS="--intervention-step ${INTERVENTION_STEP} --intervention-lr ${INTERVENTION_LR_VALUE}"
        ;;
    3)
        # Momentum experiment - Run A (baseline)
        EXP_TYPE="momentum"
        VARIANT="A"
        LR=${BASELINE_LR}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS=""
        ;;
    4)
        # Momentum experiment - Run B (changed momentum)
        EXP_TYPE="momentum"
        VARIANT="B"
        LR=${BASELINE_LR}
        MOMENTUM=${INTERVENTION_MOMENTUM_VALUE}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS=""
        ;;
    5)
        # Momentum experiment - Run C (intervention)
        EXP_TYPE="momentum"
        VARIANT="C"
        LR=${BASELINE_LR}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS="--intervention-step ${INTERVENTION_STEP} --intervention-momentum ${INTERVENTION_MOMENTUM_VALUE}"
        ;;
    6)
        # Batch experiment - Run A (baseline)
        EXP_TYPE="batch"
        VARIANT="A"
        LR=${BASELINE_LR}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS=""
        ;;
    7)
        # Batch experiment - Run B (changed batch size)
        EXP_TYPE="batch"
        VARIANT="B"
        LR=${BASELINE_LR}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${INTERVENTION_BATCH_VALUE}
        INTERVENTION_ARGS=""
        ;;
    8)
        # Batch experiment - Run C (intervention)
        EXP_TYPE="batch"
        VARIANT="C"
        LR=${BASELINE_LR}
        MOMENTUM=${BASELINE_MOMENTUM}
        BATCH=${BASELINE_BATCH}
        INTERVENTION_ARGS="--intervention-step ${INTERVENTION_STEP} --intervention-batch ${INTERVENTION_BATCH_VALUE}"
        ;;
    *)
        echo "Error: Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

EXPERIMENT_TAG="${EXPERIMENT_NAME}_${EXP_TYPE}_${VARIANT}"

echo "Experiment Type: ${EXP_TYPE}"
echo "Variant: ${VARIANT}"
echo "Experiment Tag: ${EXPERIMENT_TAG}"
echo "LR: ${LR}, Momentum: ${MOMENTUM}, Batch: ${BATCH}"
echo "Intervention Args: ${INTERVENTION_ARGS:-none}"
echo "========================================"

# Run training
python training.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --lr ${LR} \
    --momentum ${MOMENTUM} \
    --batch ${BATCH} \
    --steps ${STEPS} \
    --num-data ${NUM_DATA} \
    --init-scale 0.2 \
    --activation silu \
    --stop-loss 0.00001 \
    --disable-wandb \
    --batch-sharpness \
    --lambdamax \
    --experiment-tag ${EXPERIMENT_TAG} \
    ${INTERVENTION_ARGS} \
    ${EXPERIMENT_SUBDIR:+--experiment-subdir ${EXPERIMENT_SUBDIR}}

echo "========================================"
echo "Training completed for ${EXPERIMENT_TAG}"
echo "========================================"
