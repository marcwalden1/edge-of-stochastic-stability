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
# Usage:
#   # Default run (MLP, 100k steps)
#   sbatch slurm_scripts/training_interventions.sbatch
#
#   # Custom model and steps
#   MODEL=cnn STEPS=50000 INTERVENTION_STEP=25000 sbatch slurm_scripts/training_interventions.sbatch
#
#   # Run specific experiments only
#   sbatch --array=0-2 slurm_scripts/training_interventions.sbatch  # LR experiments only

# Environment variables with defaults
MODEL=${MODEL:-mlp}
STEPS=${STEPS:-100000}
INTERVENTION_STEP=${INTERVENTION_STEP:-50000}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-intervention}
DATASET=${DATASET:-cifar10}

# Baseline hyperparameters
BASELINE_LR=0.004
BASELINE_MOMENTUM=0.9
BASELINE_BATCH=16

# Intervention hyperparameters (the "B" values)
INTERVENTION_LR_VALUE=0.003
INTERVENTION_MOMENTUM_VALUE=0.8
INTERVENTION_BATCH_VALUE=32

# Create logs directory if needed
mkdir -p logs

echo "========================================"
echo "Training Interventions Experiment"
echo "========================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Total Steps: ${STEPS}"
echo "Intervention Step: ${INTERVENTION_STEP}"
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
    --max-steps ${STEPS} \
    --batch-sharpness \
    --experiment-tag ${EXPERIMENT_TAG} \
    ${INTERVENTION_ARGS}

echo "========================================"
echo "Training completed for ${EXPERIMENT_TAG}"
echo "========================================"
