#!/bin/bash
#SBATCH --job-name=sst_batch_sweep
#SBATCH --output=logs/sst_batch_sweep_%a.out
#SBATCH --error=logs/sst_batch_sweep_%a.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-17

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:$PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

# Batch sweep: fixed lr=0.008, mom=0.9. Target BS = 2(1-β)/lr = 0.2/0.008 = 25.
# 9 batch sizes x 2 models = 18 array jobs.
#
#  ID   model     b
#   0   sst_mlp   2
#   1   sst_mlp   4
#   2   sst_mlp   6
#   3   sst_mlp   8
#   4   sst_mlp   16
#   5   sst_mlp   32
#   6   sst_mlp   128
#   7   sst_mlp   256
#   8   sst_mlp   8192
#   9   sst_cnn   2
#  10   sst_cnn   4
#  11   sst_cnn   6
#  12   sst_cnn   8
#  13   sst_cnn   16
#  14   sst_cnn   32
#  15   sst_cnn   128
#  16   sst_cnn   256
#  17   sst_cnn   8192

BATCH_SIZES=(2 4 6 8 16 32 128 256 8192)
MODELS=(sst_mlp sst_cnn)

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 9))
B_INDEX=$((SLURM_ARRAY_TASK_ID % 9))

MODEL=${MODELS[$MODEL_INDEX]}
B=${BATCH_SIZES[$B_INDEX]}

if [ "$B" -lt 16 ]; then
  STEPS=150000
else
  STEPS=75000
fi

python training.py \
  --dataset sst2 --loss mse \
  --model $MODEL \
  --batch $B --steps $STEPS --num-data 8192 \
  --lr 0.008 --momentum 0.9 \
  --dataset-seed 0 --init-seed 0 \
  --lambdamax --batch-sharpness \
  --wandb-tag sst-batch-sweep
