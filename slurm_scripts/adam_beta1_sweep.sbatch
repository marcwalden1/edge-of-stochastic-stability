#!/bin/bash
#SBATCH --job-name=adam_beta1_sweep
#SBATCH --output=logs/adam_beta1_sweep_%a.out
#SBATCH --error=logs/adam_beta1_sweep_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-13

# 14 runs: Adam, lr=0.00002, beta2=0.99, beta1=(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
#   indices  0-6:  batch=8
#   indices  7-13: batch=8192 (full batch)

BETA1_VALUES=(0 0.25 0.5 0.75 0.9 0.95 0.99)

ID=${SLURM_ARRAY_TASK_ID}
BETA1=${BETA1_VALUES[$((ID % 7))]}

if [ "$ID" -lt 7 ]; then
    BATCH=8
else
    BATCH=8192
fi

export WANDB_MODE=offline

python training.py \
  --dataset cifar10 --model mlp \
  --steps 150000 --num-data 8192 --activation silu \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 \
  --lr 0.00002 --batch "$BATCH" \
  --batch-sharpness --lambdamax \
  --lambdamaxpreconditioned --adaptive-batch-sharpness-momentum \
  --adam "$BETA1" 0.99
