#!/bin/bash
#SBATCH --job-name=muon_beta1_sweep
#SBATCH --output=logs/muon_beta1_sweep_%a.out
#SBATCH --error=logs/muon_beta1_sweep_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-11

# 12 runs: Muon, lr=0.02, muon-momentum=(0, 0.25, 0.5, 0.75, 0.9, 0.99)
#   indices  0-5:  batch=8
#   indices  6-11: batch=8192 (full batch)
#
# lr=0.02 (Muon default) rather than Adam's 0.00002:
# Newton-Schulz orthogonalization produces updates of norm ~sqrt(n_cols),
# so Muon requires a much larger lr than Adam to take comparable-sized steps.
#
# --lambdamaxpreconditioned, --adaptive-batch-sharpness, and
# --adaptive-batch-sharpness-momentum are intentionally omitted:
# Muon has no diagonal preconditioner and training.py raises ValueError
# if those flags are combined with --muon.

MOMENTUM_VALUES=(0 0.25 0.5 0.75 0.9 0.99)

ID=${SLURM_ARRAY_TASK_ID}
MUON_MOM=${MOMENTUM_VALUES[$((ID % 6))]}

if [ "$ID" -lt 6 ]; then
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
  --lr 0.02 --batch "$BATCH" \
  --batch-sharpness --lambdamax \
  --GBS \
  --muon --muon-momentum "$MUON_MOM"
