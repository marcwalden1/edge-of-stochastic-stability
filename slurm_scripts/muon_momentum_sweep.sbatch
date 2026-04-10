#!/bin/bash
#SBATCH --job-name=muon_mom_sweep
#SBATCH --output=logs/muon_mom_sweep_%a.out
#SBATCH --error=logs/muon_mom_sweep_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-11

# 12 runs: Muon, lr=0.004, momentum=(0, 0.25, 0.5, 0.75, 0.9, 0.95)
#   indices  0-5:  batch=8
#   indices  6-11: batch=8192 (full batch)

MOM_VALUES=(0 0.25 0.5 0.75 0.9 0.95)

ID=${SLURM_ARRAY_TASK_ID}
MOM=${MOM_VALUES[$((ID % 6))]}

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
  --lr 0.004 --batch "$BATCH" \
  --batch-sharpness --lambdamax \
  --muon --muon-momentum "$MOM"
