#!/bin/bash
#SBATCH --job-name=sst_b128
#SBATCH --output=logs/sst_b128_%a.out
#SBATCH --error=logs/sst_b128_%a.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-1

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:$PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

# SST-2 SGDM large-batch runs: b=128, momentum=0.5, 200k steps, 8192 data.
# 2 intermediate LRs from the b=16 sweep range.
#
#  ID  1/lr    lr
#   0   90   0.01108
#   1  212   0.00471

LRS=(0.01108 0.00471)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

COMMON="--dataset sst2 --model sst_transformer --loss mse \
  --batch 128 --steps 200000 --num-data 8192 \
  --momentum 0.5 \
  --dataset-seed 0 --init-seed 0 \
  --lambdamax --batch-sharpness"

python training.py $COMMON --lr $LR --wandb-tag sst-sgdm-bsharp-sweep-v3
