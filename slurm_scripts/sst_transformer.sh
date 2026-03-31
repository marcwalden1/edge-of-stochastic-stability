#!/bin/bash
#SBATCH --job-name=sst_experiment
#SBATCH --output=logs/sst_experiment_%a.out
#SBATCH --error=logs/sst_experiment_%a.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-7

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:$PATH
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

# SST-2 SGDM small-batch sweep: b=16, momentum=0.5, 200k steps, 8192 data.
# 10 LRs log-spaced so that 1/lr spans [16, 773] (extending original [25,500] by one step each end).
# tok_emb frozen (BERT proj); expected batch sharpness: 2*(1-beta)/lr = 1/lr.
#
#  ID  1/lr    lr        target BSharp
#   0   16   0.06180       16   <- extra large LR
#   1   25   0.04000       25   (running separately, v2)
#   2   38   0.02607       38   (running separately, v2)
#   3   59   0.01700       59
#   4   90   0.01108       90
#   5  138   0.00722      138
#   6  212   0.00471      212
#   7  326   0.00307      326
#   8  500   0.00200      500
#   9  773   0.00129      773   <- extra small LR

# This script submits IDs 0,3-9 (IDs 1-2 already running from v2 sweep).
# Submit on kempner_requeue.

LRS=(0.06180 0.04000 0.02607 0.01700 0.01108 0.00722 0.00471 0.00307 0.00200 0.00129)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

COMMON="--dataset sst2 --model sst_transformer --loss mse \
  --batch 16 --steps 200000 --num-data 8192 \
  --momentum 0.5 \
  --dataset-seed 0 --init-seed 0 \
  --lambdamax --batch-sharpness"

python training.py $COMMON --lr $LR --wandb-tag sst-sgdm-bsharp-sweep-v3
