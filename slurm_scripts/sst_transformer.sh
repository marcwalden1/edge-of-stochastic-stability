#!/bin/bash
#SBATCH --job-name=sst_experiment
#SBATCH --output=logs/sst_experiment_%a.out
#SBATCH --error=logs/sst_experiment_%a.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-7

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:$PATH

export WANDB_MODE=offline

# SST-2 SGDM small-batch sweep: b=16, momentum=0.5, 300k steps.
# 8 learning rates log-spaced so that 1/lr spans [25, 500].
# tok_emb frozen; expected batch sharpness stabilization: 2*(1-beta)/lr = 1/lr.
#
#  ID  1/lr    lr        target BSharp
#   0   25   0.04000       25
#   1   38   0.02607       38
#   2   59   0.01700       59
#   3   90   0.01108       90
#   4  138   0.00722      138
#   5  212   0.00471      212
#   6  326   0.00307      326
#   7  500   0.00200      500

LRS=(0.04000 0.02607 0.01700 0.01108 0.00722 0.00471 0.00307 0.00200)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

COMMON="--dataset sst2 --model sst_transformer --loss mse \
  --batch 16 --steps 300000 --num-data 5000 \
  --momentum 0.5 \
  --dataset-seed 0 --init-seed 0 \
  --lambdamax --batch-sharpness"

python training.py $COMMON --lr $LR --wandb-tag sst-sgdm-bsharp-sweep-v2
