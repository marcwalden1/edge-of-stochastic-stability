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
#SBATCH --array=0-19

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:$PATH

export WANDB_MODE=offline

# SST-2 SGDM small-batch sweep: b=16, momentum=0.5, 600k steps.
# 20 learning rates log-spaced so that 1/lr spans [25, 500].
# Expected batch sharpness stabilization: 2*(1+beta)/lr = 3/lr.
#
#  ID  1/lr    lr        target BSharp
#   0   25   0.04000       75
#   1   29   0.03416       87
#   2   34   0.02919      102
#   3   40   0.02493      120
#   4   47   0.02130      141
#   5   55   0.01820      164
#   6   64   0.01556      192
#   7   75   0.01330      225
#   8   88   0.01136      264
#   9  103   0.00970      309
#  10  121   0.00829      362
#  11  141   0.00708      424
#  12  165   0.00605      497
#  13  194   0.00517      582
#  14  226   0.00441      679
#  15  265   0.00377      795
#  16  310   0.00322      930
#  17  363   0.00275     1089
#  18  425   0.00235     1275
#  19  500   0.00200     1500

LRS=(0.04000 0.03416 0.02919 0.02493 0.02130 0.01820 0.01556 0.01330 0.01136 0.00970 0.00829 0.00708 0.00605 0.00517 0.00441 0.00377 0.00322 0.00275 0.00235 0.00200)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

COMMON="--dataset sst2 --model sst_transformer --loss mse \
  --batch 16 --steps 600000 --num-data 5000 \
  --momentum 0.5 \
  --dataset-seed 0 --init-seed 0 \
  --lambdamax --batch-sharpness"

python training.py $COMMON --lr $LR --wandb-tag sst-sgdm-bsharp-sweep
