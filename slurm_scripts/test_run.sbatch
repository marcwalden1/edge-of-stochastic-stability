#!/bin/bash
#SBATCH --job-name=eoss_test
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Step 1: Baseline test with higher LR, fewer steps, and WANDB disabled
# This tests if training can complete at all under favorable conditions

export WANDB_MODE=offline
python training.py --dataset cifar10 --model cnn --momentum 0.9 --batch 128 --lr 0.005 \
  --steps 5000 --num-data 8192 --activation silu \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 \
  --batch-sharpness --lambdamax \
  --disable-wandb

echo "Step 1 test completed!"
