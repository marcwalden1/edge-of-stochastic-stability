#!/bin/bash
#SBATCH --job-name=vit_experiment
#SBATCH --output=logs/vit_experiment.out
#SBATCH --error=logs/vit_experiment.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

export WANDB_MODE=offline
python training.py --dataset cifar10 --lr 0.0001 --adam --model vit --batch 128 \
  --steps 150000 --num-data 8192 \
  --dataset-seed 111 \
  --stop-loss 0.00001 \
  --batch-sharpness --lambdamax \
