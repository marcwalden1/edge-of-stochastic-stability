#!/bin/bash
#SBATCH --job-name=cos_sim_sweep
#SBATCH --output=cos_sim_sweep_%A_%a.out
#SBATCH --error=cos_sim_sweep_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-7

# Batch sizes to sweep
BATCH_SIZES=(1 2 4 16 32 64 128 1024)
BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

# Learning rate scaled with batch size (base lr = 0.01 at batch=64)
# Using linear scaling: lr = base_lr * batch_size / 64

LR=0.004

echo "Running cosine similarity tracking with batch_size=$BATCH_SIZE, lr=$LR"

export WANDB_MODE=offline

python training.py \
  --dataset cifar10 \
  --model mlp \
  --batch $BATCH_SIZE \
  --lr $LR \
  --steps 21000 \
  --num-data 8192 \
  --activation relu \
  --init-scale 0.2 \
  --dataset-seed 111 \
  --init-seed 8312 \
  --momentum 0.9 \
  --cosine-similarity \
  --disable-wandb

echo "Completed batch_size=$BATCH_SIZE"
