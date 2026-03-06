#!/bin/bash
#SBATCH --job-name=gbs_sweep
#SBATCH --output=logs/gbs_sweep_%a.out
#SBATCH --error=logs/gbs_sweep_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-43

# 44 runs total:
#   SGD:    lr=(0.02, 0.005)   x batch=(8, 64, 256, 1048)              =  8
#   SGDM:   lr=(0.002, 0.0005) x mom=(0.9, 0.8) x batch=(8, 256, 1048) = 12
#   RMSProp:lr=(0.00003, 5e-5) x a=(0.99, 0.9)  x batch=(64, 1048)     =  8
#   Adam:   lr=(0.00003, 5e-5) x b1=(0.9, 0.8)  x b2=(0.999, 0.9)
#                              x batch=(64, 1048)                       = 16

OPTIMIZER_ARGS=(
    # --- SGD (indices 0-7) ---
    "--lr 0.02   --batch 8"
    "--lr 0.02   --batch 64"
    "--lr 0.02   --batch 256"
    "--lr 0.02   --batch 1048"
    "--lr 0.005  --batch 8"
    "--lr 0.005  --batch 64"
    "--lr 0.005  --batch 256"
    "--lr 0.005  --batch 1048"
    # --- SGDM (indices 8-19) ---
    "--lr 0.002   --momentum 0.9 --batch 8"
    "--lr 0.002   --momentum 0.9 --batch 256"
    "--lr 0.002   --momentum 0.9 --batch 1048"
    "--lr 0.002   --momentum 0.8 --batch 8"
    "--lr 0.002   --momentum 0.8 --batch 256"
    "--lr 0.002   --momentum 0.8 --batch 1048"
    "--lr 0.0005  --momentum 0.9 --batch 8"
    "--lr 0.0005  --momentum 0.9 --batch 256"
    "--lr 0.0005  --momentum 0.9 --batch 1048"
    "--lr 0.0005  --momentum 0.8 --batch 8"
    "--lr 0.0005  --momentum 0.8 --batch 256"
    "--lr 0.0005  --momentum 0.8 --batch 1048"
    # --- RMSProp (indices 20-27) ---
    "--lr 0.00003 --rmsprop 0.99 --batch 64   --adaptive-batch-sharpness --lambdamaxpreconditioned"
    "--lr 0.00003 --rmsprop 0.99 --batch 1048 --adaptive-batch-sharpness --lambdamaxpreconditioned"
    "--lr 0.00003 --rmsprop 0.9  --batch 64   --adaptive-batch-sharpness --lambdamaxpreconditioned"
    "--lr 0.00003 --rmsprop 0.9  --batch 1048 --adaptive-batch-sharpness --lambdamaxpreconditioned"
    "--lr 0.00005 --rmsprop 0.99 --batch 64   --adaptive-batch-sharpness --lambdamaxpreconditioned"
    "--lr 0.00005 --rmsprop 0.99 --batch 1048 --adaptive-batch-sharpness --lambdamaxpreconditioned"
    "--lr 0.00005 --rmsprop 0.9  --batch 64   --adaptive-batch-sharpness --lambdamaxpreconditioned"
    "--lr 0.00005 --rmsprop 0.9  --batch 1048 --adaptive-batch-sharpness --lambdamaxpreconditioned"
    # --- Adam (indices 28-43) ---
    "--lr 0.00003 --adam 0.9 0.999 --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00003 --adam 0.9 0.999 --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00003 --adam 0.9 0.9   --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00003 --adam 0.9 0.9   --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00003 --adam 0.8 0.999 --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00003 --adam 0.8 0.999 --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00003 --adam 0.8 0.9   --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00003 --adam 0.8 0.9   --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.9 0.999 --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.9 0.999 --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.9 0.9   --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.9 0.9   --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.8 0.999 --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.8 0.999 --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.8 0.9   --batch 64   --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
    "--lr 0.00005 --adam 0.8 0.9   --batch 1048 --adaptive-batch-sharpness-momentum --lambdamaxpreconditioned"
)

export WANDB_MODE=offline

python training.py \
  --dataset cifar10 --model mlp \
  --steps 150000 --num-data 8192 --activation silu \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 \
  --batch-sharpness --lambdamax \
  --GBS \
  ${OPTIMIZER_ARGS[$SLURM_ARRAY_TASK_ID]}
