#!/bin/bash
#SBATCH --job-name=eoss_cifar10
#SBATCH --output=eoss_cifar10.out
#SBATCH --error=eoss_cifar10.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
export WANDB_MODE=offline
python training.py --dataset cifar10 --model mlp --batch 16 --lr 0.02 \
  --steps 150000 --num-data 8192 --activation relu \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 \
  --batch-sharpness
echo \"Local trajectory snapshots:\"
ls -1 \"$RESULTS/wandb/local_projected_weights\" || echo \"No snapshots directory found\"
# Peek into one artifact
for step in 000000 000256 000512; do
  d=\"$RESULTS/wandb/local_projected_weights/projected_weights_step_${step}\"
  if [ -d \"$d\" ]; then
    echo \"Contents of $d:\"
    ls -lh \"$d\"
  fi
done
