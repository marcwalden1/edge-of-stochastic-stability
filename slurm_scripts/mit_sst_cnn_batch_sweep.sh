#!/bin/bash
#SBATCH --job-name=sst_cnn_batch_sweep
#SBATCH --output=logs/sst_cnn_batch_sweep_%a.out
#SBATCH --error=logs/sst_cnn_batch_sweep_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-9%2

eval "$(conda shell.bash hook)"
conda activate eoss

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

# CNN batch sweep: fixed lr=0.008, mom=0.9. Target BS = 2(1-β)/lr = 0.2/0.008 = 25.
#
#  ID   model     b      steps
#   0   sst_cnn   2      150000
#   1   sst_cnn   4      150000
#   2   sst_cnn   6      150000
#   3   sst_cnn   8      150000
#   4   sst_cnn   16      75000
#   5   sst_cnn   32      75000
#   6   sst_cnn   64      75000
#   7   sst_cnn   128     75000
#   8   sst_cnn   256     75000
#   9   sst_cnn   8192    75000

BATCH_SIZES=(2 4 6 8 16 32 64 128 256 8192)
B=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

if [ "$B" -lt 16 ]; then
  STEPS=150000
else
  STEPS=75000
fi

mkdir -p logs

python training.py \
  --dataset sst2 --loss mse \
  --model sst_cnn \
  --batch $B --steps $STEPS --num-data 8192 \
  --lr 0.008 --momentum 0.9 \
  --dataset-seed 0 --init-seed 0 \
  --lambdamax --batch-sharpness \
  --wandb-tag sst-batch-sweep
