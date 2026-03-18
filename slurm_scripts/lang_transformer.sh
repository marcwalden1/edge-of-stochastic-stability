#!/bin/bash
#SBATCH --job-name=gpt_experiment
#SBATCH --output=logs/gpt_experiment_%a.out
#SBATCH --error=logs/gpt_experiment_%a.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-1

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:$PATH

export WANDB_MODE=offline

COMMON="--dataset shakespeare --model gpt --loss lm --batch 256 \
  --steps 100000 --num-data 7781 \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --batch-sharpness --lambdamax"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python training.py $COMMON --momentum 0.95 --lr 0.02 --wandb-tag lang-transformer
else
  python training.py $COMMON --lambdamaxpreconditioned --adam --lr 0.001 --wandb-tag lang-transformer
fi
