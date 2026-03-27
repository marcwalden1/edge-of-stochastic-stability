#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=logs/smoke_test.out
#SBATCH --error=logs/smoke_test.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:$PATH
export WANDB_MODE=offline

COMMON_GPT="--dataset shakespeare --model gpt --loss lm --batch 64
  --steps 20 --num-data 256
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312
  --batch-sharpness --lambdamax --disable-wandb"

COMMON_VIT="--dataset cifar10 --model vit --batch 64
  --steps 20 --num-data 256
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312
  --batch-sharpness --lambdamax --disable-wandb"

echo ">>> [1/4] GPT SGDM"
python training.py $COMMON_GPT --momentum 0.95 --lr 0.02
echo ">>> [2/4] GPT Adam"
python training.py $COMMON_GPT --lambdamaxpreconditioned --adam --lr 0.001
echo ">>> [3/4] ViT SGDM"
python training.py $COMMON_VIT --momentum 0.95 --lr 0.001
echo ">>> [4/4] ViT Adam"
python training.py $COMMON_VIT --lambdamaxpreconditioned --adam --lr 0.001
echo ">>> All smoke tests passed"
