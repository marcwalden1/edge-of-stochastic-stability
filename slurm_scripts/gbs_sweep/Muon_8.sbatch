#!/bin/bash
#SBATCH --job-name=eoss_gbs_Muon_8
#SBATCH --output=/n/home06/mwalden/edge-of-stochastic-stability/slurm_scripts/gbs_sweep/logs/Muon_8.out
#SBATCH --error=/n/home06/mwalden/edge-of-stochastic-stability/slurm_scripts/gbs_sweep/logs/Muon_8.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00




# Results go to SCRATCH, not home directory
export RESULTS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/results
export DATASETS=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/datasets

set -euo pipefail

echo "Starting GBS sweep run: OPT=Muon  BATCH=8"
echo "Results dir: $RESULTS"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

cd /n/home06/mwalden/edge-of-stochastic-stability

/n/home06/mwalden/.conda/envs/eoss/bin/python training.py \
    --dataset cifar10 --model mlp --steps 150000 --num-data 8192 --activation silu --init-scale 0.2 --dataset-seed 111 --init-seed 8312 --stop-loss 0.00001 --gbs-suite --batch-sharpness --lambdamax --disable-wandb --checkpoint-every 5000 --experiment-subdir gbs_sweep \
    --batch 8 \
    --lr 0.001 --muon --muon-momentum 0.95 \
    --experiment-tag Muon_8

echo "Done: Muon_8"
date
