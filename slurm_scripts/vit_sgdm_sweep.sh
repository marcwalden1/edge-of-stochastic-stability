#!/bin/bash
# Sweep 2: no LN, init-scale=1.0 — lambda_max_initial~948, so LRs must keep 1/lr > 948
# beta=0.5, b=8, 100k steps, 8 LRs so 1/lr in [1000, 10000]
# Distinguishing name prefix: vit_noln_is1p0

mkdir -p logs

JOB_IDS=""

for LR in 0.04 0.02 0.01 0.004 0.002; do

  LR_TAG=$(echo $LR | tr '.' 'p')

  cat > logs/vit_noln_is1p0_b05_b8_lr${LR_TAG}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vit_noln_is1p0_lr${LR_TAG}
#SBATCH --output=logs/vit_noln_is1p0_b05_b8_lr${LR_TAG}.out
#SBATCH --error=logs/vit_noln_is1p0_b05_b8_lr${LR_TAG}.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:\$PATH
export WANDB_MODE=offline

python training.py --dataset cifar10 --model vit \\
  --momentum 0.5 --lr ${LR} --batch 8 \\
  --steps 100000 --num-data 8192 \\
  --init-scale 1.0 --dataset-seed 111 --init-seed 8312 \\
  --batch-sharpness --lambdamax
EOF

  JOB_ID=$(sbatch --parsable logs/vit_noln_is1p0_b05_b8_lr${LR_TAG}.sbatch)
  JOB_IDS="${JOB_IDS}:${JOB_ID}"
  echo "Submitted job ${JOB_ID} for LR=${LR}"

done

# Sentinel: email when all 5 jobs finish
DEPS="${JOB_IDS#:}"
sbatch --dependency=afterany:${DEPS} \
       --job-name=vit_noln_is1p0_done \
       --output=logs/vit_noln_is1p0_done.out \
       --partition=kempner_requeue \
       --account=kempner_kdbrantley_lab \
       --gres=gpu:1 --mem=4G --cpus-per-task=1 --time=00:05:00 \
       --mail-type=END \
       --mail-user=marcwalden@g.harvard.edu \
       --wrap="echo 'Sweep 2 (no-LN init=1.0, 5 jobs) finished.'"
