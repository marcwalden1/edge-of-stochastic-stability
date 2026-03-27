#!/bin/bash
# Sweep 3: Frozen LN + RS — LN enabled but weight/bias frozen, + residual-scaling
# init-scale=0.2, beta=0.5, b=8, 100k steps, 8 LRs so 1/lr in [25, 500]
# Distinguishing name prefix: vit_frozenln_rs

mkdir -p logs

JOB_IDS=""

for LR in 0.002 0.003 0.005 0.007 0.01 0.015 0.02 0.04; do

  LR_TAG=$(echo $LR | tr '.' 'p')

  cat > logs/vit_frozenln_rs_b05_b8_lr${LR_TAG}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vit_frozenln_rs_lr${LR_TAG}
#SBATCH --output=logs/vit_frozenln_rs_b05_b8_lr${LR_TAG}.out
#SBATCH --error=logs/vit_frozenln_rs_b05_b8_lr${LR_TAG}.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --requeue

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:\$PATH
export WANDB_MODE=offline

python training.py --dataset cifar10 --model vit \\
  --momentum 0.5 --lr ${LR} --batch 8 \\
  --steps 100000 --num-data 8192 \\
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --frozen-layernorm --residual-scaling \\
  --batch-sharpness --lambdamax
EOF

  JOB_ID=$(sbatch --parsable logs/vit_frozenln_rs_b05_b8_lr${LR_TAG}.sbatch)
  JOB_IDS="${JOB_IDS}:${JOB_ID}"
  echo "Submitted job ${JOB_ID} for LR=${LR}"

done

# Sentinel: email when all 8 jobs finish
DEPS="${JOB_IDS#:}"
sbatch --dependency=afterany:${DEPS} \
       --job-name=vit_frozenln_rs_done \
       --output=logs/vit_frozenln_rs_done.out \
       --partition=kempner_requeue \
       --account=kempner_kdbrantley_lab \
       --gres=gpu:1 --mem=4G --cpus-per-task=1 --time=00:05:00 \
       --mail-type=END \
       --mail-user=marcwalden@g.harvard.edu \
       --wrap="echo 'Sweep 3 (Frozen-LN+RS, 8 jobs) finished.'"
