#!/bin/bash
# SGD (no momentum), LN+RS, lr=0.005, b={4096,8192}, 150k steps, H200

mkdir -p logs

JOB_IDS=""

for BATCH in 4096 8192; do

  cat > logs/vit_ln_rs_sgd_lr0p005_b${BATCH}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vit_ln_rs_sgd_b${BATCH}
#SBATCH --output=logs/vit_ln_rs_sgd_lr0p005_b${BATCH}.out
#SBATCH --error=logs/vit_ln_rs_sgd_lr0p005_b${BATCH}.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --requeue

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:\$PATH
export WANDB_MODE=offline

python training.py --dataset cifar10 --model vit \\
  --lr 0.005 --batch ${BATCH} \\
  --steps 150000 --num-data 8192 \\
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --layernorm --residual-scaling \\
  --batch-sharpness --lambdamax
EOF

  JOB_ID=$(sbatch --parsable logs/vit_ln_rs_sgd_lr0p005_b${BATCH}.sbatch)
  JOB_IDS="${JOB_IDS}:${JOB_ID}"
  echo "Submitted job ${JOB_ID} for batch=${BATCH}"

done

# Sentinel: email when both jobs finish
DEPS="${JOB_IDS#:}"
sbatch --dependency=afterany:${DEPS} \
       --job-name=vit_ln_rs_sgd_done \
       --output=logs/vit_ln_rs_sgd_done.out \
       --partition=kempner_requeue \
       --account=kempner_kdbrantley_lab \
       --gres=gpu:1 --mem=4G --cpus-per-task=1 --time=00:05:00 \
       --mail-type=END \
       --mail-user=marcwalden@g.harvard.edu \
       --wrap="echo 'SGD LN+RS lr=0.005 b={4096,8192} finished.'"
