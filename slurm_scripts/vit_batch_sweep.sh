#!/bin/bash
# Sweep 1 config (LN+RS, lr=0.007, mom=0.5, init=0.2) for b={3,4,5,6,7,8}, 1.25M steps, H200

mkdir -p logs

JOB_IDS=""

for BATCH in 3 4 5 6 7 8 9 10; do

  cat > logs/vit_ln_rs_lr0p007_b${BATCH}_1p25M.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vit_ln_rs_b${BATCH}_1p25M
#SBATCH --output=logs/vit_ln_rs_lr0p007_b${BATCH}_1p25M.out
#SBATCH --error=logs/vit_ln_rs_lr0p007_b${BATCH}_1p25M.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --requeue
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00

echo "Starting batch size: ${BATCH}"

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:\$PATH
export WANDB_MODE=offline

python training.py --dataset cifar10 --model vit \\
  --momentum 0.5 --lr 0.007 --batch ${BATCH} \\
  --steps 1250000 --num-data 8192 \\
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --layernorm --residual-scaling \\
  --batch-sharpness --lambdamax
EOF

  JOB_ID=$(sbatch --parsable logs/vit_ln_rs_lr0p007_b${BATCH}_1p25M.sbatch)
  JOB_IDS="${JOB_IDS}:${JOB_ID}"
  echo "Submitted job ${JOB_ID} for batch=${BATCH}"

done

# Sentinel: email when all 6 jobs finish
DEPS="${JOB_IDS#:}"
sbatch --dependency=afterany:${DEPS} \
       --job-name=vit_ln_rs_1p25M_done \
       --output=logs/vit_ln_rs_1p25M_done.out \
       --partition=kempner_requeue \
       --account=kempner_kdbrantley_lab \
       --gres=gpu:1 --mem=4G --cpus-per-task=1 --time=00:05:00 \
       --mail-type=END \
       --mail-user=marcwalden@g.harvard.edu \
       --wrap="echo 'LN+RS b={3..8} 1.25M steps finished.'"
