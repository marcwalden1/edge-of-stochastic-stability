#!/bin/bash
# Sweep 1 batch size sweep: LN + RS, lr=0.007, mom=0.5, init=0.2
# 32 batch sizes log-spaced in [2, 8192]
# batch < 64: 600k steps; batch >= 64: 150k steps
# batch >= 3663: kempner_requeue H200; all others: kempner_requeue any GPU

mkdir -p logs

JOB_IDS=""

for BATCH in 2 3 4 5 6 8 10 13 17 22 29 38 50 65 86 112 146 191 250 327 428 560 732 958 1252 1638 2142 2801 3663 4790 6264 8192; do

  if [ "$BATCH" -lt 64 ]; then
    STEPS=600000
    TIME="24:00:00"
  else
    STEPS=150000
    TIME="12:00:00"
  fi

  if [ "$BATCH" -ge 3663 ]; then
    GRES="gpu:nvidia_h200:1"
  else
    GRES="gpu:1"
  fi

  cat > logs/vit_ln_rs_lr0p007_b${BATCH}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vit_ln_rs_b${BATCH}
#SBATCH --output=logs/vit_ln_rs_lr0p007_b${BATCH}.out
#SBATCH --error=logs/vit_ln_rs_lr0p007_b${BATCH}.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=${GRES}
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=${TIME}
#SBATCH --requeue

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:\$PATH
export WANDB_MODE=offline

python training.py --dataset cifar10 --model vit \\
  --momentum 0.5 --lr 0.007 --batch ${BATCH} \\
  --steps ${STEPS} --num-data 8192 \\
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --layernorm --residual-scaling \\
  --batch-sharpness --lambdamax
EOF

  JOB_ID=$(sbatch --parsable logs/vit_ln_rs_lr0p007_b${BATCH}.sbatch)
  JOB_IDS="${JOB_IDS}:${JOB_ID}"
  echo "Submitted job ${JOB_ID} for batch=${BATCH} steps=${STEPS} gres=${GRES}"

done

# Sentinel: email when all 32 jobs finish
DEPS="${JOB_IDS#:}"
sbatch --dependency=afterany:${DEPS} \
       --job-name=vit_ln_rs_bsweep_done \
       --output=logs/vit_ln_rs_bsweep_done.out \
       --partition=kempner_requeue \
       --account=kempner_kdbrantley_lab \
       --gres=gpu:1 --mem=4G --cpus-per-task=1 --time=00:05:00 \
       --mail-type=END \
       --mail-user=marcwalden@g.harvard.edu \
       --wrap="echo 'Batch size sweep (LN+RS, lr=0.007, 32 jobs) finished.'"
