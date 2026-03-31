#!/bin/bash
# MLP/CNN batch sweep — paper appendix labels A2, A4, A5, A8, A10
# A2:  MLP  relu  lr=0.01  mom=0.9  b={2,4,6,8,16,32,64}
# A4:  CNN  relu  lr=0.02  mom=0.5  b={2,4,6,8,16,32,64,128,256,8192}
# A5:  CNN  silu  lr=0.01  mom=0.5  b={2,4}
# A8:  CNN  silu  lr=0.02  mom=0.8  b={2,4}
# A10: MLP  relu  lr=0.01  mom=0.5  b={2}

mkdir -p logs

COMMON="--dataset cifar10 --num-data 8192 --loss mse --steps 150000 \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 --batch-sharpness --lambdamax"

JOB_IDS=""

submit_job() {
  local LABEL=$1 MODEL=$2 ACT=$3 LR=$4 MOM=$5 BATCH=$6
  local NAME="${LABEL}_b${BATCH}"
  cat > logs/${NAME}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --output=logs/${NAME}.out
#SBATCH --error=logs/${NAME}.err
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --requeue

echo "Label: ${LABEL}  model: ${MODEL}  activation: ${ACT}  lr: ${LR}  mom: ${MOM}  batch: ${BATCH}"

export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:\$PATH
export WANDB_MODE=offline

python training.py $COMMON \\
  --model ${MODEL} --activation ${ACT} \\
  --lr ${LR} --momentum ${MOM} --batch ${BATCH} \\
  --experiment-subdir ${LABEL}
EOF
  JOB_ID=$(sbatch --parsable logs/${NAME}.sbatch)
  JOB_IDS="${JOB_IDS}:${JOB_ID}"
  echo "Submitted ${JOB_ID} — ${NAME}"
}

# A2: MLP relu, lr=0.01, mom=0.9
for B in 2 4 6 8 16 32 64; do
  submit_job A2 mlp relu 0.01 0.9 $B
done

# A4: CNN relu, lr=0.02, mom=0.5
for B in 2 4 6 8 16 32 64 128 256 8192; do
  submit_job A4 cnn relu 0.02 0.5 $B
done

# A5: CNN silu, lr=0.01, mom=0.5
for B in 2 4; do
  submit_job A5 cnn silu 0.01 0.5 $B
done

# A8: CNN silu, lr=0.02, mom=0.8
for B in 2 4; do
  submit_job A8 cnn silu 0.02 0.8 $B
done

# A10: MLP relu, lr=0.01, mom=0.5
submit_job A10 mlp relu 0.01 0.5 2

# Sentinel: email when all 22 jobs finish
DEPS="${JOB_IDS#:}"
sbatch --dependency=afterany:${DEPS} \
       --job-name=mlp_cnn_sweep_done \
       --output=logs/mlp_cnn_sweep_done.out \
       --partition=kempner_requeue \
       --account=kempner_kdbrantley_lab \
       --gres=gpu:1 --mem=4G --cpus-per-task=1 --time=00:05:00 \
       --mail-type=END \
       --mail-user=marcwalden@g.harvard.edu \
       --wrap="echo 'MLP/CNN sweep (A2, A4, A5, A8, A10) finished.'"
