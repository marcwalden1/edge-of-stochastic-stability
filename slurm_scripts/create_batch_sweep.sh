#!/bin/bash
# GPT Shakespeare SGDM batch sweep: 31 log-spaced batch sizes from 2 to 7781, num-data=7781.
# b <= 2312: kempner_requeue --constraint=h100
# b >  2312: kempner_requeue --constraint=h200

mkdir -p logs

for BATCH in \
2 3 4 6 8 10 13 17 22 29 \
37 49 64 83 109 142 185 242 316 413 \
539 703 918 1198 1564 2042 2666 3480 4545 5934 7781
do

if [ "$BATCH" -le 2312 ]; then
  CONSTRAINT="h100"
  MEM="96G"
else
  CONSTRAINT="h200"
  MEM="128G"
fi

cat > batch_size_${BATCH}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=eoss_bs_${BATCH}
#SBATCH --output=logs/bs_${BATCH}.out
#SBATCH --error=logs/bs_${BATCH}.err
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
export PATH=/n/home06/mwalden/.conda/envs/eoss/bin:\$PATH
export WANDB_MODE=offline
python training.py --dataset shakespeare --model gpt --loss lm --momentum 0.95 --batch ${BATCH} --lr 0.001 \\
  --steps 100000 --num-data 7781 \\
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --batch-sharpness --lambdamax \\
  --rare-measure
EOF

sbatch batch_size_${BATCH}.sbatch

done
