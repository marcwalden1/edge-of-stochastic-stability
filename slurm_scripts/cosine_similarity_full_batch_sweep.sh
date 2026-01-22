mkdir -p logs
for BATCH in \
2 3 4 5 6 7 8 9 \
10 11 13 15 17 19 22 25 \
28 32 36 41 47 54 62 71 \
82 94 108 124 143 165 190 219 \
252 291 336 388 448 517 597 689 \
795 918 1059 1222 1409 1626 1876 2165 \
2500 2887 3334 3851 4447 5130 5912 6810 \
7849 8192
do
cat > batch_size_${BATCH}.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=eoss_bs_${BATCH}
#SBATCH --output=logs/bs_${BATCH}.out
#SBATCH --error=logs/bs_${BATCH}.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
export WANDB_MODE=offline
python training.py --dataset cifar10 --model cnn --momentum 0.9 --batch ${BATCH} --lr 0.001 \\
  --steps 25000 --num-data 8192 --activation silu \\
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \\
  --stop-loss 0.00001 \\
  --batch-sharpness --lambdamax \\
  --disable-wandb
EOF
done

