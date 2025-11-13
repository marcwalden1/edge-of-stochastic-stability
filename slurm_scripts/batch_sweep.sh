#!/bin/bash
# Generate batch_size_*.sbatch files identical to train_job.sh, just with varying batch sizes

# Define batch sizes
BATCH_SIZES=( \
2 3 4 5 6 7 8 9 \
10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 \
120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 660 680 700 \
800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 \
)

SCRIPT_DIR="$HOME/edge-of-stochastic-stability/slurm_scripts"
echo "Generating batch_size_*.sbatch files in $SCRIPT_DIR"
echo "Total batch sizes: ${#BATCH_SIZES[@]}"
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    script_path="${SCRIPT_DIR}/batch_size_${batch_size}.sbatch"

    cat > "$script_path" <<EOF
#!/bin/bash
#SBATCH --job-name=bs${batch_size}
#SBATCH --output=eoss_cifar10_bs${batch_size}.out
#SBATCH --error=eoss_cifar10_bs${batch_size}.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

python training.py --dataset cifar10 --model mlp --batch ${batch_size} --lr 0.01 \
  --steps 150000 --num-data 8192 \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --stop-loss 0.00001 \
  --lambdamax --batch-sharpness
EOF

    chmod +x "$script_path"
    echo "Generated: batch_size_${batch_size}.sbatch"
done

echo ""
echo "Generated ${#BATCH_SIZES[@]} batch_size_*.sbatch files."