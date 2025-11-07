#!/bin/bash
# Automatic job submission script for batch size sweep
# Submits jobs one at a time, waiting for slots to become available


BATCH_SIZES=( \
2 3 4 5 6 7 8 9 \
10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 \
100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 660 680 700 720 740 760 780 800 820 840 860 880 900 920 940 960 980 \
1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 \
)
SCRIPT_DIR="slurm_scripts"

echo "Starting automatic submission of batch size sweep jobs..."
echo "Total jobs to submit: ${#BATCH_SIZES[@]}"
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    script="${SCRIPT_DIR}/batch_size_${batch_size}.sbatch"
    
    if [ ! -f "$script" ]; then
        echo "Warning: Script $script not found, skipping..."
        continue
    fi
    
    # Check current job count
    while true; do
        current_jobs=$(squeue -u $USER -h | wc -l)
        
        # Adjust this number based on your limit (e.g., if limit is 2, use -lt 2)
        if [ "$current_jobs" -lt 2 ]; then
            echo "Submitting batch size $batch_size (current jobs: $current_jobs)..."
            submit_output=$(sbatch "$script" 2>&1)
            submit_exit=$?
            
            if [ $submit_exit -eq 0 ]; then
                job_id=$(echo "$submit_output" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
                if [ -n "$job_id" ]; then
                    echo "  ✓ Submitted successfully (Job ID: $job_id)"
                else
                    echo "  ✗ Could not parse job ID from: $submit_output"
                fi
            else
                echo "  ✗ Submission failed: $submit_output"
                echo "  Waiting before retry..."
                sleep 10
                continue
            fi
            break
        else
            echo "Waiting for job slot (current: $current_jobs jobs)..."
            sleep 30  # Check every 30 seconds
        fi
    done
    
    # Small delay between submissions
    sleep 2
done

echo ""
echo "All jobs submitted!"

