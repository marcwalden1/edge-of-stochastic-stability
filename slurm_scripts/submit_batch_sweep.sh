#!/bin/bash
# Automatic job submission script for batch size sweep
# Submits jobs one at a time, waiting for slots to become available

BATCH_SIZES=( \
2 3 4 5 6 7 8 9 \
10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 \
120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 660 680 700 720 740 760 780 800 820 840 860 880 900 920 940 960 980 \
1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 \
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
    
    # Check if this batch size has already been submitted today
    job_name="bs${batch_size}"
    already_submitted=$(sacct -u $USER --format=JobName,State --starttime=$(date +%Y-%m-%d)T00:00:00 2>/dev/null | awk -v name="$job_name" '$1 == name {found=1} END {if(found) print "yes"; else print "no"}')
    
    if [ "$already_submitted" == "yes" ]; then
        echo "Skipping batch size $batch_size (already submitted today)"
        continue
    fi
    
    # Check current job count - gpu_test partition has limit of 2 jobs per user
    while true; do
        current_jobs=$(squeue -u $USER -h | wc -l)
        
        # Must have fewer than 2 jobs to submit (limit is 2)
        if [ "$current_jobs" -lt 2 ]; then
            echo "Submitting batch size $batch_size (current jobs: $current_jobs)..."
            submit_output=$(sbatch "$script" 2>&1)
            submit_exit=$?
            
            if [ $submit_exit -eq 0 ]; then
                job_id=$(echo "$submit_output" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
                if [ -n "$job_id" ]; then
                    # Wait a moment for SLURM to register the job
                    sleep 3
                    # Verify the job is actually in the queue
                    if squeue -j "$job_id" -h &>/dev/null; then
                        echo "  ✓ Submitted successfully (Job ID: $job_id)"
                    else
                        echo "  ⚠ Job $job_id submitted but not in queue - may have failed immediately"
                        echo "     Check with: sacct -j $job_id"
                        # Wait longer before retrying
                        sleep 10
                        continue
                    fi
                else
                    echo "  ✗ Could not parse job ID from: $submit_output"
                    echo "     Output: $submit_output"
                    sleep 10
                    continue
                fi
            else
                echo "  ✗ Submission failed: $submit_output"
                echo "  Waiting before retry..."
                sleep 15
                continue
            fi
            break
        else
            echo "Waiting for job slot (current: $current_jobs jobs, limit: 2)..."
            sleep 30  # Check every 30 seconds
        fi
    done
    
    # Delay between submissions to avoid hitting rate limits
    # Give SLURM time to process the submission
    sleep 10
done

echo ""
echo "All jobs submitted!"

