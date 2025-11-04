#!/bin/bash
# Automatic job submission script for batch size sweep
# Submits jobs one at a time, waiting for slots to become available

# Submit only batch sizes 10 through 100 stepping by 4 (10,14,...,98)
BATCH_SIZES=(10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98)
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

