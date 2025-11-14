#!/bin/bash
# Submit full batch gradient flow comparison runs
# Respects cluster job limits (2 jobs at a time)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPTS=(
    "$SCRIPT_DIR/full_batch_lr0p001.sbatch"   # Gradient flow (lr=0.001)
    "$SCRIPT_DIR/full_batch_lr0p02.sbatch"    # GD lr=0.02
    "$SCRIPT_DIR/full_batch_lr0p01.sbatch"    # GD lr=0.01
    "$SCRIPT_DIR/full_batch_lr0p00666666.sbatch"  # GD lr=0.00666666
    "$SCRIPT_DIR/full_batch_lr0p005.sbatch"   # GD lr=0.005
)

echo "Submitting full batch gradient flow comparison runs..."
echo "Total jobs to submit: ${#RUN_SCRIPTS[@]}"
echo ""

for script in "${RUN_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "Warning: Script $script not found, skipping..."
        continue
    fi
    
    # Extract job name from script to check if already submitted
    script_basename=$(basename "$script")
    job_name=$(grep "^#SBATCH --job-name=" "$script" | cut -d'=' -f2)
    
    if [ -z "$job_name" ]; then
        echo "Warning: Could not extract job name from $script, skipping..."
        continue
    fi
    
    # Check if this job is currently running or recently completed successfully
    job_status=$(sacct -u $USER --format=JobName,State --starttime=$(date +%Y-%m-%d)T00:00:00 2>/dev/null | awk -v name="$job_name" '$1 == name {state=$2; found=1} END {if(found) print state; else print "none"}')
    
    # Only skip if job is currently running or completed successfully
    if [ "$job_status" == "RUNNING" ] || [ "$job_status" == "COMPLETED" ]; then
        echo "Skipping $job_name (status: $job_status)"
        continue
    fi
    
    # If job failed or is out of memory, allow resubmission
    if [ "$job_status" != "none" ] && [ "$job_status" != "RUNNING" ] && [ "$job_status" != "COMPLETED" ]; then
        echo "Previous $job_name had status: $job_status, resubmitting..."
    fi
    
    # Check current job count - gpu_test partition has limit of 2 jobs per user
    while true; do
        current_jobs=$(squeue -u $USER -h | wc -l)
        
        # Must have fewer than 2 jobs to submit (limit is 2)
        if [ "$current_jobs" -lt 2 ]; then
            echo "Submitting $job_name (current jobs: $current_jobs)..."
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
                        sleep 10
                        continue
                    fi
                else
                    echo "  ✗ Could not parse job ID from: $submit_output"
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
    sleep 10
done

echo ""
echo "All jobs submitted!"

