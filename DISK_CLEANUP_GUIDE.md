# Disk Cleanup Guide for Cluster

If you're on the cluster and don't have the cleanup scripts, use these commands directly:

## Quick Disk Usage Check

```bash
# Check your quota
quota -s

# Check home directory size
du -sh ~

# Check project directory breakdown
cd ~/edge-of-stochastic-stability-1
du -sh */ 2>/dev/null | sort -hr | head -10

# Check wandb_checkpoints size (likely the biggest culprit)
du -sh wandb_checkpoints/*/ 2>/dev/null | sort -hr
du -sh wandb_checkpoints
```

## Quick Cleanup Commands (run these on the cluster)

### 1. Clean Python cache (always safe)
```bash
cd ~/edge-of-stochastic-stability-1
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
echo "Python cache cleaned"
```

### 2. Clean old SLURM logs (keeps last 20)
```bash
cd ~/edge-of-stochastic-stability-1
if [ -d "slurm" ]; then
    cd slurm
    ls -t *.out *.err 2>/dev/null | tail -n +21 | xargs rm -f 2>/dev/null
    echo "Old SLURM logs cleaned (kept 20 most recent)"
fi
```

### 3. Clean wandb checkpoints - Option A: Delete everything (if synced to wandb.ai)
```bash
cd ~/edge-of-stochastic-stability-1
# WARNING: Only do this if you've already synced to wandb.ai!
# rm -rf wandb_checkpoints
```

### 4. Clean wandb checkpoints - Option B: Keep only final checkpoint per run (recommended)
```bash
cd ~/edge-of-stochastic-stability-1
if [ -d "wandb_checkpoints" ]; then
    for run_dir in wandb_checkpoints/*/; do
        if [ -d "$run_dir" ]; then
            # Find the checkpoint with highest step number
            final_checkpoint=$(ls -1 "$run_dir"/checkpoint_step_*.pt 2>/dev/null | sort -V | tail -1)
            if [ -n "$final_checkpoint" ]; then
                # Delete all checkpoints except the final one and metadata
                find "$run_dir" -name "checkpoint_step_*.pt" ! -name "$(basename "$final_checkpoint")" -delete 2>/dev/null
                echo "Cleaned $(basename "$run_dir") - kept $(basename "$final_checkpoint")"
            fi
        fi
    done
    echo "Done! Kept only final checkpoints per run"
fi
```

### 5. One-liner to see what's taking space
```bash
cd ~/edge-of-stochastic-stability-1 && echo "=== Top directories ===" && du -sh */ 2>/dev/null | sort -hr | head -5 && echo "" && echo "=== Wandb checkpoints ===" && du -sh wandb_checkpoints/*/ 2>/dev/null | sort -hr && echo "" && echo "=== Quota ===" && quota -s 2>/dev/null || echo "quota command not available"
```

## After Cleanup

Check your quota again:
```bash
quota -s
```

If you still need more space, check:
- `$RESULTS` directory: `du -sh $RESULTS`
- `$DATASETS` directory: `du -sh $DATASETS`
- Check for large files: `find ~ -type f -size +1G 2>/dev/null | head -10`

