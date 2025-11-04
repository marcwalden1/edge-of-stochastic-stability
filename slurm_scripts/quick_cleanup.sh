#!/bin/bash
# Quick cleanup script - removes obvious space hogs with minimal prompts

set -e

echo "=== Quick Disk Cleanup ==="
echo ""
echo "WARNING: This script will delete files. Review what will be deleted first!"
echo ""

PROJECT_DIR="$HOME/edge-of-stochastic-stability-1"

# Show what we'll clean before doing it
echo "=== Items to Clean ==="
echo ""

# 1. Python cache (safe to delete)
echo "1. Python cache files (__pycache__, *.pyc)"
python_cache_size=$(find "$PROJECT_DIR" -type d -name __pycache__ -exec du -sh {} + 2>/dev/null | awk '{s+=$1} END {if(s) print s "K"; else print "0"}' || echo "0")
find "$PROJECT_DIR" -name "*.pyc" -o -name "*.pyo" 2>/dev/null | wc -l | xargs -I {} echo "  Found {} files"
echo ""

# 2. Old SLURM logs (keep last 20)
if [ -d "$PROJECT_DIR/slurm" ]; then
    log_count=$(find "$PROJECT_DIR/slurm" -type f 2>/dev/null | wc -l)
    if [ "$log_count" -gt 20 ]; then
        echo "2. Old SLURM logs (keeping 20 most recent, deleting $((log_count - 20)) old ones)"
        slurm_size=$(du -sh "$PROJECT_DIR/slurm" 2>/dev/null | cut -f1)
        echo "  Current size: $slurm_size"
    else
        echo "2. SLURM logs (only $log_count files, skipping)"
    fi
else
    echo "2. SLURM logs (no slurm directory found)"
fi
echo ""

# 3. Wandb checkpoints - show what we have
if [ -d "$PROJECT_DIR/wandb_checkpoints" ]; then
    echo "3. Wandb checkpoints:"
    du -sh "$PROJECT_DIR/wandb_checkpoints"/* 2>/dev/null | sort -hr | head -5
    total_checkpoints=$(du -sh "$PROJECT_DIR/wandb_checkpoints" 2>/dev/null | cut -f1)
    run_count=$(ls -1d "$PROJECT_DIR/wandb_checkpoints"/*/ 2>/dev/null | wc -l)
    echo "  Total size: $total_checkpoints"
    echo "  Number of runs: $run_count"
    echo "  (Option: Delete old checkpoint files, keeping only final checkpoint per run)"
    echo ""
fi

# Ask for confirmation
echo ""
read -p "Proceed with cleanup? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "=== Cleaning ==="

# Clean Python cache
echo "Cleaning Python cache..."
find "$PROJECT_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR" -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ Python cache cleaned"

# Clean old SLURM logs
if [ -d "$PROJECT_DIR/slurm" ]; then
    log_count=$(find "$PROJECT_DIR/slurm" -type f 2>/dev/null | wc -l)
    if [ "$log_count" -gt 20 ]; then
        echo "Cleaning old SLURM logs..."
        find "$PROJECT_DIR/slurm" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | tail -n +21 | cut -d' ' -f2- | xargs rm -f 2>/dev/null || true
        echo "  ✓ Old SLURM logs cleaned"
    fi
fi

# For wandb checkpoints, ask separately
if [ -d "$PROJECT_DIR/wandb_checkpoints" ]; then
    echo ""
    echo "Wandb checkpoints found. Options:"
    echo "  a) Delete entire wandb_checkpoints directory (if already synced to wandb.ai)"
    echo "  b) Keep only final checkpoint per run (saves most space, keeps last checkpoint)"
    echo "  c) Skip checkpoint cleanup"
    read -p "Choose (a/b/c): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Aa]$ ]]; then
        read -p "  Are you SURE you've synced these to wandb.ai? (type 'yes' to confirm): " confirm
        if [ "$confirm" == "yes" ]; then
            rm -rf "$PROJECT_DIR/wandb_checkpoints"
            echo "  ✓ Deleted wandb_checkpoints directory"
        else
            echo "  ✗ Skipped (type 'yes' to confirm deletion)"
        fi
    elif [[ $REPLY =~ ^[Bb]$ ]]; then
        echo "  Keeping only final checkpoint per run..."
        for run_dir in "$PROJECT_DIR/wandb_checkpoints"/*/; do
            if [ -d "$run_dir" ]; then
                # Find the checkpoint with highest step number
                final_checkpoint=$(ls -1 "$run_dir"/checkpoint_step_*.pt 2>/dev/null | sort -V | tail -1)
                if [ -n "$final_checkpoint" ]; then
                    # Delete all checkpoints except the final one and metadata
                    find "$run_dir" -name "checkpoint_step_*.pt" ! -name "$(basename "$final_checkpoint")" -delete 2>/dev/null || true
                    echo "    ✓ Cleaned $(basename "$(dirname "$run_dir")")"
                fi
            fi
        done
        echo "  ✓ Kept only final checkpoints"
    else
        echo "  ✗ Skipped checkpoint cleanup"
    fi
fi

echo ""
echo "=== Cleanup Complete ==="
echo "Remaining disk usage:"
quota -s 2>/dev/null || du -sh "$HOME" 2>/dev/null || true

