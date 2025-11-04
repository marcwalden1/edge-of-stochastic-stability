#!/bin/bash
# Disk cleanup script for FASRC cluster
# This script helps free up space in your home directory

set -e

echo "=== Disk Cleanup Script ==="
echo ""

# Function to safely delete files with confirmation
delete_with_size() {
    local path="$1"
    local description="$2"
    
    if [ -e "$path" ]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "Found: $description"
        echo "  Path: $path"
        echo "  Size: $size"
        read -p "  Delete? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$path"
            echo "  ✓ Deleted"
            return 0
        else
            echo "  ✗ Skipped"
            return 1
        fi
    fi
    return 1
}

# Show current disk usage
echo "Current disk usage:"
quota -s 2>/dev/null || echo "  (quota command not available)"
echo ""

# Check project directory
PROJECT_DIR="$HOME/edge-of-stochastic-stability-1"
if [ -d "$PROJECT_DIR" ]; then
    echo "=== Checking project directory: $PROJECT_DIR ==="
    
    # Old slurm logs (keep last 50 most recent files)
    if [ -d "$PROJECT_DIR/slurm" ]; then
        log_count=$(find "$PROJECT_DIR/slurm" -type f | wc -l)
        if [ "$log_count" -gt 50 ]; then
            echo "Found $log_count slurm log files"
            old_logs=$(find "$PROJECT_DIR/slurm" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n -50 | cut -d' ' -f2-)
            if [ -n "$old_logs" ]; then
                old_count=$(echo "$old_logs" | wc -l)
                total_size=$(du -sh "$PROJECT_DIR/slurm" | cut -f1)
                echo "  Would delete $old_count old log files (keeping 50 most recent)"
                echo "  Total slurm directory size: $total_size"
                read -p "  Delete old slurm logs? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "$old_logs" | xargs rm -f 2>/dev/null
                    echo "  ✓ Deleted old logs"
                fi
            fi
        fi
    fi
    
    # Python cache
    echo ""
    echo "Cleaning Python cache files..."
    find "$PROJECT_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyo" -delete 2>/dev/null || true
    echo "  ✓ Python cache cleaned"
    
    # Wandb offline runs (if already synced)
    if [ -d "$PROJECT_DIR/wandb" ]; then
        wandb_size=$(du -sh "$PROJECT_DIR/wandb" 2>/dev/null | cut -f1)
        echo ""
        echo "Wandb directory size: $wandb_size"
        echo "  (These are offline runs. Delete only if already synced to wandb.ai)"
        read -p "  Delete wandb offline runs? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$PROJECT_DIR/wandb"
            echo "  ✓ Deleted wandb offline runs"
        fi
    fi
    
    # Wandb checkpoints (if already synced)
    if [ -d "$PROJECT_DIR/wandb_checkpoints" ]; then
        delete_with_size "$PROJECT_DIR/wandb_checkpoints" "Wandb checkpoints (delete if synced)"
    fi
    
    # Old virtual environment (if broken/incompatible)
    if [ -d "$PROJECT_DIR/eoss" ]; then
        echo ""
        echo "Virtual environment found"
        venv_size=$(du -sh "$PROJECT_DIR/eoss" 2>/dev/null | cut -f1)
        echo "  Size: $venv_size"
        echo "  (Delete only if you plan to recreate it)"
        read -p "  Delete virtual environment? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$PROJECT_DIR/eoss"
            echo "  ✓ Deleted virtual environment"
        fi
    fi
fi

# Check for other common large directories
echo ""
echo "=== Checking other common large directories ==="

# Check for .cache directories
for cache_dir in "$HOME/.cache" "$HOME/.local/share"; do
    if [ -d "$cache_dir" ]; then
        size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
        echo "Found cache: $cache_dir ($size)"
        echo "  (Usually safe to clean, but may require re-downloading packages)"
        read -p "  Clean cache? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$cache_dir"/*
            echo "  ✓ Cache cleaned"
        fi
    fi
done

# Summary
echo ""
echo "=== Cleanup Complete ==="
echo "Remaining disk usage:"
quota -s 2>/dev/null || du -sh "$HOME" 2>/dev/null || true

