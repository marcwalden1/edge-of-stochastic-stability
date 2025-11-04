#!/bin/bash
# Quick disk usage checker for cluster

echo "=== Disk Usage Report ==="
echo ""

# Check quota if available
echo "Current quota:"
quota -s 2>/dev/null || echo "  (quota command not available)"
echo ""

# Check home directory
if [ -n "$HOME" ]; then
    echo "Home directory size:"
    du -sh "$HOME" 2>/dev/null | head -1 || echo "  (could not determine size)"
    echo ""
fi

# Check project directory
PROJECT_DIR="$HOME/edge-of-stochastic-stability-1"
if [ -d "$PROJECT_DIR" ]; then
    echo "=== Project Directory Breakdown ==="
    
    echo ""
    echo "Top-level directories:"
    du -sh "$PROJECT_DIR"/* 2>/dev/null | sort -hr | head -10
    
    echo ""
    echo "wandb_checkpoints breakdown:"
    if [ -d "$PROJECT_DIR/wandb_checkpoints" ]; then
        du -sh "$PROJECT_DIR/wandb_checkpoints"/* 2>/dev/null | sort -hr
        total=$(du -sh "$PROJECT_DIR/wandb_checkpoints" 2>/dev/null | cut -f1)
        echo "Total wandb_checkpoints: $total"
    else
        echo "  (wandb_checkpoints directory not found)"
    fi
    
    echo ""
    echo "slurm logs:"
    if [ -d "$PROJECT_DIR/slurm" ]; then
        log_count=$(find "$PROJECT_DIR/slurm" -type f 2>/dev/null | wc -l)
        total_size=$(du -sh "$PROJECT_DIR/slurm" 2>/dev/null | cut -f1)
        echo "  Files: $log_count"
        echo "  Size: $total_size"
    else
        echo "  (slurm directory not found)"
    fi
    
    echo ""
    echo "Python cache:"
    cache_size=$(find "$PROJECT_DIR" -type d -name __pycache__ -exec du -sh {} + 2>/dev/null | awk '{s+=$1} END {print s}' || echo "0")
    echo "  Estimated size: ~$cache_size"
    
    echo ""
    echo "Results directory:"
    if [ -n "$RESULTS" ] && [ -d "$RESULTS" ]; then
        du -sh "$RESULTS" 2>/dev/null | head -1
        echo "  Location: $RESULTS"
    else
        echo "  (RESULTS env var not set or directory not found)"
    fi
    
    echo ""
    echo "Datasets directory:"
    if [ -n "$DATASETS" ] && [ -d "$DATASETS" ]; then
        du -sh "$DATASETS" 2>/dev/null | head -1
        echo "  Location: $DATASETS"
    else
        echo "  (DATASETS env var not set or directory not found)"
    fi
fi

echo ""
echo "=== Done ==="

