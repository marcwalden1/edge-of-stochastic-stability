#!/bin/bash
# Quick script to check disk usage on cluster

echo "=== Disk Usage Check ==="
echo ""

# Check quota
echo "1. Your quota:"
quota -s 2>/dev/null || echo "  (quota command not available)"
echo ""

# Check project directory (what you already saw)
echo "2. Project directory breakdown:"
cd ~/edge-of-stochastic-stability-1
du -sh */ 2>/dev/null | sort -hr | head -10
echo ""

# Check RESULTS directory (where checkpoints actually are!)
echo "3. RESULTS directory ($RESULTS or ~/results):"
if [ -n "$RESULTS" ] && [ -d "$RESULTS" ]; then
    echo "  Location: $RESULTS"
    du -sh "$RESULTS" 2>/dev/null
    echo ""
    echo "  Breakdown:"
    du -sh "$RESULTS"/*/ 2>/dev/null | sort -hr | head -10
    echo ""
    if [ -d "$RESULTS/wandb_checkpoints" ]; then
        echo "  Wandb checkpoints:"
        du -sh "$RESULTS/wandb_checkpoints"/*/ 2>/dev/null | sort -hr | head -10
        total=$(du -sh "$RESULTS/wandb_checkpoints" 2>/dev/null | cut -f1)
        echo "  Total wandb_checkpoints: $total"
    fi
elif [ -d ~/results ]; then
    echo "  Location: ~/results"
    du -sh ~/results 2>/dev/null
    echo ""
    echo "  Breakdown:"
    du -sh ~/results/*/ 2>/dev/null | sort -hr | head -10
    if [ -d ~/results/wandb_checkpoints ]; then
        echo ""
        echo "  Wandb checkpoints:"
        du -sh ~/results/wandb_checkpoints/*/ 2>/dev/null | sort -hr | head -10
        total=$(du -sh ~/results/wandb_checkpoints 2>/dev/null | cut -f1)
        echo "  Total wandb_checkpoints: $total"
    fi
else
    echo "  RESULTS directory not found"
fi
echo ""

# Check DATASETS
echo "4. DATASETS directory:"
if [ -n "$DATASETS" ] && [ -d "$DATASETS" ]; then
    echo "  Location: $DATASETS"
    du -sh "$DATASETS" 2>/dev/null
elif [ -d ~/datasets ]; then
    echo "  Location: ~/datasets"
    du -sh ~/datasets 2>/dev/null
else
    echo "  DATASETS directory not found"
fi
echo ""

# Check eoss venv
echo "5. Virtual environment (eoss/):"
if [ -d ~/edge-of-stochastic-stability-1/eoss ]; then
    venv_size=$(du -sh ~/edge-of-stochastic-stability-1/eoss 2>/dev/null | cut -f1)
    echo "  Size: $venv_size (can be recreated if needed)"
fi
echo ""

echo "=== Done ==="

