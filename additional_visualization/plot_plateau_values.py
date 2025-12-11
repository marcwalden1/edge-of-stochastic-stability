#!/usr/bin/env python3
"""
Plot plateau values vs batch size from plateau_values.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_batch_sharpness_plateau():
    """Plot batch_sharpness_plateau vs batch_size"""
    
    # Read the CSV file
    csv_path = Path(__file__).parent / 'plateau_values.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run calculate_plateau_values.py first.")
        return 1
    
    df = pd.read_csv(csv_path)
    
    # Group by batch_size and take the mean (in case there are multiple runs per batch size)
    grouped = df.groupby('batch_size').agg({
        'batch_sharpness_plateau': ['mean', 'std', 'count'],
        'batch_sharpness_moving_avg_final': 'mean',
    }).reset_index()
    
    grouped.columns = ['batch_size', 'batch_sharpness_plateau_mean', 'batch_sharpness_plateau_std', 'count', 'batch_sharpness_ma_mean']
    
    # Sort by batch size
    grouped = grouped.sort_values('batch_size')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot plateau values - just markers, no lines or error bars
    x = grouped['batch_size']
    y = grouped['batch_sharpness_plateau_mean']
    
    ax.plot(x, y, 'o', markersize=6, label='Batch Sharpness Plateau', 
            color='#2ca02c', alpha=0.7)
    
    # Add 2/eta line if learning rate is constant
    if 'learning_rate' in df.columns:
        lr = df['learning_rate'].dropna().iloc[0] if len(df['learning_rate'].dropna()) > 0 else None
        if lr:
            threshold = 2.0 / lr
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'2/η = {threshold:.2f} (η={lr})', alpha=0.7)
    
    # Normal scale for both axes
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Batch Sharpness Plateau Value', fontsize=14)
    ax.set_title('Batch Sharpness Plateau vs Batch Size', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path(__file__).parent / 'visualization' / 'img'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'batch_sharpness_plateau_vs_batch_size.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total batch sizes: {len(grouped)}")
    print(f"  Batch size range: {grouped['batch_size'].min()} - {grouped['batch_size'].max()}")
    print(f"  Batch sharpness plateau range: {y.min():.4f} - {y.max():.4f}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(plot_batch_sharpness_plateau())

