#!/usr/bin/env python3
"""
Plot weight trajectory distances and lambda_max over time.
Visualizes the relationship between weight distance and progressive sharpening.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_distance_data(csv_path: Path) -> pd.DataFrame:
    """Load distance data from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def load_lambda_max_data(run_id: str, results_root: Path) -> pd.DataFrame:
    """
    Load lambda_max values from results.txt file.
    
    Returns DataFrame with columns: step, lambda_max
    """
    results_root = Path(results_root).expanduser()
    
    # Search for results.txt files - this is simplified
    # In practice, you might need better matching logic
    for folder in sorted(results_root.glob('*/')):
        results_file = folder / 'results.txt'
        if not results_file.exists():
            continue
        
        try:
            df = pd.read_csv(
                results_file,
                skiprows=4,
                sep=',',
                header=None,
                names=['epoch', 'step', 'batch_loss', 'full_loss', 'lambda_max',
                       'step_sharpness', 'batch_sharpness', 'gni', 'total_accuracy'],
                na_values=['nan'],
                skipinitialspace=True
            )
            
            if len(df) > 0:
                return df[['step', 'lambda_max']].copy()
                
        except Exception as e:
            continue
    
    raise FileNotFoundError(f"Could not find results.txt for run {run_id}")


def plot_trajectory_distance(
    distance_df: pd.DataFrame,
    run_id1: str,
    run_id2: str,
    lambda_max_df1: pd.DataFrame = None,
    lambda_max_df2: pd.DataFrame = None,
    output_path: Path = None,
    lr1: float = None,
    lr2: float = None
):
    """
    Create plots showing weight distance and lambda_max over time.
    
    Parameters:
    -----------
    distance_df : pd.DataFrame
        DataFrame with columns: step (or step_run1), weight_distance (and optionally lambda_max columns)
    run_id1 : str
        First run ID (for labeling)
    run_id2 : str
        Second run ID (for labeling)
    lambda_max_df1 : pd.DataFrame, optional
        Lambda_max data for run 1 (columns: step, lambda_max)
    lambda_max_df2 : pd.DataFrame, optional
        Lambda_max data for run 2 (columns: step, lambda_max)
    output_path : Path, optional
        Path to save the plot
    lr1 : float, optional
        Learning rate for run 1 (for computing time = step * lr1)
    lr2 : float, optional
        Learning rate for run 2 (for computing time = step * lr2)
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Determine which step column to use and compute time axis
    if 'step_run1' in distance_df.columns:
        # Using step*eta matching - use step_run1 for time axis
        step_col = 'step_run1'
        if lr1 is not None:
            time_axis = distance_df[step_col] * lr1
            time_label = "Time (step * η)"
        else:
            time_axis = distance_df[step_col]
            time_label = "Training Step (Run 1)"
    elif 'step' in distance_df.columns:
        step_col = 'step'
        if lr1 is not None:
            time_axis = distance_df[step_col] * lr1
            time_label = "Time (step * η)"
        elif lr2 is not None:
            time_axis = distance_df[step_col] * lr2
            time_label = "Time (step * η)"
        else:
            time_axis = distance_df[step_col]
            time_label = "Training Step"
    else:
        raise ValueError("CSV must have either 'step' or 'step_run1' column")
    
    # Plot: Weight distance
    ax1.plot(time_axis, distance_df['weight_distance'], 
             'o-', linewidth=2, markersize=4, color='#2ca02c', alpha=0.7,
             label='Weight Distance (L2)')
    ax1.set_ylabel('Weight Distance (L2)', fontsize=14)
    ax1.set_xlabel(time_label, fontsize=14)
    ax1.set_title(f'Weight Trajectory Distance: {run_id1} vs {run_id2}', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot trajectory distances and lambda_max')
    parser.add_argument('distance_csv', type=str, help='CSV file with distance data')
    parser.add_argument('run_id1', type=str, help='First run ID')
    parser.add_argument('run_id2', type=str, help='Second run ID')
    parser.add_argument('--lambda-max-csv1', type=str, default=None, 
                       help='CSV with lambda_max for run 1 (optional if in distance CSV)')
    parser.add_argument('--lambda-max-csv2', type=str, default=None,
                       help='CSV with lambda_max for run 2 (optional if in distance CSV)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory to search for results.txt files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output plot path (default: visualization/img/trajectory_distance_{run_id1}_{run_id2}.png)')
    parser.add_argument('--lr1', type=float, default=None,
                       help='Learning rate for run 1 (for computing time = step * lr1)')
    parser.add_argument('--lr2', type=float, default=None,
                       help='Learning rate for run 2 (for computing time = step * lr2)')
    
    args = parser.parse_args()
    
    # Load distance data
    distance_csv = Path(args.distance_csv)
    if not distance_csv.exists():
        print(f"Error: Distance CSV not found: {distance_csv}", file=sys.stderr)
        return 1
    
    distance_df = load_distance_data(distance_csv)
    print(f"Loaded {len(distance_df)} distance measurements")
    
    # Load lambda_max data if needed
    lambda_max_df1 = None
    lambda_max_df2 = None
    
    if args.lambda_max_csv1:
        lambda_max_df1 = pd.read_csv(args.lambda_max_csv1)
    elif args.results_dir:
        try:
            lambda_max_df1 = load_lambda_max_data(args.run_id1, Path(args.results_dir))
        except FileNotFoundError:
            print(f"Warning: Could not load lambda_max for run 1", file=sys.stderr)
    
    if args.lambda_max_csv2:
        lambda_max_df2 = pd.read_csv(args.lambda_max_csv2)
    elif args.results_dir:
        try:
            lambda_max_df2 = load_lambda_max_data(args.run_id2, Path(args.results_dir))
        except FileNotFoundError:
            print(f"Warning: Could not load lambda_max for run 2", file=sys.stderr)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'visualization' / 'img'
        output_path = output_dir / f"trajectory_distance_{args.run_id1}_{args.run_id2}.png"
    
    # Create plot
    plot_trajectory_distance(
        distance_df=distance_df,
        run_id1=args.run_id1,
        run_id2=args.run_id2,
        lambda_max_df1=lambda_max_df1,
        lambda_max_df2=lambda_max_df2,
        output_path=output_path,
        lr1=args.lr1,
        lr2=args.lr2
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

