#!/usr/bin/env python3
"""
Plot trajectory comparison between two training runs.

Creates a 2-panel figure:
- Left panel: L2 weight distances
- Right panel: Test prediction distances

Each panel shows:
- Run1 vs Run2 (regular) - solid blue
- Run1 vs Run2 (TRUE min) - dashed blue
- Run1 from init - solid orange
- Run2 from init - solid green

Usage:
    # Option 1: Use config file (recommended)
    python distance_plots/plot_trajectory_comparison.py \
        --config output/comparison/plot_config.json

    # Option 2: Specify all arguments manually
    python distance_plots/plot_trajectory_comparison.py \
        --distance-csv output/comparison/all_distances.csv \
        --output output/comparison/trajectory_comparison.png \
        --run1-label "SGD (eta=0.005)" \
        --run2-label "SGD+M (eta=0.0025, beta=0.5)"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectory_comparison(
    df: pd.DataFrame,
    output_path: Path,
    run1_label: str = "Run 1",
    run2_label: str = "Run 2",
    lr1: float = None,
    lr2: float = None,
    extra_plots: bool = True,
):
    """
    Create 2-panel trajectory comparison plot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with distance columns from all_distances.csv
    output_path : Path
        Path to save the figure
    run1_label : str
        Label for Run 1 in legend
    run2_label : str
        Label for Run 2 in legend
    lr1 : float, optional
        Learning rate for Run 1 (for time axis scaling)
    lr2 : float, optional
        Learning rate for Run 2 (for time axis scaling)
    extra_plots : bool
        If True, plot TRUE min and init distances. If False, only plot
        distance between runs. Default True.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Determine x-axis
    if 'step' in df.columns:
        x = df['step'].values
        if lr1 is not None:
            x_label = r"Time ($\eta \cdot t$)"
            x = x * lr1
        else:
            x_label = "Training Step"
    else:
        print("Error: DataFrame must have 'step' column", file=sys.stderr)
        return

    # Color scheme
    color_between = '#1f77b4'  # blue
    color_run1 = '#ff7f0e'     # orange
    color_run2 = '#2ca02c'     # green

    # =========================================================================
    # Left panel: L2 weight distances
    # =========================================================================
    ax_l2 = axes[0]

    # L2 distance between runs (regular)
    if 'l2_distance' in df.columns:
        mask = df['l2_distance'].notna()
        ax_l2.plot(x[mask], df.loc[mask, 'l2_distance'],
                   '-', linewidth=2, color=color_between, alpha=0.8,
                   label=f'{run1_label} vs {run2_label}')

    # L2 TRUE min distance (only if extra_plots enabled)
    if extra_plots and 'l2_true_min' in df.columns:
        mask = df['l2_true_min'].notna()
        ax_l2.plot(x[mask], df.loc[mask, 'l2_true_min'],
                   '--', linewidth=2, color=color_between, alpha=0.8,
                   label=f'{run1_label} vs {run2_label} (TRUE min)')

    # L2 distance from init (Run1) (only if extra_plots enabled)
    if extra_plots and 'l2_init_run1' in df.columns:
        mask = df['l2_init_run1'].notna()
        ax_l2.plot(x[mask], df.loc[mask, 'l2_init_run1'],
                   '-', linewidth=2, color=color_run1, alpha=0.8,
                   label=f'{run1_label} from init')

    # L2 distance from init (Run2) (only if extra_plots enabled)
    if extra_plots and 'l2_init_run2' in df.columns:
        mask = df['l2_init_run2'].notna()
        ax_l2.plot(x[mask], df.loc[mask, 'l2_init_run2'],
                   '-', linewidth=2, color=color_run2, alpha=0.8,
                   label=f'{run2_label} from init')

    ax_l2.set_xlabel(x_label, fontsize=12)
    ax_l2.set_ylabel('L2 Weight Distance', fontsize=12)
    ax_l2.set_title('Weight Space Distances', fontsize=14, fontweight='bold')
    ax_l2.legend(loc='best', fontsize=10)
    ax_l2.grid(True, alpha=0.3)

    # =========================================================================
    # Right panel: Test prediction distances
    # =========================================================================
    ax_test = axes[1]

    # Test distance between runs (regular)
    if 'test_distance' in df.columns:
        mask = df['test_distance'].notna()
        ax_test.plot(x[mask], df.loc[mask, 'test_distance'],
                     '-', linewidth=2, color=color_between, alpha=0.8,
                     label=f'{run1_label} vs {run2_label}')

    # Test TRUE min distance (only if extra_plots enabled)
    if extra_plots and 'test_true_min' in df.columns:
        mask = df['test_true_min'].notna()
        ax_test.plot(x[mask], df.loc[mask, 'test_true_min'],
                     '--', linewidth=2, color=color_between, alpha=0.8,
                     label=f'{run1_label} vs {run2_label} (TRUE min)')

    # Test distance from init (Run1) (only if extra_plots enabled)
    if extra_plots and 'test_init_run1' in df.columns:
        mask = df['test_init_run1'].notna()
        ax_test.plot(x[mask], df.loc[mask, 'test_init_run1'],
                     '-', linewidth=2, color=color_run1, alpha=0.8,
                     label=f'{run1_label} from init')

    # Test distance from init (Run2) (only if extra_plots enabled)
    if extra_plots and 'test_init_run2' in df.columns:
        mask = df['test_init_run2'].notna()
        ax_test.plot(x[mask], df.loc[mask, 'test_init_run2'],
                     '-', linewidth=2, color=color_run2, alpha=0.8,
                     label=f'{run2_label} from init')

    ax_test.set_xlabel(x_label, fontsize=12)
    ax_test.set_ylabel('Test Prediction Distance (Frobenius)', fontsize=12)
    ax_test.set_title('Function Space Distances', fontsize=14, fontweight='bold')
    ax_test.legend(loc='best', fontsize=10)
    ax_test.grid(True, alpha=0.3)

    # =========================================================================
    # Finalize and save
    # =========================================================================
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot trajectory comparison between two training runs'
    )
    parser.add_argument('--config', type=str, default=None,
                       help='Path to plot_config.json (auto-fills other args)')
    parser.add_argument('--distance-csv', type=str, default=None,
                       help='Path to all_distances.csv')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the plot')
    parser.add_argument('--run1-label', type=str, default=None,
                       help='Label for Run 1 in legend')
    parser.add_argument('--run2-label', type=str, default=None,
                       help='Label for Run 2 in legend')
    parser.add_argument('--lr1', type=float, default=None,
                       help='Learning rate for Run 1 (for time axis)')
    parser.add_argument('--lr2', type=float, default=None,
                       help='Learning rate for Run 2 (for time axis)')
    parser.add_argument('--extra-plots', action='store_true', default=None,
                       help='Include TRUE min and init distance curves (default: auto based on lr1/lr2)')
    parser.add_argument('--no-extra-plots', action='store_true',
                       help='Only plot distance between runs (no TRUE min or init distances)')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            return 1
        with open(config_path) as f:
            config = json.load(f)
        # Fill in missing args from config
        if args.distance_csv is None:
            args.distance_csv = config.get('distance_csv')
        if args.run1_label is None:
            args.run1_label = config.get('run1_label', 'Run 1')
        if args.run2_label is None:
            args.run2_label = config.get('run2_label', 'Run 2')
        if args.lr1 is None:
            args.lr1 = config.get('lr1')
        if args.lr2 is None:
            args.lr2 = config.get('lr2')
        if args.output is None:
            # Default output next to the CSV
            csv_dir = Path(config.get('distance_csv', '.')).parent
            args.output = str(csv_dir / 'trajectory_comparison.png')
        print(f"Loaded config from {config_path}")

    # Validate required args
    if args.distance_csv is None:
        print("Error: --distance-csv is required (or use --config)", file=sys.stderr)
        return 1
    if args.output is None:
        print("Error: --output is required (or use --config)", file=sys.stderr)
        return 1

    # Set defaults for labels
    if args.run1_label is None:
        args.run1_label = 'Run 1'
    if args.run2_label is None:
        args.run2_label = 'Run 2'

    # Determine extra_plots setting
    # Default: False for gradient flow (different lr), True otherwise
    if args.no_extra_plots:
        extra_plots = False
    elif args.extra_plots:
        extra_plots = True
    else:
        # Auto-detect: gradient flow plots (lr1 != lr2) default to no extra plots
        if args.lr1 is not None and args.lr2 is not None and args.lr1 != args.lr2:
            extra_plots = False
            print("Gradient flow mode detected (lr1 != lr2): only plotting distance between runs")
        else:
            extra_plots = True

    # Load distance data
    csv_path = Path(args.distance_csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # For gradient flow plots (no extra plots), filter to only rows where
    # the distance between runs is not NaN
    if not extra_plots:
        # Find the last valid step for l2_distance or test_distance
        max_valid_step = 0
        if 'l2_distance' in df.columns:
            valid_l2 = df[df['l2_distance'].notna()]
            if len(valid_l2) > 0:
                max_valid_step = max(max_valid_step, valid_l2['step'].max())
        if 'test_distance' in df.columns:
            valid_test = df[df['test_distance'].notna()]
            if len(valid_test) > 0:
                max_valid_step = max(max_valid_step, valid_test['step'].max())

        if max_valid_step > 0:
            original_len = len(df)
            df = df[df['step'] <= max_valid_step].copy()
            print(f"Filtered to steps <= {max_valid_step} ({len(df)}/{original_len} rows)")

    # Create plot
    output_path = Path(args.output)
    plot_trajectory_comparison(
        df=df,
        output_path=output_path,
        run1_label=args.run1_label,
        run2_label=args.run2_label,
        lr1=args.lr1,
        lr2=args.lr2,
        extra_plots=extra_plots,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
