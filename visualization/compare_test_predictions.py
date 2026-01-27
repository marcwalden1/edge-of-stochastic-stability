#!/usr/bin/env python3
"""
Compare test set predictions between two training runs.

Loads prediction snapshots from test_predictions.npz files and computes
Frobenius norm distances between predictions at aligned steps.

Usage:
    python visualization/compare_test_predictions.py run1_path run2_path --plot
    python visualization/compare_test_predictions.py run1_path run2_path --output distances.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def extract_lr_from_results(run_path: Path) -> float:
    """Extract learning rate from results.txt header."""
    results_file = run_path / 'results.txt'
    if not results_file.exists():
        raise FileNotFoundError(f"No results.txt found in {run_path}")

    with open(results_file, 'r') as f:
        for line in f:
            if 'Arguments:' in line or 'Namespace(' in line:
                match = re.search(r'\blr[=:\s]+([\d.eE+-]+)', line)
                if match:
                    return float(match.group(1))
                break
    raise ValueError(f"Could not extract learning rate from {results_file}")


def load_test_predictions(run_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load test predictions from .npz file.

    Args:
        run_path: Path to the run folder containing test_predictions.npz

    Returns:
        steps: array of shape (num_snapshots,) with training steps
        predictions: array of shape (num_snapshots, test_set_size, num_classes)
    """
    npz_path = run_path / 'test_predictions.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"No test_predictions.npz found in {run_path}")

    data = np.load(npz_path)
    return data['steps'], data['predictions']


def compute_frobenius_distances(
    steps1: np.ndarray,
    preds1: np.ndarray,
    steps2: np.ndarray,
    preds2: np.ndarray,
) -> pd.DataFrame:
    """Compute Frobenius norm between predictions at aligned steps.

    Frobenius norm: ||A - B||_F = sqrt(sum((a_ij - b_ij)^2))

    Args:
        steps1: Step numbers for run 1
        preds1: Predictions for run 1, shape (num_snapshots, test_size, num_classes)
        steps2: Step numbers for run 2
        preds2: Predictions for run 2, shape (num_snapshots, test_size, num_classes)

    Returns:
        DataFrame with columns: step, frobenius_distance
    """
    # Find common steps
    common_steps = sorted(set(steps1) & set(steps2))

    if not common_steps:
        raise ValueError("No common steps found between the two runs")

    # Create step-to-index mappings
    step_to_idx1 = {step: idx for idx, step in enumerate(steps1)}
    step_to_idx2 = {step: idx for idx, step in enumerate(steps2)}

    distances = []
    for step in common_steps:
        idx1 = step_to_idx1[step]
        idx2 = step_to_idx2[step]

        p1 = preds1[idx1]  # (test_size, num_classes)
        p2 = preds2[idx2]  # (test_size, num_classes)

        # Frobenius norm of the difference
        frob_dist = np.linalg.norm(p1 - p2, ord='fro')
        distances.append({'step': step, 'frobenius_distance': frob_dist})

    return pd.DataFrame(distances)


def compute_frobenius_distances_time_aligned(
    steps1: np.ndarray,
    preds1: np.ndarray,
    lr1: float,
    steps2: np.ndarray,
    preds2: np.ndarray,
    lr2: float,
) -> pd.DataFrame:
    """Compute Frobenius distances at aligned times (t = lr × step).

    Follows same matching logic as compute_trajectory_distance.py:
    For each step s1 in run1, find s2 = round(s1 * lr1/lr2) in run2.
    """
    ratio = lr1 / lr2

    # Create step-to-index mappings
    step_to_idx1 = {int(step): idx for idx, step in enumerate(steps1)}
    step_to_idx2 = {int(step): idx for idx, step in enumerate(steps2)}
    steps2_set = set(step_to_idx2.keys())

    distances = []
    for step1, idx1 in step_to_idx1.items():
        step2 = int(round(step1 * ratio))
        if step2 in steps2_set:
            idx2 = step_to_idx2[step2]
            time = step1 * lr1  # = step2 * lr2

            p1 = preds1[idx1]
            p2 = preds2[idx2]
            frob_dist = np.linalg.norm(p1 - p2, ord='fro')

            distances.append({
                'time': time,
                'step_run1': step1,
                'step_run2': step2,
                'frobenius_distance': frob_dist
            })

    if not distances:
        raise ValueError(
            f"No matching step pairs found. "
            f"ratio={ratio:.4f}, run1 has {len(steps1)} snapshots, run2 has {len(steps2)} snapshots"
        )

    return pd.DataFrame(distances).sort_values('time')


def plot_distances(
    df: pd.DataFrame,
    run1_name: str,
    run2_name: str,
    output_path: Path = None,
    time_aligned: bool = False,
) -> plt.Figure:
    """Plot Frobenius distance vs training step or continuous time.

    Args:
        df: DataFrame with step/time and frobenius_distance columns
        run1_name: Name/label for run 1 (unused, kept for API compatibility)
        run2_name: Name/label for run 2 (unused, kept for API compatibility)
        output_path: If provided, save figure to this path
        time_aligned: If True, use 'time' column for x-axis instead of 'step'

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    x_col = 'time' if time_aligned else 'step'
    x_label = 'Continuous Time (t = η × step)' if time_aligned else 'Training Step'

    ax.plot(df[x_col], df['frobenius_distance'], 'b-', linewidth=1.5, alpha=0.8,
            label='Frobenius distance')
    ax.scatter(df[x_col], df['frobenius_distance'], s=10, alpha=0.5)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Frobenius Distance', fontsize=12)
    ax.set_title('Test Prediction Distance', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare test set predictions between two training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'run1',
        type=Path,
        help='Path to first run folder (containing test_predictions.npz)',
    )
    parser.add_argument(
        'run2',
        type=Path,
        help='Path to second run folder (containing test_predictions.npz)',
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output path for CSV file with distances',
    )
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Generate and save a plot',
    )
    parser.add_argument(
        '--plot-output',
        type=Path,
        default=None,
        help='Output path for plot (default: visualization/img/test_distance.png)',
    )
    parser.add_argument(
        '--time-alignment',
        action='store_true',
        help='Align by continuous time (t = lr × step) instead of step number',
    )
    parser.add_argument(
        '--lr1',
        type=float,
        default=None,
        help='Learning rate for run1 (overrides auto-extraction from results.txt)',
    )
    parser.add_argument(
        '--lr2',
        type=float,
        default=None,
        help='Learning rate for run2 (overrides auto-extraction from results.txt)',
    )

    args = parser.parse_args()

    # Load predictions from both runs
    print(f"Loading predictions from {args.run1}...")
    steps1, preds1 = load_test_predictions(args.run1)
    print(f"  Found {len(steps1)} snapshots, shape {preds1.shape}")

    print(f"Loading predictions from {args.run2}...")
    steps2, preds2 = load_test_predictions(args.run2)
    print(f"  Found {len(steps2)} snapshots, shape {preds2.shape}")

    # Compute distances
    print("Computing Frobenius distances...")
    if args.time_alignment:
        # Extract or use provided learning rates
        lr1 = args.lr1 if args.lr1 is not None else extract_lr_from_results(args.run1)
        lr2 = args.lr2 if args.lr2 is not None else extract_lr_from_results(args.run2)
        print(f"Using time alignment: lr1={lr1}, lr2={lr2}, ratio={lr1/lr2:.4f}")

        df = compute_frobenius_distances_time_aligned(
            steps1, preds1, lr1, steps2, preds2, lr2
        )
        print(f"  Found {len(df)} matched time points")
    else:
        df = compute_frobenius_distances(steps1, preds1, steps2, preds2)
        print(f"  Computed distances for {len(df)} common steps")

    # Print summary statistics
    print(f"\nDistance statistics:")
    print(f"  Min:  {df['frobenius_distance'].min():.4f}")
    print(f"  Max:  {df['frobenius_distance'].max():.4f}")
    print(f"  Mean: {df['frobenius_distance'].mean():.4f}")
    print(f"  Std:  {df['frobenius_distance'].std():.4f}")

    # Save CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDistances saved to {args.output}")

    # Generate plot if requested
    if args.plot:
        plot_output = args.plot_output
        if plot_output is None:
            script_dir = Path(__file__).parent
            img_dir = script_dir / 'img'
            img_dir.mkdir(exist_ok=True)
            plot_output = img_dir / 'test_distance.png'

        run1_name = args.run1.name
        run2_name = args.run2.name
        plot_distances(df, run1_name, run2_name, plot_output, time_aligned=args.time_alignment)

    # If no output specified, print the data
    if not args.output and not args.plot:
        print("\nDistances by step:")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
