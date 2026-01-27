#!/usr/bin/env python3
"""
Plot matching stabilization levels between SGDM and vanilla SGD.

Compares two training runs with equal effective learning rate (eta/(1-beta)):
- SGDM run: (eta, beta, b) with momentum
- SGD run: (eta', 0, b) with larger step size, no momentum

Shows that when batch sharpness stabilizes at the same level, lambda_max
also stabilizes at approximately the same level.

Automatically finds the two most recent runs in the results directory.

Usage:
    python visualization/plot_matching_stabilization.py --results-dir $RESULTS/plaintext
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


COLUMN_NAMES = [
    "epoch",
    "step",
    "batch_loss",
    "full_loss",
    "lambda_max",
    "step_sharpness",
    "batch_sharpness",
    "gni",
    "total_accuracy",
]

# Two distinct greens for batch sharpness
GREEN_SGDM = '#2ca02c'  # Forest green
GREEN_SGD = '#90EE90'   # Light green

# Two distinct blues for lambda_max
BLUE_SGDM = '#1f77b4'   # Standard blue
BLUE_SGD = '#87CEEB'    # Sky blue

# Two distinct grays for full loss
GRAY_SGDM = '#505050'   # Dark gray
GRAY_SGD = '#A0A0A0'    # Light gray


def require_env_path(name: str) -> Path:
    """Get path from environment variable or raise error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} environment variable before running this script.")
    return Path(value)


def find_two_most_recent_runs(results_dir: Path) -> tuple[Path, Path]:
    """Find the two most recent run folders in the results directory."""
    run_folders = []

    for dataset_folder in results_dir.iterdir():
        if not dataset_folder.is_dir():
            continue
        for run_folder in dataset_folder.iterdir():
            if not run_folder.is_dir():
                continue
            results_file = run_folder / 'results.txt'
            if results_file.exists():
                run_folders.append((run_folder, run_folder.stat().st_mtime))

    if len(run_folders) < 2:
        raise RuntimeError(f"Need at least 2 runs in {results_dir}, found {len(run_folders)}")

    # Sort by modification time (most recent first)
    run_folders.sort(key=lambda x: x[1], reverse=True)

    # Return the two most recent
    return run_folders[0][0], run_folders[1][0]


def extract_hyperparams(folder: Path) -> dict:
    """Extract hyperparameters from results.txt header."""
    file_path = folder / 'results.txt'
    params = {
        'lr': None,
        'momentum': 0.0,
        'batch_size': None,
    }

    if not file_path.exists():
        return params

    with open(file_path, 'r') as f:
        for line in f:
            if 'Arguments:' in line or 'Namespace(' in line:
                match = re.search(r'\blr[=:\s]+([\d.]+)', line)
                if match:
                    params['lr'] = float(match.group(1))

                match = re.search(r'[^_]momentum[=:\s]+([\d.]+)', line)
                if match:
                    params['momentum'] = float(match.group(1))

                match = re.search(r'\bbatch[=:\s]+(\d+)', line)
                if match:
                    params['batch_size'] = int(match.group(1))
                break

    return params


def load_results(folder: Path) -> pd.DataFrame:
    """Load results.txt from a run folder."""
    file_path = folder / 'results.txt'
    if not file_path.exists():
        raise RuntimeError(f"Missing results.txt in {folder}")

    df = pd.read_csv(
        file_path,
        skiprows=4,
        sep=',',
        header=None,
        names=COLUMN_NAMES,
        na_values=['nan'],
        skipinitialspace=True,
    )
    return df


def plot_matching_stabilization(
    sgdm_folder: Path,
    sgd_folder: Path,
    output_path: Path,
) -> None:
    """
    Plot batch_sharpness and lambda_max for both runs.
    """
    # Load data
    df_sgdm = load_results(sgdm_folder)
    df_sgd = load_results(sgd_folder)

    # Extract hyperparameters for labels
    params_sgdm = extract_hyperparams(sgdm_folder)
    params_sgd = extract_hyperparams(sgd_folder)

    eta_sgdm = params_sgdm['lr']
    beta_sgdm = params_sgdm['momentum']
    eta_sgd = params_sgd['lr']
    beta_sgd = params_sgd['momentum']

    # Calculate effective learning rates
    eff_lr_sgdm = eta_sgdm / (1 - beta_sgdm) if beta_sgdm < 1 else float('inf')
    eff_lr_sgd = eta_sgd / (1 - beta_sgd) if beta_sgd < 1 else eta_sgd

    print(f"SGDM: eta={eta_sgdm}, beta={beta_sgdm}, eff_lr={eff_lr_sgdm:.4f}")
    print(f"SGD:  eta={eta_sgd}, beta={beta_sgd}, eff_lr={eff_lr_sgd:.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create second y-axis for loss
    ax2 = ax.twinx()

    # Plot batch sharpness (greens)
    bs_sgdm = df_sgdm[['step', 'batch_sharpness']].dropna()
    bs_sgd = df_sgd[['step', 'batch_sharpness']].dropna()

    if not bs_sgdm.empty:
        ax.plot(
            bs_sgdm['step'],
            bs_sgdm['batch_sharpness'],
            color=GREEN_SGDM,
            linewidth=1.5,
            label='Batch Sharpness (SGDM)',
        )

    if not bs_sgd.empty:
        ax.plot(
            bs_sgd['step'],
            bs_sgd['batch_sharpness'],
            color=GREEN_SGD,
            linewidth=1.5,
            label='Batch Sharpness (SGD)',
        )

    # Plot lambda_max (blues, continuous)
    lm_sgdm = df_sgdm[['step', 'lambda_max']].dropna()
    lm_sgd = df_sgd[['step', 'lambda_max']].dropna()

    if not lm_sgdm.empty:
        ax.plot(
            lm_sgdm['step'],
            lm_sgdm['lambda_max'],
            color=BLUE_SGDM,
            linewidth=1.5,
            label=r'$\lambda_{\max}$ (SGDM)',
        )

    if not lm_sgd.empty:
        ax.plot(
            lm_sgd['step'],
            lm_sgd['lambda_max'],
            color=BLUE_SGD,
            linewidth=1.5,
            label=r'$\lambda_{\max}$ (SGD)',
        )

    # Plot full loss (grays) on secondary axis
    fl_sgdm = df_sgdm[['step', 'full_loss']].dropna()
    fl_sgd = df_sgd[['step', 'full_loss']].dropna()

    if not fl_sgdm.empty:
        ax2.plot(
            fl_sgdm['step'],
            fl_sgdm['full_loss'],
            color=GRAY_SGDM,
            linewidth=1.5,
            linestyle='--',
            label='Full Loss (SGDM)',
        )

    if not fl_sgd.empty:
        ax2.plot(
            fl_sgd['step'],
            fl_sgd['full_loss'],
            color=GRAY_SGD,
            linewidth=1.5,
            linestyle='--',
            label='Full Loss (SGD)',
        )

    # Add theoretical stabilization line: 2/eta_eff
    theoretical_level = 2 / eff_lr_sgdm
    ax.axhline(
        y=theoretical_level,
        color='black',
        linestyle='--',
        linewidth=2,
        label=r'$2/\eta_{\mathrm{SGD}}$',
    )

    # Formatting
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Sharpness', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_yscale('log')
    ax.set_title('SGD vs SGDM at Equal Effective Learning Rate', fontsize=18)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=13, framealpha=0.9)

    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot matching stabilization levels between SGDM and SGD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        help='Path to results directory (default: $RESULTS/plaintext)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('visualization/img/matching_stabilization.png'),
        help='Output path for the plot (default: visualization/img/matching_stabilization.png)',
    )

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = require_env_path('RESULTS') / 'plaintext'

    if not results_dir.exists():
        raise RuntimeError(f"Results directory does not exist: {results_dir}")

    # Find the two most recent runs
    run1, run2 = find_two_most_recent_runs(results_dir)
    print(f"Found runs:")
    print(f"  Run 1: {run1.name}")
    print(f"  Run 2: {run2.name}")

    # Determine which is SGDM (has momentum) and which is SGD (no momentum)
    params1 = extract_hyperparams(run1)
    params2 = extract_hyperparams(run2)

    if params1['momentum'] > params2['momentum']:
        sgdm_folder, sgd_folder = run1, run2
    else:
        sgdm_folder, sgd_folder = run2, run1

    print(f"  SGDM: {sgdm_folder.name}")
    print(f"  SGD:  {sgd_folder.name}")

    plot_matching_stabilization(
        sgdm_folder=sgdm_folder,
        sgd_folder=sgd_folder,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
