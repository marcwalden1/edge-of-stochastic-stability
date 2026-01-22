#!/usr/bin/env python3
"""
Plot training intervention experiments.

Finds result folders by experiment tag pattern and plots batch_sharpness vs step
for runs A, B, C on the same axes, with intervention step marked.

Usage:
    python visualization/plot_interventions.py --type lr --results-dir $RESULTS/plaintext
    python visualization/plot_interventions.py --type momentum --results-dir $RESULTS/plaintext
    python visualization/plot_interventions.py --type batch --results-dir $RESULTS/plaintext
    python visualization/plot_interventions.py --all --results-dir $RESULTS/plaintext
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

VARIANT_COLORS = {
    'A': '#1f77b4',  # Blue - baseline
    'B': '#ff7f0e',  # Orange - changed from start
    'C': '#2ca02c',  # Green - intervention
}

VARIANT_LABELS = {
    'A': 'Run A (baseline)',
    'B': 'Run B (changed)',
    'C': 'Run C (intervention)',
}


@dataclass
class RunInfo:
    folder: Path
    experiment_tag: str
    variant: str
    lr: float
    batch_size: int


def require_env_path(name: str) -> Path:
    """Get path from environment variable or raise error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} environment variable before running this script.")
    return Path(value)


def find_intervention_runs(results_dir: Path, experiment_type: str,
                           experiment_name: str = "intervention") -> dict[str, RunInfo]:
    """
    Find A, B, C runs for a given experiment type.

    Returns dict mapping variant ('A', 'B', 'C') to RunInfo.
    """
    runs = {}
    pattern = re.compile(
        rf'^{experiment_name}_{experiment_type}_([ABC])_\d+_\d+_\d+_lr([\d.]+)_b(\d+)$'
    )

    for dataset_folder in results_dir.iterdir():
        if not dataset_folder.is_dir():
            continue
        for run_folder in dataset_folder.iterdir():
            if not run_folder.is_dir():
                continue

            match = pattern.match(run_folder.name)
            if match:
                variant = match.group(1)
                lr = float(match.group(2))
                batch_size = int(match.group(3))

                # Keep the most recent run for each variant
                if variant not in runs or run_folder.stat().st_mtime > runs[variant].folder.stat().st_mtime:
                    runs[variant] = RunInfo(
                        folder=run_folder,
                        experiment_tag=f"{experiment_name}_{experiment_type}_{variant}",
                        variant=variant,
                        lr=lr,
                        batch_size=batch_size,
                    )

    return runs


def load_results(run: RunInfo) -> pd.DataFrame:
    """Load results.txt from a run folder."""
    file_path = run.folder / 'results.txt'
    if not file_path.exists():
        raise RuntimeError(f"Missing results.txt in {run.folder}")

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


def extract_intervention_step(run_c: RunInfo) -> Optional[int]:
    """Extract intervention step from Run C's results.txt header."""
    file_path = run_c.folder / 'results.txt'
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        for line in f:
            if 'intervention_step' in line.lower() or 'intervention-step' in line:
                # Try to extract the step number from the arguments
                match = re.search(r'intervention.step[=:\s]+(\d+)', line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
    return None


def plot_intervention_comparison(
    runs: dict[str, RunInfo],
    experiment_type: str,
    intervention_step: Optional[int] = None,
) -> plt.Figure:
    """
    Plot batch_sharpness vs step for runs A, B, C on the same axes.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine LR for 2/η line (use baseline run A)
    baseline_lr = runs.get('A', list(runs.values())[0]).lr if runs else 0.004

    # Add 2/η reference line
    ax.axhline(y=2 / baseline_lr, color='black', linestyle='--',
               label=r'2/$\eta$ = ' + f'{2/baseline_lr:.1f}', alpha=0.7)

    # Plot each variant
    for variant in ['A', 'B', 'C']:
        if variant not in runs:
            continue

        run = runs[variant]
        try:
            df = load_results(run)
        except RuntimeError as e:
            print(f"Warning: {e}")
            continue

        batch_sharp = df[['step', 'batch_sharpness']].dropna()
        if not batch_sharp.empty:
            ax.plot(
                batch_sharp['step'],
                batch_sharp['batch_sharpness'],
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
                linewidth=1.5,
                alpha=0.8,
            )

    # Mark intervention step with vertical line
    if intervention_step is not None:
        ax.axvline(x=intervention_step, color='red', linestyle=':',
                   label=f'Intervention step ({intervention_step})', linewidth=2)

    # Formatting
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Batch Sharpness', fontsize=12)

    type_labels = {
        'lr': 'Learning Rate',
        'momentum': 'Momentum',
        'batch': 'Batch Size',
    }
    ax.set_title(f'{type_labels.get(experiment_type, experiment_type)} Intervention Experiment',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis to start from 0 with some headroom
    ax.set_ylim(bottom=0)

    return fig


def save_figure(fig: plt.Figure, experiment_type: str, model: str = "") -> Path:
    """Save figure to visualization/img/."""
    script_dir = Path(__file__).parent
    img_dir = script_dir / 'img'
    img_dir.mkdir(exist_ok=True)

    suffix = f"_{model}" if model else ""
    filename = f"intervention_{experiment_type}{suffix}.png"
    output_path = img_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training intervention experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--type',
        choices=['lr', 'momentum', 'batch'],
        help='Type of intervention experiment to plot',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Plot all intervention types',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        help='Path to results directory (default: $RESULTS/plaintext)',
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='intervention',
        help='Experiment name prefix (default: intervention)',
    )
    parser.add_argument(
        '--intervention-step',
        type=int,
        default=None,
        help='Intervention step to mark on plot (auto-detected if not specified)',
    )

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = require_env_path('RESULTS') / 'plaintext'

    if not results_dir.exists():
        raise RuntimeError(f"Results directory does not exist: {results_dir}")

    # Determine which types to plot
    if args.all:
        experiment_types = ['lr', 'momentum', 'batch']
    elif args.type:
        experiment_types = [args.type]
    else:
        parser.error("Must specify --type or --all")

    # Plot each experiment type
    for exp_type in experiment_types:
        print(f"\nProcessing {exp_type} intervention experiment...")

        runs = find_intervention_runs(
            results_dir,
            exp_type,
            experiment_name=args.experiment_name,
        )

        if not runs:
            print(f"  No runs found for {exp_type} experiment")
            continue

        print(f"  Found runs: {', '.join(sorted(runs.keys()))}")
        for variant, run in sorted(runs.items()):
            print(f"    {variant}: {run.folder.name}")

        # Determine intervention step
        intervention_step = args.intervention_step
        if intervention_step is None and 'C' in runs:
            intervention_step = extract_intervention_step(runs['C'])

        # Extract model from folder path (e.g., cifar10_mlp)
        model = ""
        if runs:
            first_run = list(runs.values())[0]
            parent_name = first_run.folder.parent.name
            if '_' in parent_name:
                model = parent_name.split('_')[1]

        # Generate plot
        fig = plot_intervention_comparison(runs, exp_type, intervention_step)
        output_path = save_figure(fig, exp_type, model)
        print(f"  Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
