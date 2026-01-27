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
    'B': '#9467bd',  # Purple - changed from start
    'C': '#d62728',  # Red - intervention
}


@dataclass
class RunInfo:
    folder: Path
    experiment_tag: str
    variant: str
    lr: float
    batch_size: int
    momentum: float = 0.0
    # Intervention values (only relevant for Run C)
    intervention_lr: Optional[float] = None
    intervention_momentum: Optional[float] = None
    intervention_batch: Optional[int] = None
    intervention_step: Optional[int] = None


def require_env_path(name: str) -> Path:
    """Get path from environment variable or raise error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} environment variable before running this script.")
    return Path(value)


def extract_hyperparams_from_results(folder: Path) -> dict:
    """Extract hyperparameters from results.txt header (Arguments line).

    Returns dict with keys: lr, momentum, batch_size, intervention_lr,
    intervention_momentum, intervention_batch, intervention_step
    """
    file_path = folder / 'results.txt'
    params = {
        'lr': None,
        'momentum': 0.0,
        'batch_size': None,
        'intervention_lr': None,
        'intervention_momentum': None,
        'intervention_batch': None,
        'intervention_step': None,
    }

    if not file_path.exists():
        return params

    with open(file_path, 'r') as f:
        for line in f:
            if 'Arguments:' in line or 'Namespace(' in line:
                # Extract lr
                match = re.search(r'\blr[=:\s]+([\d.]+)', line)
                if match:
                    params['lr'] = float(match.group(1))

                # Extract momentum (not intervention_momentum)
                match = re.search(r'[^_]momentum[=:\s]+([\d.]+)', line)
                if match:
                    params['momentum'] = float(match.group(1))

                # Extract batch size
                match = re.search(r'\bbatch[=:\s]+(\d+)', line)
                if match:
                    params['batch_size'] = int(match.group(1))

                # Extract intervention values
                match = re.search(r'intervention_lr[=:\s]+([\d.]+)', line)
                if match:
                    params['intervention_lr'] = float(match.group(1))

                match = re.search(r'intervention_momentum[=:\s]+([\d.]+)', line)
                if match:
                    params['intervention_momentum'] = float(match.group(1))

                match = re.search(r'intervention_batch[=:\s]+(\d+)', line)
                if match:
                    params['intervention_batch'] = int(match.group(1))

                match = re.search(r'intervention_step[=:\s]+(\d+)', line)
                if match:
                    params['intervention_step'] = int(match.group(1))

                break

    return params


def extract_momentum_from_results(folder: Path) -> float:
    """Extract momentum value from results.txt header (Arguments line)."""
    params = extract_hyperparams_from_results(folder)
    return params['momentum']


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
                    params = extract_hyperparams_from_results(run_folder)
                    runs[variant] = RunInfo(
                        folder=run_folder,
                        experiment_tag=f"{experiment_name}_{experiment_type}_{variant}",
                        variant=variant,
                        lr=params['lr'] if params['lr'] is not None else lr,
                        batch_size=params['batch_size'] if params['batch_size'] is not None else batch_size,
                        momentum=params['momentum'],
                        intervention_lr=params['intervention_lr'],
                        intervention_momentum=params['intervention_momentum'],
                        intervention_batch=params['intervention_batch'],
                        intervention_step=params['intervention_step'],
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
    show_mini_loss: bool = False,
    vert_color: str = 'red',
) -> plt.Figure:
    """
    Plot batch_sharpness vs step for runs A, B, C on the same axes.

    Top subplot: batch_sharpness and lambda_max
    Bottom subplot: loss curves (full_loss always, batch_loss if show_mini_loss=True)
    """
    fig, (ax, ax_loss) = plt.subplots(
        2, 1, figsize=(12, 8), height_ratios=[2, 1], sharex=True
    )

    # Get Run A (baseline) parameters
    run_a = runs.get('A')
    if run_a:
        eta_a = run_a.lr
        beta_a = run_a.momentum
    else:
        # Fallback to first available run
        first_run = list(runs.values())[0] if runs else None
        eta_a = first_run.lr if first_run else 0.004
        beta_a = first_run.momentum if first_run else 0.0

    # Theoretical stabilization value for Run A: 2/η_A * (1 - β_A)
    theoretical_a = (2 / eta_a) * (1 - beta_a)
    ax.axhline(y=theoretical_a, color='#1f77b4', linestyle='--', linewidth=2.4, alpha=0.7)

    # Check if Run B has different LR or momentum from Run A
    run_b = runs.get('B')
    if run_b:
        eta_b = run_b.lr
        beta_b = run_b.momentum
        # Add Run B theoretical line if LR or momentum differs
        if abs(eta_b - eta_a) > 1e-8 or abs(beta_b - beta_a) > 1e-8:
            theoretical_b = (2 / eta_b) * (1 - beta_b)
            ax.axhline(y=theoretical_b, color='#9467bd', linestyle='--', linewidth=2.4, alpha=0.7)

    # Add single legend entry for theoretical lines (black)
    ax.axhline(y=-1000, color='black', linestyle='--', linewidth=2.4, alpha=0.7,
               label=r'$\frac{2}{\eta}(1-\beta)$')

    # Plot loss curves in bottom subplot
    for variant in ['A', 'B', 'C']:
        if variant not in runs:
            continue
        run = runs[variant]
        try:
            df = load_results(run)
        except RuntimeError:
            continue
        # Always plot full_loss
        full_loss_data = df[['step', 'full_loss']].dropna()
        if not full_loss_data.empty:
            ax_loss.plot(
                full_loss_data['step'],
                full_loss_data['full_loss'],
                color=VARIANT_COLORS[variant],
                linewidth=2.4,
                alpha=0.8,
            )
        # Optionally plot batch_loss (mini-batch loss)
        if show_mini_loss:
            batch_loss_data = df[['step', 'batch_loss']].dropna()
            if not batch_loss_data.empty:
                ax_loss.plot(
                    batch_loss_data['step'],
                    batch_loss_data['batch_loss'],
                    color=VARIANT_COLORS[variant],
                    linewidth=0.6,
                    alpha=0.4,
                )
    # Add legend entries for loss types (black lines)
    ax_loss.plot([], [], color='black', linewidth=2.4, alpha=0.8, label='Full Loss')
    if show_mini_loss:
        ax_loss.plot([], [], color='black', linewidth=0.6, alpha=0.4, label='Mini-batch Loss')
    ax_loss.set_ylabel('Loss', fontsize=14)
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc='upper right', fontsize=12, framealpha=0.9,
                   prop={'weight': 'medium'})

    # Generate dynamic labels based on experiment type
    def get_variant_label(variant: str, run: RunInfo, experiment_type: str,
                          run_a: Optional[RunInfo], run_b: Optional[RunInfo]) -> str:
        """Generate label showing the relevant hyperparameter for this experiment type.

        For Run C, uses the intervention values stored in the run itself (from results.txt),
        falling back to Run B's values if not available.
        """
        if experiment_type == 'lr':
            if variant == 'A':
                return f'$\\eta$={run.lr}'
            elif variant == 'B':
                return f'$\\eta$={run.lr}'
            elif variant == 'C':
                start_lr = run.lr
                end_lr = run.intervention_lr if run.intervention_lr is not None else (run_b.lr if run_b else '?')
                return f'$\\eta$={start_lr} $\\rightarrow$ {end_lr}'
        elif experiment_type == 'momentum':
            if variant == 'A':
                return f'$\\beta$={run.momentum}'
            elif variant == 'B':
                return f'$\\beta$={run.momentum}'
            elif variant == 'C':
                start_mom = run.momentum
                end_mom = run.intervention_momentum if run.intervention_momentum is not None else (run_b.momentum if run_b else '?')
                return f'$\\beta$={start_mom} $\\rightarrow$ {end_mom}'
        elif experiment_type == 'batch':
            if variant == 'A':
                return f'$b$={run.batch_size}'
            elif variant == 'B':
                return f'$b$={run.batch_size}'
            elif variant == 'C':
                start_batch = run.batch_size
                end_batch = run.intervention_batch if run.intervention_batch is not None else (run_b.batch_size if run_b else '?')
                return f'$b$={start_batch} $\\rightarrow$ {end_batch}'
        # Fallback
        return f'{variant}'

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
            label = get_variant_label(variant, run, experiment_type, run_a, run_b)
            ax.plot(
                batch_sharp['step'],
                batch_sharp['batch_sharpness'],
                label=label,
                color=VARIANT_COLORS[variant],
                linewidth=1.8,
                alpha=0.8,
            )

        # Plot lambda_max with circle markers
        lambda_max = df[['step', 'lambda_max']].dropna()
        if not lambda_max.empty:
            ax.plot(
                lambda_max['step'],
                lambda_max['lambda_max'],
                color=VARIANT_COLORS[variant],
                linestyle='',
                marker='o',
                markersize=4,
                alpha=0.8,
            )

    # Add legend entries for lambda_max and batch sharpness (black)
    ax.plot([], [], color='black', linestyle='', marker='o', markersize=4, alpha=0.8, label=r'$\lambda_{\max}$')
    ax.plot([], [], color='black', linewidth=1.8, alpha=0.8, label='Batch Sharpness')

    # Mark intervention step with vertical line (no legend entry)
    if intervention_step is not None:
        ax.axvline(x=intervention_step, color=vert_color, linestyle='--', linewidth=2.0)
        ax_loss.axvline(x=intervention_step, color=vert_color, linestyle='--', linewidth=2.0)

    # Formatting
    ax_loss.set_xlabel('Step', fontsize=18)
    ax.set_ylabel('Sharpness', fontsize=18)

    type_labels = {
        'lr': 'Learning Rate',
        'momentum': 'Momentum',
        'batch': 'Batch Size',
    }
    ax.set_title(f'Intervening {type_labels.get(experiment_type, experiment_type)} Mid-Training',
                 fontsize=26)

    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9,
              prop={'weight': 'medium'})

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

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
    parser.add_argument(
        '--mini-loss',
        action='store_true',
        help='Also show mini-batch loss in addition to full loss',
    )
    parser.add_argument(
        '--vert-color',
        type=str,
        default='red',
        help='Color of the vertical intervention line (default: red)',
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

        # Determine intervention step (prefer command line arg, then from Run C's results)
        intervention_step = args.intervention_step
        if intervention_step is None and 'C' in runs:
            intervention_step = runs['C'].intervention_step

        # Extract model from folder path (e.g., cifar10_mlp)
        model = ""
        if runs:
            first_run = list(runs.values())[0]
            parent_name = first_run.folder.parent.name
            if '_' in parent_name:
                model = parent_name.split('_')[1]

        # Generate plot
        fig = plot_intervention_comparison(
            runs, exp_type, intervention_step,
            show_mini_loss=args.mini_loss,
            vert_color=args.vert_color,
        )
        output_path = save_figure(fig, exp_type, model)
        print(f"  Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
