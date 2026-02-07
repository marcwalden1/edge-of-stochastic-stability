#!/usr/bin/env python3
"""
Plot batch sharpness and lambda_max vs training steps.

Unlike plot_results.py, this script does not restrict the y-axis range,
allowing the full range of values to be visible.

Usage:
    # Plot the most recent run
    python visualization/plot_sharpness_lmax.py

    # Plot all runs
    python visualization/plot_sharpness_lmax.py --all

    # Plot a specific run folder
    python visualization/plot_sharpness_lmax.py --run path/to/run/folder
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


class ResultsConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunInfo:
    folder: Path
    batch_size: int
    lr: float


def require_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ResultsConfigError(f"Set {name} before running this script.")
    return Path(value)


def iter_run_folders(results_root: Path) -> Iterable[Path]:
    for dataset_folder in results_root.iterdir():
        if not dataset_folder.is_dir():
            continue
        yield from (child for child in dataset_folder.iterdir() if child.is_dir())


def parse_run_info(folder: Path) -> RunInfo:
    """Parse batch size and learning rate from folder name."""
    parts = folder.name.split('_')
    try:
        lr_token = next(p for p in parts if p.startswith('lr'))
        batch_token = next(p for p in parts if p.startswith('b') and p[1:].isdigit())
        lr = float(lr_token[2:])
        batch_size = int(batch_token[1:])
    except (StopIteration, ValueError) as exc:
        raise ResultsConfigError(f"Unrecognised folder naming scheme: {folder.name}") from exc
    return RunInfo(folder=folder, batch_size=batch_size, lr=lr)


def latest_run(results_root: Path) -> RunInfo:
    runs = sorted(iter_run_folders(results_root), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise ResultsConfigError(f"No runs found under {results_root}")
    return parse_run_info(runs[-1])


def load_results(run: RunInfo) -> pd.DataFrame:
    file_path = run.folder / 'results.txt'
    if not file_path.exists():
        raise ResultsConfigError(f"Missing results.txt in {run.folder}")

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


def plot_metrics(df: pd.DataFrame, run: RunInfo) -> plt.Figure:
    """Plot batch sharpness and lambda_max vs training steps."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Reference line at 2/eta
    ax.axhline(y=2 / run.lr, color='black', linestyle='--', label=r'2/$\eta$')

    # Plot batch sharpness
    batch_sharp = df[['step', 'batch_sharpness']].dropna()
    if not batch_sharp.empty:
        ax.plot(batch_sharp['step'], batch_sharp['batch_sharpness'],
                label='batch sharpness', color='#2ca02c')

    # Plot lambda_max
    lmax = df[['step', 'lambda_max']].dropna()
    if not lmax.empty:
        ax.plot(lmax['step'], lmax['lambda_max'],
                label=r'$\lambda_{max}$', color='#1f77b4')

    ax.set_xlabel('steps')
    ax.set_ylabel('sharpness')
    ax.set_title(f'Batch Sharpness & λ_max (batch size {run.batch_size}, lr={run.lr})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Loss on secondary y-axis (log scale)
    ax_loss = ax.twinx()
    loss = df[['step', 'full_loss']].dropna()
    if not loss.empty:
        ax_loss.plot(loss['step'], loss['full_loss'], color='gray', label='full batch loss')
        ax_loss.set_yscale('log')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(loc='upper right')

    return fig


def save_figure(fig: plt.Figure, run: RunInfo) -> Path:
    script_dir = Path(__file__).parent
    img_dir = script_dir / 'img'
    img_dir.mkdir(exist_ok=True)

    filename = f"{run.folder.name}_sharpness_lmax.png"
    output_path = img_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot batch sharpness and lambda_max vs training steps."
    )
    parser.add_argument("--all", action="store_true",
                        help="Plot all runs under $RESULTS/plaintext")
    parser.add_argument("--run", type=str, default=None,
                        help="Path to a specific run folder")
    args = parser.parse_args()

    if args.run:
        # Plot a specific run
        folder = Path(args.run)
        if not folder.exists():
            raise ResultsConfigError(f"Run folder not found: {folder}")
        run = parse_run_info(folder)
        df = load_results(run)
        fig = plot_metrics(df, run)
        output_path = save_figure(fig, run)
        print(f"Plot saved to: {output_path}")
    elif args.all:
        # Plot all runs
        results_root = require_env_path('RESULTS') / 'plaintext'
        any_plotted = False

        for folder in iter_run_folders(results_root):
            try:
                run = parse_run_info(folder)
            except ResultsConfigError:
                continue

            try:
                df = load_results(run)
            except ResultsConfigError:
                continue

            fig = plot_metrics(df, run)
            output_path = save_figure(fig, run)
            print(f"Plot saved to: {output_path}")
            any_plotted = True

        if not any_plotted:
            raise ResultsConfigError(f"No valid runs found under {results_root}")
    else:
        # Plot the most recent run
        results_root = require_env_path('RESULTS') / 'plaintext'
        run = latest_run(results_root)
        print(f"Using the most recent folder: {run.folder}")

        df = load_results(run)
        fig = plot_metrics(df, run)
        output_path = save_figure(fig, run)
        print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
