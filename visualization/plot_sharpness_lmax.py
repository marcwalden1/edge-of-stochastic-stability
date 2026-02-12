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
    "adaptive_batch_sharpness",
    "adaptive_batch_sharpness_momentum",
    "lmax_preconditioned",
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


def plot_metrics(df: pd.DataFrame, run: RunInfo,
                 ylimtop: float = None, ylimbottom: float = None) -> plt.Figure:
    """Plot batch sharpness and lambda_max vs training steps.

    Top subplot: batch sharpness, lambda_max, loss (always shown).
    Bottom subplot: adaptive sharpness, adaptive sharpness (momentum),
    lmax_preconditioned, loss (only if any preconditioned data exists).
    """
    # Check if any preconditioned data exists
    has_preconditioned = False
    for col in ('adaptive_batch_sharpness', 'adaptive_batch_sharpness_momentum', 'lmax_preconditioned'):
        if col in df.columns and df[col].notna().any():
            has_preconditioned = True

    if has_preconditioned:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    else:
        fig, ax_top = plt.subplots(figsize=(10, 5))
        ax_bot = None

    # --- Top subplot: batch sharpness, lambda_max, loss ---
    batch_sharp = df[['step', 'batch_sharpness']].dropna()
    if not batch_sharp.empty:
        ax_top.plot(batch_sharp['step'], batch_sharp['batch_sharpness'],
                    label='batch sharpness', color='#2ca02c')

    lmax = df[['step', 'lambda_max']].dropna()
    if not lmax.empty:
        ax_top.plot(lmax['step'], lmax['lambda_max'],
                    label=r'$\lambda_{max}$', color='#1f77b4')

    ax_top.set_ylabel('sharpness')
    two_over_eta = 2 / run.lr
    ax_top.set_title(f'Batch Sharpness & λ_max (batch size {run.batch_size}, lr={run.lr}, 2/η={two_over_eta:.1f})')
    ax_top.legend(loc='upper left')
    ax_top.grid(True, alpha=0.3)
    if ylimtop is not None:
        ax_top.set_ylim(bottom=0, top=ylimtop)

    loss_ax_top = ax_top.twinx()
    loss = df[['step', 'full_loss']].dropna()
    if not loss.empty:
        loss_ax_top.plot(loss['step'], loss['full_loss'], color='gray', alpha=0.3, label='loss')
        loss_ax_top.set_yscale('log')
        loss_ax_top.set_ylabel('Loss (log)')
        loss_ax_top.legend(loc='upper right')

    # --- Bottom subplot: preconditioned metrics + loss ---
    if ax_bot is not None:
        if 'adaptive_batch_sharpness' in df.columns:
            abs_data = df[['step', 'adaptive_batch_sharpness']].dropna()
            if not abs_data.empty:
                ax_bot.plot(abs_data['step'], abs_data['adaptive_batch_sharpness'],
                            label='adaptive batch sharpness', color='#ff7f0e')

        if 'adaptive_batch_sharpness_momentum' in df.columns:
            absm_data = df[['step', 'adaptive_batch_sharpness_momentum']].dropna()
            if not absm_data.empty:
                ax_bot.plot(absm_data['step'], absm_data['adaptive_batch_sharpness_momentum'],
                            label='adaptive batch sharpness momentum', color='#d62728')

        if 'lmax_preconditioned' in df.columns:
            lmax_pc = df[['step', 'lmax_preconditioned']].dropna()
            if not lmax_pc.empty:
                ax_bot.plot(lmax_pc['step'], lmax_pc['lmax_preconditioned'],
                            label=r'$\lambda_{max}(P^{-1}H)$', color='#9467bd')

        ax_bot.set_xlabel('steps')
        ax_bot.set_ylabel('sharpness')
        ax_bot.legend(loc='upper left')
        ax_bot.grid(True, alpha=0.3)
        if ylimbottom is not None:
            ax_bot.set_ylim(bottom=0, top=ylimbottom)

        loss_ax_bot = ax_bot.twinx()
        if not loss.empty:
            loss_ax_bot.plot(loss['step'], loss['full_loss'], color='gray', alpha=0.3, label='loss')
            loss_ax_bot.set_yscale('log')
            loss_ax_bot.set_ylabel('Loss (log)')
            loss_ax_bot.legend(loc='upper right')
    else:
        ax_top.set_xlabel('steps')

    plt.tight_layout()
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
    parser.add_argument("--ylimtop", type=float, default=None,
                        help="Upper y-axis limit for top subplot (e.g., 500)")
    parser.add_argument("--ylimbottom", type=float, default=None,
                        help="Upper y-axis limit for bottom subplot (e.g., 100)")
    args = parser.parse_args()

    if args.run:
        # Plot a specific run
        folder = Path(args.run)
        if not folder.exists():
            raise ResultsConfigError(f"Run folder not found: {folder}")
        run = parse_run_info(folder)
        df = load_results(run)
        fig = plot_metrics(df, run, ylimtop=args.ylimtop, ylimbottom=args.ylimbottom)
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

            fig = plot_metrics(df, run, ylimtop=args.ylimtop, ylimbottom=args.ylimbottom)
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
        fig = plot_metrics(df, run, ylimtop=args.ylimtop, ylimbottom=args.ylimbottom)
        output_path = save_figure(fig, run)
        print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
