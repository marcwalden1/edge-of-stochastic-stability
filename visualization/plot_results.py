from __future__ import annotations

import os
from dataclasses import dataclass
import argparse
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


def latest_run(results_root: Path) -> RunInfo:
    runs = sorted(iter_run_folders(results_root), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise ResultsConfigError(f"No runs found under {results_root}")

    folder = runs[-1]
    parts = folder.name.split('_')

    try:
        lr_token = next(p for p in parts if p.startswith('lr'))
        batch_token = next(p for p in parts if p.startswith('b'))
        lr = float(lr_token[2:])
        batch_size = int(batch_token[1:])
    except (StopIteration, ValueError) as exc:  # pragma: no cover - folder naming fallback
        raise ResultsConfigError(f"Unrecognised folder naming scheme: {folder.name}") from exc

    return RunInfo(folder=folder, batch_size=batch_size, lr=lr)


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


def rolling_average(series: pd.Series, window_fraction: float = 0.02) -> pd.Series:
    if series.empty:
        return series

    window = max(1, int(len(series) * window_fraction))
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_metrics(df: pd.DataFrame, run: RunInfo) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(y=2 / run.lr, color='black', linestyle='--', label=r'2/$\eta$')

    batch_sharp = df[['step', 'batch_sharpness']].dropna()
    if not batch_sharp.empty:
        ax.plot(batch_sharp['step'], batch_sharp['batch_sharpness'], label='batch sharpness', color='#2ca02c')

    step_sharp = df[['step', 'step_sharpness']].dropna()
    if not step_sharp.empty:
        averaged = rolling_average(step_sharp['step_sharpness'])
        ax.plot(step_sharp['step'], averaged, label='step sharpness (avg)', color='#d62728')

    lmax = df[['step', 'lambda_max']].dropna()
    if not lmax.empty:
        ax.plot(lmax['step'], lmax['lambda_max'], label=r'$\lambda_{max}$', color='#1f77b4')

    gni = df[['step', 'gni']].dropna()
    if not gni.empty:
        ax.plot(gni['step'], gni['gni'], label='GNI', color='#9467bd', alpha=0.7)

    # Plot adaptive batch sharpness
    if 'adaptive_batch_sharpness' in df.columns:
        abs_data = df[['step', 'adaptive_batch_sharpness']].dropna()
        if not abs_data.empty:
            ax.plot(abs_data['step'], abs_data['adaptive_batch_sharpness'],
                    label='adaptive batch sharpn', color='#ff7f0e')

    # Plot adaptive batch sharpness with momentum
    if 'adaptive_batch_sharpness_momentum' in df.columns:
        absm_data = df[['step', 'adaptive_batch_sharpness_momentum']].dropna()
        if not absm_data.empty:
            ax.plot(absm_data['step'], absm_data['adaptive_batch_sharpness_momentum'],
                    label='adaptive batch sharpn (mom)', color='#e377c2')

    # Plot preconditioned lambda_max
    if 'lmax_preconditioned' in df.columns:
        lmax_pc = df[['step', 'lmax_preconditioned']].dropna()
        if not lmax_pc.empty:
            ax.plot(lmax_pc['step'], lmax_pc['lmax_preconditioned'],
                    label=r'$\lambda_{max}(P^{-1}H)$', color='#8c564b')

    ax.set_ylim(1, 4 / run.lr)
    ax.set_xlabel('steps')
    ax.set_title(f'batch size {run.batch_size}')
    ax.legend(loc='upper left')

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

    filename = f"{run.folder.name}_results.png"
    output_path = img_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training results.")
    parser.add_argument("--all", action="store_true", help="Plot all runs under $RESULTS/plaintext")
    args = parser.parse_args()

    results_root = require_env_path('RESULTS') / 'plaintext'

    if args.all:
        any_plotted = False
        for folder in iter_run_folders(results_root):
            try:
                run = latest_run(folder.parent) if False else RunInfo(
                    folder=folder,
                    batch_size=int(next(p for p in folder.name.split('_') if p.startswith('b'))[1:]),
                    lr=float(next(p for p in folder.name.split('_') if p.startswith('lr'))[2:]),
                )
            except Exception:
                # Skip folders that don't follow naming convention
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
        run = latest_run(results_root)
        print(f"Using the most recent folder: {run.folder}")

        df = load_results(run)
        fig = plot_metrics(df, run)
        output_path = save_figure(fig, run)
        print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
