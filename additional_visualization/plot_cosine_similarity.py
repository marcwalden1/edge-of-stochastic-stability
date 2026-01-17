from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class ResultsConfigError(RuntimeError):
    pass


def require_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ResultsConfigError(f"Set {name} before running this script.")
    return Path(value)


def iter_run_folders(results_root: Path):
    """Iterate over run folders within a results directory."""
    for dataset_folder in results_root.iterdir():
        if not dataset_folder.is_dir():
            continue
        yield from (child for child in dataset_folder.iterdir() if child.is_dir())


def parse_batch_size_from_folder(folder: Path) -> int | None:
    """Extract batch size from folder name (e.g., ...b64... -> 64)."""
    parts = folder.name.split('_')
    for p in parts:
        if p.startswith('b') and p[1:].isdigit():
            return int(p[1:])
    return None


def load_cosine_similarity(folder: Path) -> pd.DataFrame | None:
    """Load cosine_similarity.csv from a run folder."""
    file_path = folder / 'cosine_similarity.csv'
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    return df


def rolling_average(series: pd.Series, window: int = 50) -> pd.Series:
    """Apply rolling average smoothing."""
    if series.empty:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity metrics across batch sizes."
    )
    parser.add_argument(
        "--subdir",
        type=str,
        required=True,
        help="Relative subdir under $RESULTS/plaintext (e.g., cifar10_mlp)"
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Explicit run folder names to include"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Batch sizes to include (e.g., 1 2 4 16 32 64 128 1024)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps to plot"
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Rolling average window for smoothing (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image filename prefix (default: auto)"
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Skip histogram plots"
    )

    args = parser.parse_args()

    base_root = require_env_path('RESULTS') / 'plaintext'
    results_root = base_root / args.subdir

    # Find matching runs
    runs: List[Tuple[Path, int]] = []  # (folder, batch_size)

    if args.runs:
        candidate_folders = [results_root / r for r in args.runs]
    else:
        candidate_folders = list(iter_run_folders(results_root))

    for folder in candidate_folders:
        if not folder.is_dir():
            continue

        # Try to get batch size from folder name
        batch_size = parse_batch_size_from_folder(folder)

        # Check if cosine_similarity.csv exists
        csv_path = folder / 'cosine_similarity.csv'
        if not csv_path.exists():
            continue

        # If batch_size not in folder name, try to read from CSV
        if batch_size is None:
            try:
                df = pd.read_csv(csv_path, nrows=1)
                if 'batch_size' in df.columns:
                    batch_size = int(df['batch_size'].iloc[0])
            except Exception:
                continue

        if batch_size is None:
            continue

        if args.batch_sizes and batch_size not in args.batch_sizes:
            continue

        runs.append((folder, batch_size))

    if not runs:
        raise ResultsConfigError(
            f"No runs with cosine_similarity.csv found under {results_root}"
        )

    # Sort runs by batch size
    runs.sort(key=lambda x: x[1])

    # Color palette for different batch sizes
    cmap = plt.cm.viridis
    batch_sizes = sorted(set(bs for _, bs in runs))
    colors = {bs: cmap(i / max(1, len(batch_sizes) - 1)) for i, bs in enumerate(batch_sizes)}

    # Load all data
    data: Dict[int, pd.DataFrame] = {}
    for folder, batch_size in runs:
        df = load_cosine_similarity(folder)
        if df is not None:
            if args.max_steps is not None:
                df = df[df['step'] <= args.max_steps]
            if batch_size in data:
                # Append if same batch size from multiple runs
                data[batch_size] = pd.concat([data[batch_size], df], ignore_index=True)
            else:
                data[batch_size] = df

    # --------------------------------------------------------------------------
    # Plot 1: Time series of cosine similarity (consecutive gradients)
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 5))

    for batch_size in batch_sizes:
        if batch_size not in data:
            continue
        df = data[batch_size]
        df_sorted = df.sort_values('step')

        y = df_sorted['cos_sim_consecutive_grad'].dropna()
        x = df_sorted.loc[y.index, 'step']

        if len(y) == 0:
            continue

        if args.smooth > 1:
            y_smooth = rolling_average(y, window=args.smooth)
        else:
            y_smooth = y

        ax1.plot(x, y_smooth, label=f'B={batch_size}', color=colors[batch_size], alpha=0.8)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Consecutive Gradient Cosine Similarity: cos(g_t, g_{t+1})')
    ax1.legend(loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.05, 1.05)

    # --------------------------------------------------------------------------
    # Plot 2: Time series of cosine similarity (gradient-momentum)
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    for batch_size in batch_sizes:
        if batch_size not in data:
            continue
        df = data[batch_size]
        df_sorted = df.sort_values('step')

        y = df_sorted['cos_sim_grad_momentum'].dropna()
        x = df_sorted.loc[y.index, 'step']

        if len(y) == 0:
            continue

        if args.smooth > 1:
            y_smooth = rolling_average(y, window=args.smooth)
        else:
            y_smooth = y

        ax2.plot(x, y_smooth, label=f'B={batch_size}', color=colors[batch_size], alpha=0.8)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Gradient-Momentum Cosine Similarity: cos(g_t, v_t)')
    ax2.legend(loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.05, 1.05)

    # --------------------------------------------------------------------------
    # Plot 3 & 4: Histograms (if not disabled)
    # --------------------------------------------------------------------------
    if not args.no_histogram:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        fig4, ax4 = plt.subplots(figsize=(12, 5))

        for batch_size in batch_sizes:
            if batch_size not in data:
                continue
            df = data[batch_size]

            # Consecutive gradient histogram
            vals1 = df['cos_sim_consecutive_grad'].dropna()
            if len(vals1) > 0:
                ax3.hist(
                    vals1, bins=50, alpha=0.5,
                    label=f'B={batch_size}', color=colors[batch_size],
                    density=True
                )

            # Gradient-momentum histogram
            vals2 = df['cos_sim_grad_momentum'].dropna()
            if len(vals2) > 0:
                ax4.hist(
                    vals2, bins=50, alpha=0.5,
                    label=f'B={batch_size}', color=colors[batch_size],
                    density=True
                )

        ax3.set_xlabel('Cosine Similarity')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Consecutive Gradient Cosine Similarity')
        ax3.legend(loc='best', ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-1.05, 1.05)

        ax4.set_xlabel('Cosine Similarity')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Gradient-Momentum Cosine Similarity')
        ax4.legend(loc='best', ncol=2)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-1.05, 1.05)

    # --------------------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------------------
    script_dir = Path(__file__).parent
    img_dir = script_dir / 'img'
    img_dir.mkdir(exist_ok=True)

    if args.output:
        prefix = args.output
    else:
        bs_tag = '_'.join(str(bs) for bs in batch_sizes)
        prefix = f"cosine_sim_{args.subdir}_bs_{bs_tag}"

    path1 = img_dir / f"{prefix}_consecutive.png"
    path2 = img_dir / f"{prefix}_momentum.png"

    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    print(f"Saved: {path1}")
    print(f"Saved: {path2}")

    if not args.no_histogram:
        path3 = img_dir / f"{prefix}_hist_consecutive.png"
        path4 = img_dir / f"{prefix}_hist_momentum.png"
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        print(f"Saved: {path3}")
        print(f"Saved: {path4}")

    plt.close('all')


if __name__ == '__main__':
    main()
