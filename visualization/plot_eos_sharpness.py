from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Iterable, List

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


def parse_run_info(folder: Path) -> tuple[int, float]:
    parts = folder.name.split('_')
    lr = float(next(p for p in parts if p.startswith('lr'))[2:])
    batch_size = int(next(p for p in parts if p.startswith('b'))[1:])
    return batch_size, lr


def load_results(folder: Path) -> pd.DataFrame:
    file_path = folder / 'results.txt'
    if not file_path.exists():
        raise ResultsConfigError(f"Missing results.txt in {folder}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="EoS plot: metrics vs steps across multiple runs.")
    parser.add_argument("--subdir", type=str, required=True, help="Relative subdir under $RESULTS/plaintext (e.g., cifar10_mlp)")
    parser.add_argument("--batch", type=int, required=True, help="Batch size to include (e.g., 8192)")
    parser.add_argument("--lrs", type=float, nargs="+", required=True, help="List of learning rates to include (e.g., 0.02 0.01 0.0066667 0.005)")
    parser.add_argument("--include-batch-sharpness", action="store_true", help="Include batch_sharpness curves")
    parser.add_argument("--include-sharpness", action="store_true", help="Include step_sharpness (averaged) curves")
    parser.add_argument("--include-loss", action="store_true", help="Include full_loss curve (secondary axis)")
    parser.add_argument("--output", type=str, default=None, help="Output image filename (default auto)")

    args = parser.parse_args()

    # Default to batch sharpness only if no metric flags provided
    if not any([args.include_batch_sharpness, args.include_sharpness, args.include_loss]):
        args.include_batch_sharpness = True

    base_root = require_env_path('RESULTS') / 'plaintext'
    results_root = base_root / args.subdir

    # Find matching runs
    runs: List[tuple[Path, int, float]] = []
    for folder in iter_run_folders(results_root):
        try:
            bsz, lr = parse_run_info(folder)
        except Exception:
            continue
        if bsz != args.batch:
            continue
        if lr not in args.lrs:
            continue
        if not (folder / 'results.txt').exists():
            continue
        runs.append((folder, bsz, lr))

    if not runs:
        raise ResultsConfigError(f"No runs found under {results_root} matching batch={args.batch} and lrs={args.lrs}")

    # Colors per curve (up to four): blue, orange, green, red
    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(10, 5))
    loss_ax = ax.twinx() if args.include_loss else None

    # Plot each run
    for idx, (folder, bsz, lr) in enumerate(sorted(runs, key=lambda r: r[2])):
        df = load_results(folder)
        color = color_cycle[idx % len(color_cycle)]
        label = f"lr={lr}"

        if args.include_batch_sharpness:
            bs_df = df[['step', 'batch_sharpness']].dropna()
            if not bs_df.empty:
                ax.plot(bs_df['step'], bs_df['batch_sharpness'], color=color, label=label)

        if args.include_sharpness:
            ss_df = df[['step', 'step_sharpness']].dropna()
            if not ss_df.empty:
                averaged = rolling_average(ss_df['step_sharpness'])
                ax.plot(ss_df['step'], averaged, color=color, linestyle='--', label=label + ' (step)')

        if args.include_loss and loss_ax is not None:
            loss_df = df[['step', 'full_loss']].dropna()
            if not loss_df.empty:
                loss_ax.plot(loss_df['step'], loss_df['full_loss'], color=color, alpha=0.4, label=label + ' loss')
                loss_ax.set_yscale('log')
                loss_ax.set_ylabel('Loss (log)')

    ax.set_xlabel('steps')
    ax.set_ylabel('sharpness')
    ax.set_title(f"Metrics vs steps ({args.subdir}) batch={args.batch}")
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left')
    if loss_ax is not None:
        loss_ax.legend(loc='upper right')

    # Save output
    script_dir = Path(__file__).parent
    img_dir = script_dir / 'img'
    img_dir.mkdir(exist_ok=True)
    if args.output:
        output_path = img_dir / args.output
    else:
        lr_tag = '_'.join(str(x) for x in args.lrs)
        output_path = img_dir / f"eos_{args.subdir}_b{args.batch}_lrs_{lr_tag}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
