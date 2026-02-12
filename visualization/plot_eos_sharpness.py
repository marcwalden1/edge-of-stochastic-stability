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
    "adaptive_batch_sharpness",
    "adaptive_batch_sharpness_momentum",
    "lmax_preconditioned",
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
    lr_token = next(p for p in parts if p.startswith('lr'))
    b_token = next(p for p in parts if p.startswith('b'))
    # Support both lr0.00667 and lr0.0066666 etc.
    lr_str = lr_token[2:]
    batch_size = int(b_token[1:])
    try:
        lr = float(lr_str)
    except ValueError:
        # In case of unexpected formatting, raise a clear error
        raise ResultsConfigError(f"Unable to parse learning rate from folder name: {folder.name}")
    return batch_size, lr


def lr_matches(lr_value: float, targets: List[float]) -> bool:
    # Consider near-equality with small tolerance and common rounding presentation
    for t in targets:
        if abs(lr_value - t) <= max(1e-8, 1e-5 * max(1.0, abs(t))):
            return True
        # Compare with 5 decimal places as strings to handle tokens like 0.00667 vs 0.0066666
        if f"{lr_value:.5f}" == f"{t:.5f}":
            return True
    return False


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
    parser.add_argument("--subdir", type=str, default="cifar10_mlp", help="Relative subdir under $RESULTS/plaintext (default: cifar10_mlp)")
    parser.add_argument("--batch", type=int, required=True, help="Batch size to include (e.g., 8192)")
    parser.add_argument("--lrs", type=float, nargs="+", required=False, default=[], help="List of learning rates to include (e.g., 0.02 0.01 0.0066667 0.005)")
    parser.add_argument("--runs", type=str, nargs="+", required=False, default=None, help="Explicit run folder names under the subdir (e.g., 20251210_2101_35_lr0.00667_b8192)")
    parser.add_argument("--output", type=str, default=None, help="Output image filename (default auto)")
    parser.add_argument("--max-steps", type=int, default=None, help="If set, only plot data with step <= this value (e.g., 12000)")
    parser.add_argument("--ylimtop", type=float, default=None, help="Upper y-axis limit for top subplot (e.g., 500)")
    parser.add_argument("--ylimbottom", type=float, default=None, help="Upper y-axis limit for bottom subplot (e.g., 100)")

    args = parser.parse_args()

    base_root = require_env_path('RESULTS') / 'plaintext'
    results_root = base_root / args.subdir

    # Find matching runs
    runs: List[tuple[Path, int, float]] = []
    candidate_folders: List[Path]
    if args.runs:
        candidate_folders = [results_root / r for r in args.runs]
    else:
        candidate_folders = list(iter_run_folders(results_root))

    for folder in candidate_folders:
        try:
            bsz, lr = parse_run_info(folder)
        except Exception:
            continue
        if bsz != args.batch:
            continue
        if args.lrs:
            if not lr_matches(lr, args.lrs):
                continue
        if not (folder / 'results.txt').exists():
            continue
        runs.append((folder, bsz, lr))

    if not runs:
        raise ResultsConfigError(f"No runs found under {results_root} matching batch={args.batch} and lrs={args.lrs}")

    # Load all dataframes and check if any preconditioned data exists
    sorted_runs = sorted(runs, key=lambda r: r[2])
    run_data = []
    has_preconditioned = False
    for folder, bsz, lr in sorted_runs:
        df = load_results(folder)
        run_data.append((folder, bsz, lr, df))
        for col in ('adaptive_batch_sharpness', 'adaptive_batch_sharpness_momentum', 'lmax_preconditioned'):
            if col in df.columns and df[col].notna().any():
                has_preconditioned = True

    # Colors per curve (up to four): blue, orange, green, red
    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create figure: 1 or 2 subplots depending on whether preconditioned data exists
    if has_preconditioned:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    else:
        fig, ax_top = plt.subplots(figsize=(10, 5))
        ax_bot = None

    loss_ax_top = ax_top.twinx()
    loss_ax_bot = ax_bot.twinx() if ax_bot is not None else None

    # Plot each run
    for idx, (folder, bsz, lr, df) in enumerate(run_data):
        color = color_cycle[idx % len(color_cycle)]
        label = f"lr={lr}"

        # --- Top subplot: batch sharpness, lambda_max, loss ---
        bs_df = df[['step', 'batch_sharpness']].dropna()
        if args.max_steps is not None:
            bs_df = bs_df[bs_df['step'] <= args.max_steps]
        if not bs_df.empty:
            ax_top.plot(bs_df['step'], bs_df['batch_sharpness'], color=color, label=label + ' batch sharpness')

        lmax_df = df[['step', 'lambda_max']].dropna()
        if args.max_steps is not None:
            lmax_df = lmax_df[lmax_df['step'] <= args.max_steps]
        if not lmax_df.empty:
            ax_top.plot(lmax_df['step'], lmax_df['lambda_max'], color=color,
                        marker='o', markersize=2, linestyle='', label=label + r' $\lambda_{max}$')

        loss_df = df[['step', 'full_loss']].dropna()
        if args.max_steps is not None:
            loss_df = loss_df[loss_df['step'] <= args.max_steps]
        if not loss_df.empty:
            loss_ax_top.plot(loss_df['step'], loss_df['full_loss'], color=color, alpha=0.3, label=label + ' loss')
            loss_ax_top.set_yscale('log')
            loss_ax_top.set_ylabel('Loss (log)')

        # 2/eta line on top subplot
        try:
            xmax = args.max_steps if args.max_steps is not None else int(df['step'].max())
        except Exception:
            xmax = None
        if xmax is None or xmax <= 0:
            xmax = 1
        y_level = 2.0 / lr
        ax_top.hlines(y=y_level, xmin=0, xmax=xmax, colors=color, linestyles=':')

        # --- Bottom subplot: preconditioned metrics + loss ---
        if ax_bot is not None:
            if 'adaptive_batch_sharpness' in df.columns:
                abs_df = df[['step', 'adaptive_batch_sharpness']].dropna()
                if args.max_steps is not None:
                    abs_df = abs_df[abs_df['step'] <= args.max_steps]
                if not abs_df.empty:
                    ax_bot.plot(abs_df['step'], abs_df['adaptive_batch_sharpness'], color=color,
                                label=label + ' adaptive sharpness')

            if 'adaptive_batch_sharpness_momentum' in df.columns:
                absm_df = df[['step', 'adaptive_batch_sharpness_momentum']].dropna()
                if args.max_steps is not None:
                    absm_df = absm_df[absm_df['step'] <= args.max_steps]
                if not absm_df.empty:
                    ax_bot.plot(absm_df['step'], absm_df['adaptive_batch_sharpness_momentum'], color=color,
                                linestyle='--', label=label + ' adaptive sharpness (mom)')

            if 'lmax_preconditioned' in df.columns:
                lpc_df = df[['step', 'lmax_preconditioned']].dropna()
                if args.max_steps is not None:
                    lpc_df = lpc_df[lpc_df['step'] <= args.max_steps]
                if not lpc_df.empty:
                    ax_bot.plot(lpc_df['step'], lpc_df['lmax_preconditioned'], color=color,
                                marker='s', markersize=2, linestyle='',
                                label=label + r' $\lambda_{max}(P^{-1}H)$')

            if not loss_df.empty:
                loss_ax_bot.plot(loss_df['step'], loss_df['full_loss'], color=color, alpha=0.3, label=label + ' loss')
                loss_ax_bot.set_yscale('log')
                loss_ax_bot.set_ylabel('Loss (log)')

    # Format top subplot
    ax_top.set_ylabel('sharpness')
    ax_top.set_title(f"Metrics vs steps ({args.subdir}) batch={args.batch}")
    ax_top.grid(True, alpha=0.2)
    ax_top.legend(loc='upper left', fontsize=8)
    loss_ax_top.legend(loc='upper right', fontsize=8)
    if args.ylimtop is not None:
        ax_top.set_ylim(top=args.ylimtop)

    # Format bottom subplot
    if ax_bot is not None:
        ax_bot.set_xlabel('steps')
        ax_bot.set_ylabel('preconditioned sharpness')
        ax_bot.grid(True, alpha=0.2)
        ax_bot.legend(loc='upper left', fontsize=8)
        loss_ax_bot.legend(loc='upper right', fontsize=8)
        if args.ylimbottom is not None:
            ax_bot.set_ylim(top=args.ylimbottom)
    else:
        ax_top.set_xlabel('steps')

    plt.tight_layout()

    # Save output
    script_dir = Path(__file__).parent
    img_dir = script_dir / 'img'
    img_dir.mkdir(exist_ok=True)
    if args.output:
        output_path = img_dir / args.output
    else:
        if args.lrs:
            lr_tag = '_'.join(str(x) for x in args.lrs)
        else:
            lr_tag = '_'.join(f"{r[2]:.5f}" for r in sorted_runs)
        output_path = img_dir / f"eos_{args.subdir}_b{args.batch}_lrs_{lr_tag}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
