#!/usr/bin/env python3
"""
Calculate plateau/stabilizing values from training results.
Reads results.txt files written by training.py and computes moving averages and plateau values,
then writes plateau_values.csv and plateau_values.json next to this script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import argparse


def _parse_batch_and_lr_from_folder(folder_name: str) -> tuple[int | None, float | None]:
    batch_size: int | None = None
    lr: float | None = None
    parts = folder_name.split('_')
    for part in parts:
        if part.startswith('b') and part[1:].isdigit():
            batch_size = int(part[1:])
        elif part.startswith('lr'):
            try:
                lr = float(part[2:])
            except ValueError:
                pass
    return batch_size, lr


def calculate_plateau_values(results_root: Path) -> List[Dict[str, Any]]:
    """Calculate plateau values for all runs under results_root.

    Returns a list of dicts (one per run) containing plateau/moving-average/final values.
    """
    results: List[Dict[str, Any]] = []

    for folder in sorted(p for p in results_root.iterdir() if p.is_dir()):
        results_file = folder / 'results.txt'
        if not results_file.exists():
            continue

        batch_size, lr = _parse_batch_and_lr_from_folder(folder.name)
        if batch_size is None:
            # Skip folders that don't follow the naming convention
            continue

        try:
            # Read metrics; ignore commented header lines starting with '#'
            df = pd.read_csv(
                results_file,
                sep=',',
                header=None,
                names=[
                    'epoch',
                    'step',
                    'batch_loss',
                    'full_loss',
                    'lambda_max',
                    'step_sharpness',
                    'batch_sharpness',
                    'gni',
                    'full_accuracy',
                    'adaptive_batch_sharpness',
                    'adaptive_batch_sharpness_momentum',
                    'lmax_preconditioned',
                ],
                na_values=['nan'],
                skipinitialspace=True,
                comment='#',  # skip the single header line written by initialize_folders
                engine='python',
            )

            if df.empty:
                continue

            # Robust typing
            for col in ['epoch', 'step']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            result: Dict[str, Any] = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'folder': folder.name,
                'total_steps': int(len(df)),
                'final_step': int(df['step'].dropna().iloc[-1]) if 'step' in df.columns and not df['step'].isna().all() else None,
            }

            # Metrics to aggregate
            metrics = ['batch_sharpness', 'lambda_max', 'step_sharpness', 'gni', 'full_loss',
                       'adaptive_batch_sharpness', 'adaptive_batch_sharpness_momentum',
                       'lmax_preconditioned']
            for metric in metrics:
                if metric not in df.columns:
                    continue
                series = pd.to_numeric(df[metric], errors='coerce').dropna()
                if series.empty:
                    continue

                # Moving average over the run (10% window of actual measurements)
                window = max(1, int(len(series) * 0.1))
                ma = series.rolling(window=window, center=True, min_periods=1).mean()
                result[f'{metric}_moving_avg_final'] = float(ma.iloc[-1]) if not ma.empty else None

                # Plateau: average of the last 20% of actual measurements (not rows)
                metric_plateau_idx = int(len(series) * 0.8)
                plateau_series = series.iloc[metric_plateau_idx:]
                result[f'{metric}_plateau'] = float(plateau_series.mean()) if not plateau_series.empty else None
                result[f'{metric}_plateau_std'] = float(plateau_series.std()) if not plateau_series.empty else None

                # Final value in the log
                result[f'{metric}_final'] = float(series.iloc[-1]) if not series.empty else None

            results.append(result)

        except Exception as e:
            print(f"Error processing {folder.name}: {e}", file=os.sys.stderr)
            continue

    return sorted(results, key=lambda x: x['batch_size'])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute plateau values from results.")
    parser.add_argument(
        "--subdir",
        default="plaintext/cifar10_cnn",
        help='Relative subdir under RESULTS (e.g., "plaintext/cifar10_cnn" or "plaintext/cifar10_mlp")',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help='Output directory for CSV file (default: tries /tmp, then current directory, then home)',
    )
    args = parser.parse_args()

    # Resolve RESULTS directory and default to ~/results
    results_base = os.environ.get('RESULTS')
    if results_base:
        base = Path(results_base)
    else:
        base = Path(os.path.expanduser('~/results'))

    # Expected structure: RESULTS/<subdir>/<timestamp>_lrXXXX_bYYYY
    results_root = base / Path(args.subdir)

    if not results_root.exists():
        print(f"Error: Results directory not found: {results_root}")
        print("Set RESULTS env var or pass --subdir (e.g., plaintext/cifar10_cnn)")
        return 1

    print(f"Analyzing results in: {results_root}")
    print("Calculating plateau values...")

    results = calculate_plateau_values(results_root)

    if not results:
        print("No results found!")
        return 1

    # Determine output directory: try user-specified, then /tmp, then current dir, then home
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Try multiple locations in order of preference
        candidates = [
            Path('/tmp'),  # Usually has more space on clusters
            Path('.'),     # Current directory
            Path(os.path.expanduser('~')),  # Home directory (may have quota issues)
        ]
        out_dir = None
        for candidate in candidates:
            try:
                # Test write access
                test_file = candidate / '.plateaucsv_test'
                test_file.write_text('test')
                test_file.unlink()
                out_dir = candidate
                break
            except (OSError, PermissionError):
                continue
        
        if out_dir is None:
            print("Error: Could not find a writable output directory. Try --output-dir")
            return 1

    output_csv = out_dir / 'plateau_values.csv'

    # Save CSV with error handling
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV results to: {output_csv}")
    except OSError as e:
        if e.errno == 122:  # Disk quota exceeded
            print(f"Error: Disk quota exceeded when writing to {output_csv}")
            print(f"Try specifying a different location with --output-dir (e.g., --output-dir /tmp)")
            return 1
        else:
            raise

    # Print brief summary (first 10 runs)
    print(f"\nFound {len(results)} completed runs")
    print("\nSummary (batch_size, batch_sharpness_plateau, lambda_max_plateau):")
    for r in results[:10]:
        bs = r.get('batch_size')
        bsp = r.get('batch_sharpness_plateau')
        lmp = r.get('lambda_max_plateau')
        bsp_str = f"{bsp:.4f}" if isinstance(bsp, (int, float)) and bsp is not None else "N/A"
        lmp_str = f"{lmp:.4f}" if isinstance(lmp, (int, float)) and lmp is not None else "N/A"
        print(f"  Batch {bs:4d}: batch_sharpness={bsp_str:>8}, lambda_max={lmp_str:>8}")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())