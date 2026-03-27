"""
plot_gbs_sweep.py — Visualization for the GBS suite sweep.

Plots all 9 GBS suite quantities vs training step, with 4 batch sizes overlaid
per optimizer. Produces 5 optimizers × 9 quantities = 45 figures.

Usage:
    python visualization/plot_gbs_sweep.py \\
        --results-dir $RESULTS \\
        --out-dir figures/gbs_sweep/

Discovery:
    Globs $RESULTS/plaintext/cifar10_mlp/gbs_sweep/*/gbs_suite.csv
    Folder name prefix expected: {OPT}_{BATCH}_{timestamp}_...
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTITIES = ['gbs', 'gbs_u', 'gbs_ufull', 'gbs_g', 'ss', 'bs', 'cos_sg', 'cos_su', 'cos_gu']

QUANTITY_LABELS = {
    'gbs':      r'GBS  $= E_B[s^T H_B s \,/\, (-s^T g)]$',
    'gbs_u':    r'GBS$_u$  $= E_B[(s \cdot u_B)\,\lambda_B \,/\, (-g \cdot u_B)]$',
    'gbs_ufull': r'GBS$_{u_\mathrm{full}}$  $= E_B[(s \cdot u)\,\lambda_\mathrm{full} \,/\, (-g \cdot u)]$',
    'gbs_g':    r'GBS$_g$  $= E_B[(s \cdot \hat{g})\,\hat{g}^T H \hat{g} \,/\, (-\|g\|)]$',
    'ss':       r'SS  $= E_B[s^T H_B s \,/\, \|s\|^2]$',
    'bs':       r'BS  $= E_B[g^T H_B g \,/\, \|g\|^2]$',
    'cos_sg':   r'$\cos(s_B,\, g_B)$',
    'cos_su':   r'$\cos(s_B,\, u_B)$',
    'cos_gu':   r'$\cos(g_B,\, u_B)$',
}

OPTIMIZERS = ['SGD', 'SGD_m', 'SGD_nest', 'Adam', 'Muon']

BATCH_COLORS = {
    1024: '#1f77b4',   # blue
    256:  '#ff7f0e',   # orange
    64:   '#2ca02c',   # green
    8:    '#d62728',   # red
}

BATCH_SIZES = [1024, 256, 64, 8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_runs(results_dir: Path):
    """Return list of (opt, batch, csv_path) tuples found under gbs_sweep/.

    When multiple runs exist for the same (opt, batch) — e.g. due to requeue
    preemption — picks the one with the most data rows.
    """
    best: dict = {}  # (opt, batch) -> (n_rows, csv_path)

    for csv_path in sorted((results_dir / 'plaintext' / 'cifar10_mlp' / 'gbs_sweep').glob('*/gbs_suite.csv')):
        folder = csv_path.parent.name
        m = re.match(r'^(SGD_nest|SGD_m|SGD|Adam|Muon)_(\d+)_', folder)
        if m is None:
            continue
        opt   = m.group(1)
        batch = int(m.group(2))
        try:
            n_rows = sum(1 for _ in open(csv_path)) - 1  # subtract header
        except Exception:
            n_rows = 0
        key = (opt, batch)
        if key not in best or n_rows > best[key][0]:
            best[key] = (n_rows, csv_path)

    runs = [(opt, batch, info[1]) for (opt, batch), info in best.items()]
    return runs


def load_suite_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['step'])
    df['step'] = df['step'].astype(int)
    return df


def load_results_bs(run_dir: Path) -> pd.DataFrame | None:
    """Load batch sharpness (column 6) from results.txt for cross-check."""
    results_path = run_dir / 'results.txt'
    if not results_path.exists():
        return None
    rows = []
    try:
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 7:
                    continue
                try:
                    step = int(parts[1].strip())
                    bs   = float(parts[6].strip())
                    if not np.isnan(bs) and bs > 0:
                        rows.append({'step': step, 'bs_results': bs})
                except (ValueError, IndexError):
                    continue
    except Exception:
        return None
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_quantity(opt: str, qty: str, run_data: dict, out_dir: Path):
    """Create one figure: optimizer=opt, quantity=qty, 4 batch-size lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for batch in BATCH_SIZES:
        key = (opt, batch)
        if key not in run_data or run_data[key] is None:
            continue
        df = run_data[key]
        if qty not in df.columns:
            continue
        valid = df[['step', qty]].dropna()
        if valid.empty:
            continue
        ax.plot(
            valid['step'], valid[qty],
            color=BATCH_COLORS[batch],
            label=f'B={batch}',
            linewidth=1.5,
            alpha=0.85,
        )

        # Cross-check: overlay BS from results.txt when quantity is 'bs'
        if qty == 'bs' and 'bs_results_df' in run_data.get((opt, batch, 'meta'), {}):
            bs_df = run_data[(opt, batch, 'meta')]['bs_results_df']
            ax.scatter(
                bs_df['step'], bs_df['bs_results'],
                color=BATCH_COLORS[batch],
                s=8, alpha=0.4, marker='x', zorder=5,
                label=f'B={batch} (results.txt)',
            )

    ax.set_xlabel('Training step', fontsize=12)
    ax.set_ylabel(QUANTITY_LABELS.get(qty, qty), fontsize=11)
    ax.set_title(f'{opt} — {qty}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Warn if any values are negative (sanity check)
    for batch in BATCH_SIZES:
        key = (opt, batch)
        if key not in run_data or run_data[key] is None:
            continue
        df = run_data[key]
        if qty in df.columns:
            neg_count = (df[qty].dropna() < 0).sum()
            if neg_count > 0 and qty not in ('cos_sg', 'cos_su', 'cos_gu'):
                print(f"  WARNING: {opt} B={batch} {qty} has {neg_count} negative values")

    out_path = out_dir / f'{opt}_{qty}.pdf'
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_all_batches_per_optimizer(opt: str, run_data: dict, out_dir: Path):
    """9-panel figure with all quantities for one optimizer."""
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12))
    fig.suptitle(f'GBS Suite — {opt}', fontsize=14)

    for ax, qty in zip(axes.flat, QUANTITIES):
        for batch in BATCH_SIZES:
            key = (opt, batch)
            if key not in run_data or run_data[key] is None:
                continue
            df = run_data[key]
            if qty not in df.columns:
                continue
            valid = df[['step', qty]].dropna()
            if valid.empty:
                continue
            ax.scatter(
                valid['step'], valid[qty],
                color=BATCH_COLORS[batch],
                label=f'B={batch}',
                s=4,
                alpha=0.85,
            )
        ax.set_title(qty, fontsize=10)
        ax.set_xlabel('step', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        if qty in ('gbs', 'gbs_u', 'gbs_ufull', 'gbs_g'):
            ax.set_ylim(0, 6)

    out_path = out_dir / f'{opt}_all_quantities.pdf'
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot GBS sweep results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to $RESULTS directory')
    parser.add_argument('--out-dir', type=str, default='figures/gbs_sweep',
                        help='Output directory for figures (default: figures/gbs_sweep/)')
    parser.add_argument('--no-summary', action='store_true',
                        help='Skip per-optimizer summary panels')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover runs ----
    print(f"Discovering runs in: {results_dir}/plaintext/cifar10_mlp/gbs_sweep/")
    runs = discover_runs(results_dir)
    print(f"Found {len(runs)} run(s).")
    if not runs:
        print("No runs found. Make sure RESULTS is set correctly and runs have completed.")
        return

    # ---- Load data ----
    run_data = {}   # (opt, batch) -> DataFrame or None
    for opt, batch, csv_path in runs:
        print(f"  Loading: {opt}  B={batch}  {csv_path}")
        try:
            df = load_suite_csv(csv_path)
            run_data[(opt, batch)] = df

            # Also try to load BS from results.txt for cross-check
            bs_df = load_results_bs(csv_path.parent)
            if bs_df is not None:
                run_data[(opt, batch, 'meta')] = {'bs_results_df': bs_df}
        except Exception as e:
            print(f"    ERROR loading {csv_path}: {e}")
            run_data[(opt, batch)] = None

    # ---- Identify which optimizers/batches are present ----
    present_opts = sorted(
        {key[0] for key in run_data if len(key) == 2 and isinstance(key[1], int)},
        key=lambda o: OPTIMIZERS.index(o) if o in OPTIMIZERS else 99,
    )

    print(f"\nOptimizers found: {present_opts}")

    # ---- Plot 45 individual figures (opt × quantity) ----
    print("\nGenerating per-quantity figures...")
    for opt in present_opts:
        for qty in QUANTITIES:
            plot_quantity(opt, qty, run_data, out_dir)

    # ---- Plot 5 summary panels (one per optimizer, all 9 quantities) ----
    if not args.no_summary:
        print("\nGenerating per-optimizer summary panels...")
        for opt in present_opts:
            plot_all_batches_per_optimizer(opt, run_data, out_dir)

    print(f"\nDone. Figures written to: {out_dir}")


if __name__ == '__main__':
    main()
