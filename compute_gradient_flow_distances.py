#!/usr/bin/env python3
"""
Compute distances between gradient flow (lr=0.001) and multiple gradient descent runs
with different step sizes, as in Figure 29.

For each step size η, computes distance between:
- Gradient flow iterate at time t (lr=0.001, step = t/0.001)
- Gradient descent iterate at iteration t/η (lr=η, step = t/η)

Where t = step_gf * 0.001 = step_gd * η
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Cannot load artifacts.", file=sys.stderr)

# Import functions from compute_trajectory_distance.py
sys.path.insert(0, str(Path(__file__).parent))
from compute_trajectory_distance import (
    get_available_steps,
    download_all_artifacts_to_cache,
    find_offline_run_directory
)


def compute_gradient_flow_distances(
    gradient_flow_run_id: str,
    gd_run_ids: Dict[float, str],  # Dict mapping lr -> run_id
    project: Optional[str] = None,
    entity: Optional[str] = None,
    wandb_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None
) -> Dict[float, pd.DataFrame]:
    """
    Compute distances between gradient flow and multiple GD runs.
    
    Parameters:
    -----------
    gradient_flow_run_id : str
        Run ID for gradient flow (lr=0.001)
    gd_run_ids : Dict[float, str]
        Dictionary mapping learning rate -> run ID for GD runs
    project : str, optional
        Wandb project name
    entity : str, optional
        Wandb entity
    wandb_dir : Path, optional
        Wandb directory
    cache_dir : Path, optional
        Cache directory for artifacts
        
    Returns:
    --------
    Dict[float, pd.DataFrame]
        Dictionary mapping lr -> DataFrame with columns: time, step_gf, step_gd, distance
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is not available")
    
    if project is None:
        project = os.environ.get("WANDB_PROJECT", "eoss2")
    
    gradient_flow_lr = 0.001
    
    # Get available steps for gradient flow
    print(f"Getting available steps for gradient flow run {gradient_flow_run_id}...")
    gf_steps = sorted(get_available_steps(gradient_flow_run_id, project, entity, wandb_dir))
    print(f"Found {len(gf_steps)} steps for gradient flow")
    
    results = {}
    
    for gd_lr, gd_run_id in gd_run_ids.items():
        print(f"\nProcessing GD run with lr={gd_lr} (run_id={gd_run_id})...")
        
        # Get available steps for GD run
        gd_steps = sorted(get_available_steps(gd_run_id, project, entity, wandb_dir))
        print(f"Found {len(gd_steps)} steps for GD run")
        
        # Match steps: step_gf * 0.001 = step_gd * gd_lr
        # So: step_gd = step_gf * (0.001 / gd_lr)
        ratio = gradient_flow_lr / gd_lr
        step_pairs = []
        gd_steps_set = set(gd_steps)
        
        for step_gf in gf_steps:
            step_gd = int(round(step_gf * ratio))
            if step_gd in gd_steps_set:
                step_pairs.append((step_gf, step_gd))
        
        print(f"Found {len(step_pairs)} matching step pairs (ratio={ratio:.6f})")
        
        if not step_pairs:
            print(f"Warning: No matching steps found for lr={gd_lr}")
            continue
        
        # Download and cache artifacts
        print("Downloading gradient flow artifacts...")
        gf_steps_needed = [s1 for s1, s2 in step_pairs]
        weights_gf = download_all_artifacts_to_cache(
            gradient_flow_run_id, gf_steps_needed, project, entity,
            cache_dir / gradient_flow_run_id if cache_dir else None
        )
        
        print("Downloading GD artifacts...")
        gd_steps_needed = [s2 for s1, s2 in step_pairs]
        weights_gd = download_all_artifacts_to_cache(
            gd_run_id, gd_steps_needed, project, entity,
            cache_dir / gd_run_id if cache_dir else None
        )
        
        # Compute distances
        distances = []
        times = []
        steps_gf = []
        steps_gd = []
        
        for step_gf, step_gd in step_pairs:
            if step_gf not in weights_gf or step_gd not in weights_gd:
                continue
            
            w_gf = weights_gf[step_gf]
            w_gd = weights_gd[step_gd]
            
            if w_gf.shape != w_gd.shape:
                print(f"Warning: Shape mismatch at steps ({step_gf}, {step_gd})")
                continue
            
            # Compute L2 distance
            distance = np.linalg.norm(w_gf - w_gd)
            time = step_gf * gradient_flow_lr  # time = step * eta
            
            distances.append(float(distance))
            times.append(time)
            steps_gf.append(step_gf)
            steps_gd.append(step_gd)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': times,
            'step_gf': steps_gf,
            'step_gd': steps_gd,
            'distance': distances
        })
        
        results[gd_lr] = df
        print(f"Computed {len(distances)} distances for lr={gd_lr}")
    
    return results


def find_sharpness_crossing_time(
    run_id: str,
    lr: float,
    results_root: Optional[Path] = None
) -> Optional[float]:
    """
    Find the time when sharpness first crosses 2/η.
    
    Returns:
    --------
    float or None
        Time when sharpness crosses 2/η, or None if not found
    """
    if results_root is None:
        results_root = Path(os.environ.get('RESULTS', '~/results')) / 'plaintext' / 'cifar10_mlp'
    
    results_root = Path(results_root).expanduser()
    
    threshold = 2.0 / lr
    
    # Search for results.txt files
    for folder in sorted(results_root.glob('*/')):
        results_file = folder / 'results.txt'
        if not results_file.exists():
            continue
        
        try:
            df = pd.read_csv(
                results_file,
                skiprows=4,
                sep=',',
                header=None,
                names=['epoch', 'step', 'batch_loss', 'full_loss', 'lambda_max',
                       'step_sharpness', 'batch_sharpness', 'gni', 'total_accuracy'],
                na_values=['nan'],
                skipinitialspace=True
            )
            
            # Find first step where lambda_max > threshold
            crossing = df[df['lambda_max'] > threshold]
            if len(crossing) > 0:
                first_crossing_step = crossing.iloc[0]['step']
                crossing_time = first_crossing_step * lr
                return float(crossing_time)
        except Exception as e:
            continue
    
    return None


def plot_gradient_flow_distances(
    results: Dict[float, pd.DataFrame],
    crossing_times: Dict[float, float],
    output_path: Optional[Path] = None
):
    """
    Plot distances for all step sizes on one graph, with vertical lines at crossing times.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (lr, df) in enumerate(sorted(results.items())):
        label = f'η = {lr:.5f}'.rstrip('0').rstrip('.')
        ax.plot(df['time'], df['distance'], 
               'o-', linewidth=2, markersize=3, color=colors[i], alpha=0.7,
               label=label)
        
        # Add vertical line at crossing time
        if lr in crossing_times and crossing_times[lr] is not None:
            crossing_time = crossing_times[lr]
            ax.axvline(x=crossing_time, color=colors[i], linestyle='--', 
                      linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Time (step * η)', fontsize=14)
    ax.set_ylabel('L2 Distance from Gradient Flow', fontsize=14)
    ax.set_title('Distance from Gradient Flow vs Time\n(Figure 29 analogue)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Compute and plot distances from gradient flow for multiple step sizes'
    )
    parser.add_argument('--gradient-flow-run-id', type=str, required=True,
                       help='Run ID for gradient flow (lr=0.001)')
    parser.add_argument('--gd-runs', type=str, nargs='+', required=True,
                       help='Run IDs for GD runs, in order: lr=0.02, 0.01, 0.00666666, 0.005')
    parser.add_argument('--project', type=str, default=None,
                       help='Wandb project (default: WANDB_PROJECT env var)')
    parser.add_argument('--entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--wandb-dir', type=str, default=None,
                       help='Wandb directory for offline runs')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory to cache artifacts')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory for finding sharpness crossing times')
    parser.add_argument('--output', type=str, default=None,
                       help='Output plot path')
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        print("Error: wandb is required", file=sys.stderr)
        return 1
    
    # Expected learning rates
    expected_lrs = [0.02, 0.01, 0.00666666, 0.005]
    
    if len(args.gd_runs) != len(expected_lrs):
        print(f"Error: Expected {len(expected_lrs)} GD run IDs, got {len(args.gd_runs)}", 
              file=sys.stderr)
        return 1
    
    gd_run_ids = {lr: run_id for lr, run_id in zip(expected_lrs, args.gd_runs)}
    
    wandb_dir = Path(args.wandb_dir).expanduser() if args.wandb_dir else None
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    
    print("Computing distances from gradient flow...")
    print(f"Gradient flow run: {args.gradient_flow_run_id}")
    print(f"GD runs: {gd_run_ids}")
    
    # Compute distances
    results = compute_gradient_flow_distances(
        gradient_flow_run_id=args.gradient_flow_run_id,
        gd_run_ids=gd_run_ids,
        project=args.project,
        entity=args.entity,
        wandb_dir=wandb_dir,
        cache_dir=cache_dir
    )
    
    if not results:
        print("Error: No distances computed", file=sys.stderr)
        return 1
    
    # Find crossing times
    print("\nFinding sharpness crossing times (2/η)...")
    crossing_times = {}
    results_root = Path(args.results_dir).expanduser() if args.results_dir else None
    
    for lr in expected_lrs:
        if lr in gd_run_ids:
            crossing_time = find_sharpness_crossing_time(
                gd_run_ids[lr], lr, results_root
            )
            crossing_times[lr] = crossing_time
            if crossing_time:
                print(f"lr={lr}: crossing time = {crossing_time:.4f}")
            else:
                print(f"lr={lr}: crossing time not found")
    
    # Plot
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'visualization' / 'img'
        output_path = output_dir / 'gradient_flow_distances_figure29.png'
    
    plot_gradient_flow_distances(results, crossing_times, output_path)
    
    # Save data
    csv_path = output_path.with_suffix('.csv')
    all_data = []
    for lr, df in results.items():
        df_copy = df.copy()
        df_copy['lr'] = lr
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

