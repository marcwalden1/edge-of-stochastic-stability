#!/usr/bin/env python3
"""
Compute all distance metrics for comparing two training runs.

Computes 8 distance types:
1. L2 weight distance (regular step-aligned)
2. L2 TRUE min distance (min distance to any step in run2)
3. Test prediction distance (Frobenius norm, step-aligned)
4. Test prediction TRUE min distance
5. L2 distance from init (Run1)
6. L2 distance from init (Run2)
7. Test prediction distance from init (Run1)
8. Test prediction distance from init (Run2)

Usage:
    python distance_plots/compute_comparison_distances.py \
        /path/to/run1_folder \
        /path/to/run2_folder \
        --output-dir output/comparison \
        --lr1 0.005 --lr2 0.0025 \
        --offline
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.compute_trajectory_distance import (
    compute_distances,
    compute_true_min_distances,
    compute_test_distances,
    compute_init_distance_single_run,
    find_run_directory,
    find_offline_run_directory,
)


def extract_run_id_from_path(run_path: Path) -> str:
    """Extract wandb run ID from a run folder path.

    Searches for wandb metadata or offline run directories to find the run ID.
    Falls back to using the folder name as an identifier.
    """
    run_path = Path(run_path)

    # Check for wandb-metadata.json in the folder or parent wandb dirs
    meta_paths = [
        run_path / "wandb-metadata.json",
        run_path / "files" / "wandb-metadata.json",
    ]

    for meta_path in meta_paths:
        if meta_path.exists():
            import json
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if "id" in meta:
                    return meta["id"]
            except (json.JSONDecodeError, KeyError):
                pass

    # Search for offline-run-* directories that might match
    # Check if there's a wandb directory with our run
    wandb_base = run_path.parent
    while wandb_base != wandb_base.parent:
        wandb_dir = wandb_base / "wandb"
        if wandb_dir.exists():
            # Look for offline run that might correspond to this folder
            for offline_dir in wandb_dir.glob("offline-run-*"):
                # Check if this offline run has matching folder name
                meta_path = offline_dir / "files" / "wandb-metadata.json"
                if meta_path.exists():
                    import json
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        # Return the run ID
                        return meta.get("id", offline_dir.name.split("-")[-1])
                    except (json.JSONDecodeError, KeyError):
                        pass
        wandb_base = wandb_base.parent

    # Fallback: use folder name as identifier
    return run_path.name


def merge_all_distances(
    df_l2: pd.DataFrame,
    df_l2_true: pd.DataFrame,
    df_test: pd.DataFrame,
    df_test_true: pd.DataFrame,
    df_l2_init_run1: pd.DataFrame,
    df_l2_init_run2: pd.DataFrame,
    df_test_init_run1: pd.DataFrame,
    df_test_init_run2: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all distance DataFrames on the step column."""
    # Start with L2 distance
    result = df_l2.copy()
    if 'weight_distance' in result.columns:
        result = result.rename(columns={'weight_distance': 'l2_distance'})

    # Merge L2 TRUE min
    if not df_l2_true.empty:
        df_l2_true = df_l2_true.rename(columns={'true_distance': 'l2_true_min'})
        result = result.merge(
            df_l2_true[['step', 'l2_true_min']],
            on='step',
            how='outer'
        )

    # Merge test distance
    if not df_test.empty:
        result = result.merge(
            df_test[['step', 'test_distance']],
            on='step',
            how='outer'
        )

    # Merge test TRUE min
    if not df_test_true.empty:
        df_test_true = df_test_true.rename(columns={'true_test_distance': 'test_true_min'})
        result = result.merge(
            df_test_true[['step', 'test_true_min']],
            on='step',
            how='outer'
        )

    # Merge L2 init distances
    if not df_l2_init_run1.empty:
        result = result.merge(
            df_l2_init_run1[['step', 'distance_from_init']].rename(
                columns={'distance_from_init': 'l2_init_run1'}
            ),
            on='step',
            how='outer'
        )

    if not df_l2_init_run2.empty:
        result = result.merge(
            df_l2_init_run2[['step', 'distance_from_init']].rename(
                columns={'distance_from_init': 'l2_init_run2'}
            ),
            on='step',
            how='outer'
        )

    # Merge test init distances
    if not df_test_init_run1.empty:
        result = result.merge(
            df_test_init_run1[['step', 'distance_from_init']].rename(
                columns={'distance_from_init': 'test_init_run1'}
            ),
            on='step',
            how='outer'
        )

    if not df_test_init_run2.empty:
        result = result.merge(
            df_test_init_run2[['step', 'distance_from_init']].rename(
                columns={'distance_from_init': 'test_init_run2'}
            ),
            on='step',
            how='outer'
        )

    return result.sort_values('step')


def main():
    parser = argparse.ArgumentParser(
        description='Compute all distance metrics for comparing two training runs'
    )
    parser.add_argument('run1_path', type=str, help='Path to first run folder')
    parser.add_argument('run2_path', type=str, help='Path to second run folder')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save output CSVs')
    parser.add_argument('--lr1', type=float, default=None,
                       help='Learning rate for run1 (for time alignment)')
    parser.add_argument('--lr2', type=float, default=None,
                       help='Learning rate for run2 (for time alignment)')
    parser.add_argument('--offline', action='store_true',
                       help='Force offline mode (skip wandb API)')
    parser.add_argument('--wandb-dir', type=str, default=None,
                       help='Wandb directory for offline runs')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory to cache artifacts')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory for finding run folders')

    args = parser.parse_args()

    # Resolve paths
    run1_path = Path(args.run1_path).expanduser().resolve()
    run2_path = Path(args.run2_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_dir = Path(args.wandb_dir).expanduser() if args.wandb_dir else None
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    results_dir = Path(args.results_dir).expanduser() if args.results_dir else None

    # Validate run paths
    if not run1_path.is_dir():
        print(f"Error: Run1 path does not exist: {run1_path}", file=sys.stderr)
        return 1
    if not run2_path.is_dir():
        print(f"Error: Run2 path does not exist: {run2_path}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Trajectory Comparison Distance Computation")
    print("=" * 60)
    print(f"Run 1: {run1_path}")
    print(f"Run 2: {run2_path}")
    print(f"Output: {output_dir}")
    if args.lr1 and args.lr2:
        print(f"Learning rates: lr1={args.lr1}, lr2={args.lr2}")
    print("=" * 60)

    # Extract run IDs for L2 weight distance computation
    run_id1 = extract_run_id_from_path(run1_path)
    run_id2 = extract_run_id_from_path(run2_path)
    print(f"Run ID 1: {run_id1}")
    print(f"Run ID 2: {run_id2}")

    # Initialize result DataFrames
    df_l2 = pd.DataFrame()
    df_l2_true = pd.DataFrame()
    df_test = pd.DataFrame()
    df_test_true = pd.DataFrame()
    df_l2_init_run1 = pd.DataFrame()
    df_l2_init_run2 = pd.DataFrame()
    df_test_init_run1 = pd.DataFrame()
    df_test_init_run2 = pd.DataFrame()

    # 1. L2 weight distance between runs (regular step-aligned)
    print("\n[1/8] Computing L2 weight distance (step-aligned)...")
    try:
        distances = compute_distances(
            run_id1, run_id2,
            lr1=args.lr1,
            lr2=args.lr2,
            wandb_dir=wandb_dir,
            cache_dir=cache_dir,
            offline=args.offline,
            output_file=output_dir / "l2_distance.csv",
        )
        if distances:
            steps = sorted(distances.keys())
            df_l2 = pd.DataFrame({
                'step': steps,
                'weight_distance': [distances[s] for s in steps]
            })
            df_l2.to_csv(output_dir / "l2_distance.csv", index=False)
            print(f"  Saved {len(df_l2)} rows to l2_distance.csv")
    except Exception as e:
        print(f"  Warning: L2 distance computation failed: {e}", file=sys.stderr)

    # 2. L2 TRUE min distance
    print("\n[2/8] Computing L2 TRUE min distance...")
    try:
        df_l2_true = compute_true_min_distances(
            run_id1, run_id2,
            wandb_dir=wandb_dir,
            cache_dir=cache_dir,
            offline=args.offline,
        )
        if not df_l2_true.empty:
            df_l2_true.to_csv(output_dir / "l2_true_min.csv", index=False)
            print(f"  Saved {len(df_l2_true)} rows to l2_true_min.csv")
    except Exception as e:
        print(f"  Warning: L2 TRUE min computation failed: {e}", file=sys.stderr)

    # 3. Test prediction distance (Frobenius norm)
    print("\n[3/8] Computing test prediction distance...")
    try:
        df_test = compute_test_distances(
            run1_path, run2_path,
            lr1=args.lr1,
            lr2=args.lr2,
        )
        if not df_test.empty:
            df_test.to_csv(output_dir / "test_distance.csv", index=False)
            print(f"  Saved {len(df_test)} rows to test_distance.csv")
    except Exception as e:
        print(f"  Warning: Test distance computation failed: {e}", file=sys.stderr)

    # 4. Test prediction TRUE min distance
    print("\n[4/8] Computing test prediction TRUE min distance...")
    try:
        df_test_true = compute_test_distances(
            run1_path, run2_path,
            true_min=True,
        )
        if not df_test_true.empty:
            df_test_true.to_csv(output_dir / "test_true_min.csv", index=False)
            print(f"  Saved {len(df_test_true)} rows to test_true_min.csv")
    except Exception as e:
        print(f"  Warning: Test TRUE min computation failed: {e}", file=sys.stderr)

    # 5. L2 distance from init (Run1)
    print("\n[5/8] Computing L2 distance from init (Run1)...")
    try:
        df_l2_init_run1 = compute_init_distance_single_run(
            run_id1,
            distance_type='l2',
            wandb_dir=wandb_dir,
            cache_dir=cache_dir,
            offline=args.offline,
        )
        if not df_l2_init_run1.empty:
            df_l2_init_run1.to_csv(output_dir / "l2_init_run1.csv", index=False)
            print(f"  Saved {len(df_l2_init_run1)} rows to l2_init_run1.csv")
    except Exception as e:
        print(f"  Warning: L2 init distance (Run1) failed: {e}", file=sys.stderr)

    # 6. L2 distance from init (Run2)
    print("\n[6/8] Computing L2 distance from init (Run2)...")
    try:
        df_l2_init_run2 = compute_init_distance_single_run(
            run_id2,
            distance_type='l2',
            wandb_dir=wandb_dir,
            cache_dir=cache_dir,
            offline=args.offline,
        )
        if not df_l2_init_run2.empty:
            df_l2_init_run2.to_csv(output_dir / "l2_init_run2.csv", index=False)
            print(f"  Saved {len(df_l2_init_run2)} rows to l2_init_run2.csv")
    except Exception as e:
        print(f"  Warning: L2 init distance (Run2) failed: {e}", file=sys.stderr)

    # 7. Test prediction distance from init (Run1)
    print("\n[7/8] Computing test prediction distance from init (Run1)...")
    try:
        df_test_init_run1 = compute_init_distance_single_run(
            str(run1_path),
            distance_type='test',
            results_dir=results_dir,
        )
        if not df_test_init_run1.empty:
            df_test_init_run1.to_csv(output_dir / "test_init_run1.csv", index=False)
            print(f"  Saved {len(df_test_init_run1)} rows to test_init_run1.csv")
    except Exception as e:
        print(f"  Warning: Test init distance (Run1) failed: {e}", file=sys.stderr)

    # 8. Test prediction distance from init (Run2)
    print("\n[8/8] Computing test prediction distance from init (Run2)...")
    try:
        df_test_init_run2 = compute_init_distance_single_run(
            str(run2_path),
            distance_type='test',
            results_dir=results_dir,
        )
        if not df_test_init_run2.empty:
            df_test_init_run2.to_csv(output_dir / "test_init_run2.csv", index=False)
            print(f"  Saved {len(df_test_init_run2)} rows to test_init_run2.csv")
    except Exception as e:
        print(f"  Warning: Test init distance (Run2) failed: {e}", file=sys.stderr)

    # Merge all distances into a single CSV
    print("\nMerging all distances...")
    all_distances = merge_all_distances(
        df_l2, df_l2_true, df_test, df_test_true,
        df_l2_init_run1, df_l2_init_run2,
        df_test_init_run1, df_test_init_run2,
    )

    if not all_distances.empty:
        all_distances.to_csv(output_dir / "all_distances.csv", index=False)
        print(f"Saved combined data to all_distances.csv ({len(all_distances)} rows)")
        print(f"Columns: {list(all_distances.columns)}")
    else:
        print("Warning: No distances were computed.", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("Distance computation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
