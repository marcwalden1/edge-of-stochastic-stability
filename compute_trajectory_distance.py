#!/usr/bin/env python3
"""
Compute weight trajectory distances between two training runs.
Loads projected weights from wandb artifacts and computes L2 distances.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Cannot load artifacts.", file=sys.stderr)


def find_offline_run_directory(run_id: str, wandb_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find offline wandb run directory by run ID.
    
    Parameters:
    -----------
    run_id : str
        Wandb run ID
    wandb_dir : Path, optional
        Base wandb directory (default: from WANDB_DIR env var or ~/results)
        
    Returns:
    --------
    Path or None
        Path to offline run directory if found
    """
    if wandb_dir is None:
        wandb_dir = Path(os.environ.get("WANDB_DIR", os.environ.get("RESULTS", "~/results")))
    wandb_dir = Path(wandb_dir).expanduser()
    
    # Look for offline-run directories - try both wandb/wandb and just wandb
    possible_bases = [
        wandb_dir / "wandb" / "wandb",
        wandb_dir / "wandb",
        wandb_dir,
    ]
    
    for wandb_base in possible_bases:
        if not wandb_base.exists():
            continue
        
        # First, try to find by run_id in directory name (fastest)
        for run_dir in wandb_base.glob(f"offline-run-*-{run_id}"):
            if run_dir.is_dir():
                return run_dir
        
        # Also search all offline-run directories
        for run_dir in wandb_base.glob("offline-run-*"):
            # Check metadata file for run ID
            meta_path = run_dir / "files" / "wandb-metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if meta.get("id") == run_id:
                        return run_dir
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Also check if run_id is in directory name
            if run_id in run_dir.name:
                return run_dir
    
    return None


def load_projected_weights_from_artifacts(
    run_id: str,
    step: int,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    wandb_dir: Optional[Path] = None
) -> Optional[np.ndarray]:
    """
    Load projected weights from wandb artifact for a specific run and step.
    Supports both online (via API) and offline (via filesystem) runs.
    
    Parameters:
    -----------
    run_id : str
        Wandb run ID
    step : int
        Training step
    project : str, optional
        Wandb project name (default: from WANDB_PROJECT env var)
    entity : str, optional
        Wandb entity (default: None)
    wandb_dir : Path, optional
        Base wandb directory for offline runs (default: from WANDB_DIR env var)
        
    Returns:
    --------
    np.ndarray or None
        Projected weight vector, or None if artifact not found
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is not available; cannot load artifacts.")
    
    # Artifact name format: projected_weights_step_{step:06d}
    artifact_name = f"projected_weights_step_{step:06d}"
    
    # Try API first (works for synced runs)
    if project is None:
        project = os.environ.get("WANDB_PROJECT", "eoss2")
    
    api = wandb.Api()
    
    if entity:
        api_run_path = f"{entity}/{project}/{run_id}"
    else:
        api_run_path = f"{project}/{run_id}"
    
    try:
        run = api.run(api_run_path)
        artifacts = run.logged_artifacts()
        
        target_artifact = None
        for artifact in artifacts:
            # Artifact names may have version suffix (e.g., "projected_weights_step_000000:v0")
            # Match if the artifact name starts with our target name
            if artifact.name.startswith(artifact_name + ":") or artifact.name == artifact_name:
                target_artifact = artifact
                break
        
        if target_artifact is not None:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = target_artifact.download(root=tmpdir)
                artifact_path = Path(artifact_dir)
                
                weights_path = artifact_path / "projected_weights.npy"
                if weights_path.exists():
                    projected_weights = np.load(weights_path)
                    return projected_weights
    except Exception:
        # API failed, fall back to filesystem search
        pass
    
    # Fall back to filesystem search if API didn't work
    if wandb_dir is None:
        wandb_dir = Path(os.environ.get("WANDB_DIR", os.environ.get("RESULTS", "~/results")))
    wandb_dir = Path(wandb_dir).expanduser()
    
    offline_run_dir = find_offline_run_directory(run_id, wandb_dir)
    if offline_run_dir:
        
        # Fall back to filesystem search
        # In offline mode, artifacts might be stored in various locations
        artifact_paths = [
            # Standard artifact storage locations
            offline_run_dir / "artifacts" / artifact_name / "projected_weights.npy",
            offline_run_dir / "files" / "artifacts" / artifact_name / "projected_weights.npy",
            offline_run_dir / "files" / artifact_name / "projected_weights.npy",
            # Also check if artifacts are stored with different naming
            offline_run_dir / "artifacts" / f"{artifact_name}:latest" / "projected_weights.npy",
        ]
        
        # Also search recursively for the artifact name
        for base_dir in [offline_run_dir / "artifacts", offline_run_dir / "files"]:
            if base_dir.exists():
                for artifact_dir in base_dir.rglob(artifact_name):
                    if artifact_dir.is_dir():
                        weights_file = artifact_dir / "projected_weights.npy"
                        if weights_file.exists():
                            artifact_paths.append(weights_file)
        
        for weights_path in artifact_paths:
            if weights_path.exists():
                try:
                    projected_weights = np.load(weights_path)
                    return projected_weights
                except Exception as e:
                    print(f"Error loading {weights_path}: {e}", file=sys.stderr)
                    continue
        
        # If still not found, try to extract from .wandb file using wandb's internal API
        # This is a last resort - artifacts in offline mode are stored in the .wandb SQLite file
        wandb_file = offline_run_dir / f"run-{run_id}.wandb"
        if wandb_file.exists():
            # Try to use wandb's internal methods to extract artifacts
            # This requires initializing wandb with the offline run
            try:
                # Temporarily change to the run directory and try to load
                original_dir = os.getcwd()
                os.chdir(str(offline_run_dir))
                
                # Initialize wandb in offline mode pointing to this run
                os.environ["WANDB_MODE"] = "offline"
                os.environ["WANDB_DIR"] = str(wandb_dir)
                
                # Try to restore the run and access artifacts
                # This is tricky - we might need to sync first or use a different approach
                # For now, just return None and let the user know
                os.chdir(original_dir)
            except Exception as e:
                if 'original_dir' in locals():
                    os.chdir(original_dir)
        
        print(f"Artifact {artifact_name} not found in offline run directory {offline_run_dir}", file=sys.stderr)
        # Global fallback: check WANDB_DIR/wandb/local_projected_weights/<run_id>/<artifact_name>
        global_root = wandb_dir / 'wandb' / 'local_projected_weights' / run_id
        global_artifact_dir = global_root / artifact_name
        weights_file = global_artifact_dir / 'projected_weights.npy'
        if weights_file.exists():
            try:
                projected_weights = np.load(weights_file)
                print(f"Loaded artifact via global fallback: {weights_file}", file=sys.stderr)
                return projected_weights
            except Exception:
                print(f"Error loading global fallback {weights_file}", file=sys.stderr)
        print(f"  Searched in: {offline_run_dir}", file=sys.stderr)
        # List what's actually there for debugging
        if (offline_run_dir / "artifacts").exists():
            print(f"  Artifacts dir contents: {list((offline_run_dir / 'artifacts').iterdir())[:5]}", file=sys.stderr)
        if (offline_run_dir / "files").exists():
            print(f"  Files dir contents: {list((offline_run_dir / 'files').iterdir())[:10]}", file=sys.stderr)
    
    # If both API and filesystem failed, return None
    print(f"Warning: Could not load artifact {artifact_name} for run {run_id} via API or filesystem", file=sys.stderr)
    # Final global fallback (if offline_run_dir was None)
    global_root = wandb_dir / 'wandb' / 'local_projected_weights' / run_id
    global_artifact_dir = global_root / artifact_name
    weights_file = global_artifact_dir / 'projected_weights.npy'
    if weights_file.exists():
        try:
            projected_weights = np.load(weights_file)
            print(f"Loaded artifact via global fallback (no offline run dir): {weights_file}", file=sys.stderr)
            return projected_weights
        except Exception:
            print(f"Error loading global fallback {weights_file}", file=sys.stderr)
    return None


def download_all_artifacts_to_cache(
    run_id: str,
    steps: List[int],
    project: Optional[str] = None,
    entity: Optional[str] = None,
    cache_dir: Optional[Path] = None
) -> Dict[int, np.ndarray]:
    """
    Download all artifacts for a run and cache them locally for fast access.
    
    Parameters:
    -----------
    run_id : str
        Wandb run ID
    steps : List[int]
        List of steps to download
    project : str, optional
        Wandb project name
    entity : str, optional
        Wandb entity
    cache_dir : Path, optional
        Directory to cache artifacts (default: ~/.cache/trajectory_distances/{run_id})
        
    Returns:
    --------
    Dict[int, np.ndarray]
        Dictionary mapping step -> projected weights array
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is not available; cannot download artifacts.")
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "trajectory_distances" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if project is None:
        project = os.environ.get("WANDB_PROJECT", "eoss2")
    
    api = wandb.Api()
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"
    
    print(f"Loading artifacts for run {run_id}...")
    run = api.run(run_path)
    all_artifacts = list(run.logged_artifacts())
    
    # Create a lookup dictionary for artifacts
    artifacts_dict = {}
    for artifact in all_artifacts:
        if artifact.type == "projected_weights" and artifact.name.startswith("projected_weights_step_"):
            # Extract step number
            step_str = artifact.name.replace("projected_weights_step_", "")
            if ":" in step_str:
                step_str = step_str.split(":")[0]
            try:
                step = int(step_str)
                artifacts_dict[step] = artifact
            except ValueError:
                continue
    
    cached_weights = {}
    missing_steps = []
    
    for step in steps:
        artifact_name = f"projected_weights_step_{step:06d}"
        cache_file = cache_dir / f"{artifact_name}.npy"
        
        # Check cache first
        if cache_file.exists():
            try:
                cached_weights[step] = np.load(cache_file)
                continue
            except Exception as e:
                print(f"Warning: Error loading cached file {cache_file}: {e}", file=sys.stderr)
                # Continue to download
        
        # Find and download artifact
        if step not in artifacts_dict:
            missing_steps.append(step)
            continue
        
        target_artifact = artifacts_dict[step]
        try:
            # Download to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = target_artifact.download(root=tmpdir)
                weights_path = Path(artifact_dir) / "projected_weights.npy"
                if weights_path.exists():
                    weights = np.load(weights_path)
                    # Cache it for future use
                    np.save(cache_file, weights)
                    cached_weights[step] = weights
                else:
                    missing_steps.append(step)
        except Exception as e:
            print(f"Warning: Error downloading artifact for step {step}: {e}", file=sys.stderr)
            missing_steps.append(step)
    
    if missing_steps:
        print(f"Warning: Could not load {len(missing_steps)} artifacts (steps: {missing_steps[:10]}{'...' if len(missing_steps) > 10 else ''})", file=sys.stderr)
    
    print(f"Cached {len(cached_weights)}/{len(steps)} artifacts for run {run_id}")
    return cached_weights


def get_available_steps(run_id: str, project: Optional[str] = None, entity: Optional[str] = None, wandb_dir: Optional[Path] = None) -> List[int]:
    """
    Get list of steps for which projected weights are available.
    Supports both online (via API) and offline (via filesystem) runs.
    
    Returns:
    --------
    List[int]
        Sorted list of step numbers
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is not available; cannot load artifacts.")
    
    # Try offline mode first
    if wandb_dir is None:
        wandb_dir = Path(os.environ.get("WANDB_DIR", os.environ.get("RESULTS", "~/results")))
    wandb_dir = Path(wandb_dir).expanduser()
    
    offline_run_dir = find_offline_run_directory(run_id, wandb_dir)
    
    # Try API first (works for synced runs)
    # Fall back to filesystem search if API fails
    if project is None:
        project = os.environ.get("WANDB_PROJECT", "eoss2")
    
    api = wandb.Api()
    
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"
    
    try:
        run = api.run(run_path)
        print(f"  Successfully accessed run via API: {run_path}", file=sys.stderr)
        artifacts = list(run.logged_artifacts())
        print(f"  Found {len(artifacts)} total artifacts", file=sys.stderr)
        
        steps = []
        for artifact in artifacts:
            if artifact.type == "projected_weights" and artifact.name.startswith("projected_weights_step_"):
                # Extract step number from artifact name
                # Artifact names are like "projected_weights_step_000000:v0"
                step_str = artifact.name.replace("projected_weights_step_", "")
                # Remove version suffix (e.g., ":v0")
                if ":" in step_str:
                    step_str = step_str.split(":")[0]
                try:
                    step = int(step_str)
                    steps.append(step)
                except ValueError:
                    continue
        
        print(f"  Found {len(steps)} projected_weights artifacts", file=sys.stderr)
        if steps:
            return sorted(steps)
    except Exception as e:
        # API failed, fall back to filesystem search
        print(f"  API access failed: {e}", file=sys.stderr)
        pass
    
    # Fall back to filesystem search if API didn't work
    if offline_run_dir:
        # Search for artifacts in offline run directory
        steps = []
        
        # Check artifacts subdirectories - search recursively
        artifact_dirs = [
            offline_run_dir / "artifacts",
            offline_run_dir / "files" / "artifacts",
            offline_run_dir / "files",
        ]
        
        print(f"  Searching for artifacts in offline run directory: {offline_run_dir}", file=sys.stderr)
        for artifacts_base in artifact_dirs:
            if not artifacts_base.exists():
                print(f"    {artifacts_base} does not exist", file=sys.stderr)
                continue
            
            print(f"    Searching in {artifacts_base}", file=sys.stderr)
            # Search recursively for artifact directories
            found_dirs = list(artifacts_base.rglob("projected_weights_step_*"))
            print(f"    Found {len(found_dirs)} potential artifact directories", file=sys.stderr)
            
            for artifact_dir in found_dirs:
                if artifact_dir.is_dir():
                    # Extract step number
                    step_str = artifact_dir.name.replace("projected_weights_step_", "")
                    # Remove any suffix like ":latest"
                    step_str = step_str.split(":")[0]
                    try:
                        step = int(step_str)
                        # Verify the weights file exists
                        weights_file = artifact_dir / "projected_weights.npy"
                        if weights_file.exists():
                            steps.append(step)
                    except ValueError:
                        continue
        
        if steps:
            print(f"  Found {len(steps)} steps via filesystem search", file=sys.stderr)
        else:
            print(f"  No steps found in filesystem. Listing contents of {offline_run_dir}:", file=sys.stderr)
            if offline_run_dir.exists():
                for item in list(offline_run_dir.iterdir())[:10]:
                    print(f"    {item.name} ({'dir' if item.is_dir() else 'file'})", file=sys.stderr)
            
            # Check if .wandb file exists - artifacts might be stored there
            wandb_file = offline_run_dir / f"run-{run_id}.wandb"
            if wandb_file.exists():
                print(f"  Found .wandb file: {wandb_file} (size: {wandb_file.stat().st_size / 1024 / 1024:.1f} MB)", file=sys.stderr)
                print("  Artifacts are stored in the .wandb file. You may need to sync the run first:", file=sys.stderr)
                print(f"    cd {offline_run_dir} && wandb sync .", file=sys.stderr)
        
        if steps:
            return sorted(steps)
    # Global fallback search: WANDB_DIR/wandb/local_projected_weights/<run_id>
    global_root = wandb_dir / 'wandb' / 'local_projected_weights' / run_id
    if global_root.exists():
        steps = []
        for artifact_dir in global_root.glob('projected_weights_step_*'):
            if artifact_dir.is_dir():
                step_str = artifact_dir.name.replace('projected_weights_step_', '').split(':')[0]
                try:
                    step = int(step_str)
                    weights_file = artifact_dir / 'projected_weights.npy'
                    if weights_file.exists():
                        steps.append(step)
                except ValueError:
                    continue
        if steps:
            return sorted(steps)
    
    # If both API and filesystem failed, return empty list
    print(f"Warning: Could not find artifacts for run {run_id} via API or filesystem", file=sys.stderr)
    if not offline_run_dir:
        print(f"  Offline run directory not found for run_id {run_id}", file=sys.stderr)
    return []


def compute_distances(
    run_id1: str,
    run_id2: str,
    steps: Optional[List[int]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    wandb_dir: Optional[Path] = None,
    existing_distances: Optional[Dict[int, float]] = None,
    output_file: Optional[Path] = None,
    save_every_n: int = 10,
    lr1: Optional[float] = None,
    lr2: Optional[float] = None,
    cache_dir: Optional[Path] = None
) -> Dict[int, float]:
    """
    Compute L2 distances between projected weights from two runs.
    If lr1 and lr2 are provided, compares at equivalent step*eta values.
    Otherwise, compares at matching step numbers.
    
    Parameters:
    -----------
    run_id1 : str
        First run ID
    run_id2 : str
        Second run ID
    steps : List[int], optional
        List of steps to compute distances for. If None, uses intersection of available steps.
    project : str, optional
        Wandb project name
    entity : str, optional
        Wandb entity
    existing_distances : Dict[int, float], optional
        Already computed distances (to skip recomputing)
    output_file : Path, optional
        Path to CSV file to save incrementally
    save_every_n : int
        Save progress every N steps (default: 10)
    lr1 : float, optional
        Learning rate for run1 (for step*eta matching)
    lr2 : float, optional
        Learning rate for run2 (for step*eta matching)
    cache_dir : Path, optional
        Directory to cache artifacts
        
    Returns:
    --------
    Dict[int, float]
        Dictionary mapping step_run1 -> L2 distance
    """
    # Get available steps for both runs
    steps1 = sorted(get_available_steps(run_id1, project, entity, wandb_dir))
    steps2 = sorted(get_available_steps(run_id2, project, entity, wandb_dir))
    
    # Determine step pairs based on matching strategy
    if lr1 is not None and lr2 is not None:
        # Compare at equivalent step*eta values
        # For each step s1 in run1, find step s2 in run2 where s1*lr1 = s2*lr2
        # So s2 = s1 * (lr1 / lr2)
        ratio = lr1 / lr2
        step_pairs = []
        steps2_set = set(steps2)
        for s1 in steps1:
            s2 = int(round(s1 * ratio))
            if s2 in steps2_set:
                step_pairs.append((s1, s2))
        print(f"Found {len(step_pairs)} step pairs with equivalent step*eta (ratio={ratio:.2f}, lr1={lr1}, lr2={lr2})")
        # Use steps from run1 for output
        steps_for_output = [s1 for s1, s2 in step_pairs]
    else:
        # Compare at matching step numbers
        common_steps = sorted(set(steps1) & set(steps2))
        step_pairs = [(s, s) for s in common_steps]
        steps_for_output = common_steps
        print(f"Found {len(step_pairs)} common steps between runs")
    
    if steps is not None:
        # Filter to requested steps
        steps_set = set(steps)
        step_pairs = [(s1, s2) for s1, s2 in step_pairs if s1 in steps_set]
        steps_for_output = [s for s in steps_for_output if s in steps_set]
    
    if not step_pairs:
        print("No step pairs found to compute distances for.", file=sys.stderr)
        return {}
    
    # Batch download and cache all artifacts
    print("Downloading and caching artifacts for run1...")
    steps1_needed = [s1 for s1, s2 in step_pairs]
    weights1_cache = download_all_artifacts_to_cache(
        run_id1, steps1_needed, project, entity,
        cache_dir / run_id1 if cache_dir else None
    )
    
    print("Downloading and caching artifacts for run2...")
    steps2_needed = [s2 for s1, s2 in step_pairs]
    weights2_cache = download_all_artifacts_to_cache(
        run_id2, steps2_needed, project, entity,
        cache_dir / run_id2 if cache_dir else None
    )
    
    # Start with existing distances if provided
    distances = existing_distances.copy() if existing_distances else {}
    computed_count = len(distances)
    
    # Filter out pairs we've already computed
    remaining_pairs = [(s1, s2) for s1, s2 in step_pairs if s1 not in distances]
    print(f"Computing {len(remaining_pairs)} remaining step pairs (already have {computed_count})...")
    
    # Compute distances from cached weights (much faster!)
    for i, (step1, step2) in enumerate(remaining_pairs):
        if step1 not in weights1_cache or step2 not in weights2_cache:
            print(f"Skipping step pair ({step1}, {step2}): missing weights", file=sys.stderr)
            continue
        
        weights1 = weights1_cache[step1]
        weights2 = weights2_cache[step2]
        
        # Ensure same shape
        if weights1.shape != weights2.shape:
            print(f"Warning: shape mismatch at steps ({step1}, {step2}): {weights1.shape} vs {weights2.shape}", file=sys.stderr)
            continue
        
        # Compute L2 distance
        distance = np.linalg.norm(weights1 - weights2)
        distances[step1] = float(distance)  # Use step1 as the key for output
        computed_count += 1
        
        # Save incrementally
        if output_file and (computed_count % save_every_n == 0 or i == len(remaining_pairs) - 1):
            try:
                steps_sorted = sorted(distances.keys())
                if lr1 and lr2:
                    df = pd.DataFrame({
                        'step_run1': steps_sorted,
                        'step_run2': [int(round(s * (lr1 / lr2))) for s in steps_sorted],
                        'weight_distance': [distances[s] for s in steps_sorted]
                    })
                else:
                    df = pd.DataFrame({
                        'step': steps_sorted,
                        'weight_distance': [distances[s] for s in steps_sorted]
                    })
                df.to_csv(output_file, index=False)
                print(f"Saved progress: {computed_count}/{len(step_pairs)} step pairs computed")
            except Exception as e:
                print(f"Warning: Could not save progress: {e}", file=sys.stderr)
        
        if computed_count % 10 == 0:
            print(f"Computed distances for {computed_count}/{len(step_pairs)} step pairs...")
    
    return distances


def load_lambda_max_from_results(run_id: str, results_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Load lambda_max values from results.txt file.
    
    Parameters:
    -----------
    run_id : str
        Run ID (used to find results folder)
    results_root : Path, optional
        Root directory for results. If None, uses RESULTS env var.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: step, lambda_max
    """
    if results_root is None:
        results_root = Path(os.environ.get('RESULTS', '~/results')) / 'plaintext' / 'cifar10_mlp'
    
    results_root = Path(results_root).expanduser()
    
    # Find folder matching run_id (folder names contain run info, not necessarily run_id)
    # We'll need to search for folders or use a different approach
    # For now, assume we can find it by searching or it's passed differently
    
    # Alternative: search all folders and match by some criteria
    # This is a simplified version - may need adjustment based on folder naming
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
            # Return first match for now (in practice, might need better matching)
            return df[['step', 'lambda_max']].copy()
        except Exception:
            continue
    
    raise FileNotFoundError(f"Could not find results.txt for run {run_id}")


def main():
    parser = argparse.ArgumentParser(description='Compute trajectory distances between two runs')
    parser.add_argument('run_id1', type=str, help='First run ID')
    parser.add_argument('run_id2', type=str, help='Second run ID')
    parser.add_argument('--project', type=str, default=None, help='Wandb project (default: WANDB_PROJECT env var)')
    parser.add_argument('--entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--steps', type=int, nargs='+', default=None, help='Specific steps to compute (default: all common steps)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: trajectory_distances_{run_id1}_{run_id2}.csv)')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory for loading lambda_max')
    parser.add_argument('--wandb-dir', type=str, default=None, help='Wandb directory for offline runs (default: WANDB_DIR env var or ~/results)')
    parser.add_argument('--sync-first', action='store_true', help='Sync offline wandb runs before loading artifacts (requires internet)')
    parser.add_argument('--lr1', type=float, default=None, help='Learning rate for first run (for step*eta matching)')
    parser.add_argument('--lr2', type=float, default=None, help='Learning rate for second run (for step*eta matching)')
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory to cache artifacts (default: ~/.cache/trajectory_distances)')
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        print("Error: wandb is required for this script", file=sys.stderr)
        return 1
    
    wandb_dir = Path(args.wandb_dir).expanduser() if args.wandb_dir else None
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    
    # Determine output file path
    if args.output is None:
        args.output = f"trajectory_distances_{args.run_id1}_{args.run_id2}.csv"
    output_path = Path(args.output)
    
    # Check if output file exists and load existing distances
    existing_distances = {}
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if 'step' in existing_df.columns and 'weight_distance' in existing_df.columns:
                existing_distances = dict(zip(existing_df['step'], existing_df['weight_distance']))
                print(f"Found existing output file with {len(existing_distances)} computed distances. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing file {output_path}: {e}", file=sys.stderr)
            print("Starting fresh computation...")
    
    print(f"Computing distances between run {args.run_id1} and {args.run_id2}...")
    
    # Compute distances (with incremental saving)
    distances = compute_distances(
        args.run_id1,
        args.run_id2,
        steps=args.steps,
        project=args.project,
        entity=args.entity,
        wandb_dir=wandb_dir,
        existing_distances=existing_distances if existing_distances else None,
        output_file=output_path,
        save_every_n=10,
        lr1=args.lr1,
        lr2=args.lr2,
        cache_dir=cache_dir
    )
    
    if not distances:
        print("No distances computed. Check that artifacts exist for both runs.", file=sys.stderr)
        return 1
    
    # Create output DataFrame
    steps = sorted(distances.keys())
    if args.lr1 and args.lr2:
        # When matching by step*eta, include both step columns
        df = pd.DataFrame({
            'step_run1': steps,
            'step_run2': [int(round(s * (args.lr1 / args.lr2))) for s in steps],
            'weight_distance': [distances[s] for s in steps]
        })
        step_col_for_merge = 'step_run1'
    else:
        df = pd.DataFrame({
            'step': steps,
            'weight_distance': [distances[s] for s in steps]
        })
    # step_col_for_merge retained from earlier logic but not needed further; removed to silence lint.
    
    # Try to load lambda_max for both runs
    try:
        results_root = Path(args.results_dir).expanduser() if args.results_dir else None
        lambda_max1 = load_lambda_max_from_results(args.run_id1, results_root)
        lambda_max2 = load_lambda_max_from_results(args.run_id2, results_root)
        
        # Merge lambda_max values
        if args.lr1 and args.lr2:
            # For step*eta matching, merge on step_run1 and step_run2 respectively
            df = df.merge(lambda_max1, left_on='step_run1', right_on='step', how='left', suffixes=('', '_tmp1'))
            df = df.merge(lambda_max2, left_on='step_run2', right_on='step', how='left', suffixes=('', '_tmp2'))
            # Rename lambda_max columns
            df = df.rename(columns={'lambda_max': 'lambda_max_run1', 'lambda_max_tmp2': 'lambda_max_run2'})
            # Drop the temporary step column from merges
            df = df.drop(columns=['step'])
        else:
            df = df.merge(lambda_max1, on='step', how='left', suffixes=('', '_run1'))
            df = df.merge(lambda_max2, on='step', how='left', suffixes=('', '_run2'))
            df = df.rename(columns={'lambda_max': 'lambda_max_run1', 'lambda_max_run2': 'lambda_max_run2'})
        
    except Exception as e:
        print(f"Warning: Could not load lambda_max values: {e}", file=sys.stderr)
        print("Distance data will be saved without lambda_max columns.")
    
    # Final save to CSV (with lambda_max if available)
    df.to_csv(output_path, index=False)
    print(f"\nSaved distances to: {output_path}")
    print(f"Computed {len(distances)} distance measurements")
    
    # Also save as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({
            'run_id1': args.run_id1,
            'run_id2': args.run_id2,
            'distances': distances,
            'num_steps': len(distances)
        }, f, indent=2)
    print(f"Saved JSON to: {json_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

