#!/usr/bin/env python3
"""
Compute weight trajectory distances between two training runs.
Loads projected weights from wandb artifacts and computes L2 distances.
Also supports test prediction distances (Frobenius norm).
"""

import os
import sys
import json
import re
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from bisect import bisect_left

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Cannot load artifacts.", file=sys.stderr)


# =============================================================================
# Test Prediction Distance Helpers
# =============================================================================

def find_run_directory(run_id_or_path: str, results_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find the results directory for a run by searching common locations.

    Args:
        run_id_or_path: Either a wandb run ID or a direct path to the run folder

    Searches in:
    1. Direct path (if run_id_or_path is a valid directory)
    2. results_dir/plaintext/*/{folder containing run_id}
    3. results_dir/*/{folder containing run_id}
    4. Wandb offline directories
    """
    # Check if it's a direct path
    direct_path = Path(run_id_or_path).expanduser()
    if direct_path.is_dir() and (direct_path / "test_predictions.npz").exists():
        return direct_path

    if results_dir is None:
        results_dir = Path(os.environ.get("RESULTS", "."))
    results_dir = Path(results_dir).expanduser()

    # Search in plaintext directories for folders containing run_id
    for search_base in [results_dir / "plaintext", results_dir]:
        if not search_base.exists():
            continue
        for subdir in search_base.iterdir():
            if subdir.is_dir():
                # Check direct match
                candidate = subdir / run_id_or_path
                if candidate.is_dir() and (candidate / "test_predictions.npz").exists():
                    return candidate
                # Check if run_id is contained in any folder name
                for folder in subdir.iterdir():
                    if folder.is_dir() and run_id_or_path in folder.name:
                        if (folder / "test_predictions.npz").exists():
                            return folder
                # Also check subdir itself if it matches
                if run_id_or_path in subdir.name and (subdir / "test_predictions.npz").exists():
                    return subdir

    # Check wandb offline directories - they may have test_predictions.npz
    wandb_dir = Path(os.environ.get("WANDB_DIR", results_dir))
    for wandb_base in [wandb_dir / "wandb", wandb_dir]:
        if not wandb_base.exists():
            continue
        for run_dir in wandb_base.glob(f"offline-run-*-{run_id_or_path}"):
            files_dir = run_dir / "files"
            if files_dir.is_dir() and (files_dir / "test_predictions.npz").exists():
                return files_dir
            if (run_dir / "test_predictions.npz").exists():
                return run_dir

    return None


def load_test_predictions(run_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load test predictions from .npz file.

    Args:
        run_path: Path to the run folder containing test_predictions.npz

    Returns:
        steps: array of shape (num_snapshots,) with training steps
        predictions: array of shape (num_snapshots, test_set_size, num_classes)
    """
    npz_path = run_path / 'test_predictions.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"No test_predictions.npz found in {run_path}")

    data = np.load(npz_path)
    return data['steps'], data['predictions']


def extract_lr_from_results(run_path: Path) -> float:
    """Extract learning rate from results.txt header."""
    results_file = run_path / 'results.txt'
    if not results_file.exists():
        raise FileNotFoundError(f"No results.txt found in {run_path}")

    with open(results_file, 'r') as f:
        for line in f:
            if 'Arguments:' in line or 'Namespace(' in line:
                match = re.search(r'\blr[=:\s]+([\d.eE+-]+)', line)
                if match:
                    return float(match.group(1))
                break
    raise ValueError(f"Could not extract learning rate from {results_file}")


def compute_test_distances(
    run_path1: Path,
    run_path2: Path,
    lr1: Optional[float] = None,
    lr2: Optional[float] = None,
    include_init_distance: bool = False,
    true_min: bool = False,
) -> pd.DataFrame:
    """
    Compute Frobenius norm distances between test predictions of two runs.

    Args:
        run_path1: Path to first run folder
        run_path2: Path to second run folder
        lr1: Learning rate for run1 (for time alignment, auto-extracted if None)
        lr2: Learning rate for run2 (for time alignment, auto-extracted if None)
        include_init_distance: If True, also compute distance from init predictions
        true_min: If True, compute TRUE distance (min distance to any step in run2)

    Returns:
        DataFrame with columns: step, test_distance, [distance_run1_init, distance_run2_init]
    """
    # Load predictions
    print(f"Loading test predictions from {run_path1}...")
    steps1, preds1 = load_test_predictions(run_path1)
    print(f"  Found {len(steps1)} snapshots, shape {preds1.shape}")

    print(f"Loading test predictions from {run_path2}...")
    steps2, preds2 = load_test_predictions(run_path2)
    print(f"  Found {len(steps2)} snapshots, shape {preds2.shape}")

    # Create step-to-index mappings
    step_to_idx1 = {int(step): idx for idx, step in enumerate(steps1)}
    step_to_idx2 = {int(step): idx for idx, step in enumerate(steps2)}
    steps2_list = sorted(step_to_idx2.keys())

    # Determine alignment strategy (only used if not true_min)
    use_time_alignment = lr1 is not None and lr2 is not None and not true_min

    if use_time_alignment:
        ratio = lr1 / lr2
        print(f"Using time alignment: lr1={lr1}, lr2={lr2}, ratio={ratio:.4f}")

    if true_min:
        print("Computing TRUE test distance (min Frobenius distance to any step in run2)...")

    # Find step pairs to compare
    results = []
    steps2_set = set(step_to_idx2.keys())

    # Get init predictions if needed
    init_pred1 = None
    init_pred2 = None
    if include_init_distance:
        init_step1 = min(step_to_idx1.keys())
        init_step2 = min(step_to_idx2.keys())
        init_pred1 = preds1[step_to_idx1[init_step1]]
        init_pred2 = preds2[step_to_idx2[init_step2]]
        print(f"Using step {init_step1} as init for run1, step {init_step2} as init for run2")

    print("Computing Frobenius distances...")
    sorted_steps1 = sorted(step_to_idx1.keys())
    for i, step1 in enumerate(sorted_steps1):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing step {i+1}/{len(sorted_steps1)} (step {step1})...")

        idx1 = step_to_idx1[step1]
        p1 = preds1[idx1]  # (test_size, num_classes)

        if true_min:
            # Compute distance to ALL steps in run2 and find minimum
            min_dist = float('inf')
            closest_step2 = None
            closest_idx2 = None

            for step2 in steps2_list:
                idx2 = step_to_idx2[step2]
                p2 = preds2[idx2]
                dist = np.linalg.norm(p1 - p2, ord='fro')
                if dist < min_dist:
                    min_dist = dist
                    closest_step2 = step2
                    closest_idx2 = idx2

            row = {
                'step': step1,
                'true_test_distance': float(min_dist),
                'closest_step_run2': closest_step2,
            }

            if include_init_distance:
                row['distance_run1_init'] = float(np.linalg.norm(p1 - init_pred1, ord='fro'))
                p2_closest = preds2[closest_idx2]
                row['distance_run2_init'] = float(np.linalg.norm(p2_closest - init_pred2, ord='fro'))

            results.append(row)
        else:
            # Regular step-aligned or time-aligned comparison
            if use_time_alignment:
                step2 = int(round(step1 * ratio))
            else:
                step2 = step1

            if step2 not in steps2_set:
                continue

            idx2 = step_to_idx2[step2]
            p2 = preds2[idx2]  # (test_size, num_classes)

            # Frobenius norm of the difference
            test_dist = np.linalg.norm(p1 - p2, ord='fro')

            row = {
                'step': step1,
                'test_distance': float(test_dist),
            }

            if use_time_alignment:
                row['step_run2'] = step2

            if include_init_distance:
                row['distance_run1_init'] = float(np.linalg.norm(p1 - init_pred1, ord='fro'))
                row['distance_run2_init'] = float(np.linalg.norm(p2 - init_pred2, ord='fro'))

            results.append(row)

    if not results:
        print("Warning: No matching steps found between runs", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values('step')
    print(f"Computed distances for {len(df)} step pairs")
    return df


# =============================================================================
# Projected Weights Distance Helpers
# =============================================================================

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
    wandb_dir: Optional[Path] = None,
    offline: bool = False,
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
    # We allow offline operation without wandb; only API access requires wandb.
    # If wandb is missing or offline flag set, we'll skip API access and rely on filesystem.
    use_api = WANDB_AVAILABLE and (not offline)
    
    # Artifact name format: projected_weights_step_{step:06d}
    artifact_name = f"projected_weights_step_{step:06d}"
    
    if use_api:
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
            except Exception:
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
    cache_dir: Optional[Path] = None,
    wandb_dir: Optional[Path] = None,
    offline: bool = False,
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
    # If wandb isn't available or offline mode, we'll skip API and load via filesystem.
    use_api = WANDB_AVAILABLE and (not offline)
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "trajectory_distances" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts_dict = {}
    if use_api:
        if project is None:
            project = os.environ.get("WANDB_PROJECT", "eoss2")
        api = wandb.Api()
        if entity:
            run_path = f"{entity}/{project}/{run_id}"
        else:
            run_path = f"{project}/{run_id}"
        print(f"Loading artifacts for run {run_id} via API...")
        try:
            run = api.run(run_path)
            all_artifacts = list(run.logged_artifacts())
            for artifact in all_artifacts:
                if artifact.type == "projected_weights" and artifact.name.startswith("projected_weights_step_"):
                    step_str = artifact.name.replace("projected_weights_step_", "")
                    if ":" in step_str:
                        step_str = step_str.split(":")[0]
                    try:
                        step = int(step_str)
                        artifacts_dict[step] = artifact
                    except ValueError:
                        continue
        except Exception:
            print(f"API access failed for run {run_id}. Falling back to offline filesystem.", file=sys.stderr)
            use_api = False
    else:
        print(f"Offline mode: loading artifacts for run {run_id} from filesystem only.")
    
    cached_weights = {}
    missing_steps = []
    total_steps = len(steps)

    for i, step in enumerate(steps):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Loading artifact {i+1}/{total_steps} (step {step})...")
        artifact_name = f"projected_weights_step_{step:06d}"
        cache_file = cache_dir / f"{artifact_name}.npy"
        # Cache first
        if cache_file.exists():
            try:
                cached_weights[step] = np.load(cache_file)
                continue
            except Exception as e:
                print(f"Warning: Error loading cached file {cache_file}: {e}", file=sys.stderr)
        if use_api and step in artifacts_dict:
            target_artifact = artifacts_dict[step]
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    artifact_dir = target_artifact.download(root=tmpdir)
                    weights_path = Path(artifact_dir) / "projected_weights.npy"
                    if weights_path.exists():
                        weights = np.load(weights_path)
                        np.save(cache_file, weights)
                        cached_weights[step] = weights
                        continue
            except Exception as e:
                print(f"Warning: Error downloading artifact for step {step}: {e}", file=sys.stderr)
        # Offline fallback: load directly from filesystem
        weights = load_projected_weights_from_artifacts(
            run_id, step, project=project, entity=entity, wandb_dir=wandb_dir, offline=(not use_api)
        )
        if weights is not None:
            try:
                np.save(cache_file, weights)
            except Exception:
                pass
            cached_weights[step] = weights
        else:
            missing_steps.append(step)
    
    if missing_steps:
        print(f"Warning: Could not load {len(missing_steps)} artifacts (steps: {missing_steps[:10]}{'...' if len(missing_steps) > 10 else ''})", file=sys.stderr)
    
    print(f"Cached {len(cached_weights)}/{len(steps)} artifacts for run {run_id}")
    return cached_weights


def get_available_steps(run_id: str, project: Optional[str] = None, entity: Optional[str] = None, wandb_dir: Optional[Path] = None, offline: bool = False) -> List[int]:
    """
    Get list of steps for which projected weights are available.
    Supports both online (via API) and offline (via filesystem) runs.
    
    Returns:
    --------
    List[int]
        Sorted list of step numbers
    """
    # Allow offline mode without wandb; skip API if offline or wandb missing
    use_api = WANDB_AVAILABLE and (not offline)
    
    # Try offline mode first
    if wandb_dir is None:
        wandb_dir = Path(os.environ.get("WANDB_DIR", os.environ.get("RESULTS", "~/results")))
    wandb_dir = Path(wandb_dir).expanduser()
    
    offline_run_dir = find_offline_run_directory(run_id, wandb_dir)
    
    # Try API first (works for synced runs)
    # Fall back to filesystem search if API fails
    if project is None:
        project = os.environ.get("WANDB_PROJECT", "eoss2")
    
    if use_api:
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
                    step_str = artifact.name.replace("projected_weights_step_", "")
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
            print(f"  API access failed: {e}", file=sys.stderr)
            use_api = False
    
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
    cache_dir: Optional[Path] = None,
    offline: bool = False,
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
    steps1 = sorted(get_available_steps(run_id1, project, entity, wandb_dir, offline=offline))
    steps2 = sorted(get_available_steps(run_id2, project, entity, wandb_dir, offline=offline))
    
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
        cache_dir / run_id1 if cache_dir else None,
        wandb_dir=wandb_dir,
        offline=offline
    )
    
    print("Downloading and caching artifacts for run2...")
    steps2_needed = [s2 for s1, s2 in step_pairs]
    weights2_cache = download_all_artifacts_to_cache(
        run_id2, steps2_needed, project, entity,
        cache_dir / run_id2 if cache_dir else None,
        wandb_dir=wandb_dir,
        offline=offline
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


def compute_true_min_distances(
    run_id1: str,
    run_id2: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    wandb_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
    include_init_distance: bool = False,
) -> pd.DataFrame:
    """
    Compute TRUE distance: for each step in run1, find minimum distance to any step in run2.

    Returns DataFrame with columns:
    - step: step number from run1
    - true_distance: min distance to any point in run2
    - distance_run1_init: (optional) distance from init for run1
    - distance_run2_init: (optional) distance from init for run2 at the closest step
    """
    # Discover steps
    steps1 = sorted(get_available_steps(run_id1, project, entity, wandb_dir, offline=offline))
    steps2 = sorted(get_available_steps(run_id2, project, entity, wandb_dir, offline=offline))

    if not steps1 or not steps2:
        print("Error: No artifacts available for one or both runs.", file=sys.stderr)
        return pd.DataFrame()

    print(f"Run 1 has {len(steps1)} steps, Run 2 has {len(steps2)} steps")

    # Cache/load all artifacts for both runs
    print("Downloading and caching artifacts for run1...")
    cache1 = download_all_artifacts_to_cache(
        run_id1, steps1, project, entity,
        cache_dir / run_id1 if cache_dir else None,
        wandb_dir=wandb_dir, offline=offline
    )
    print("Downloading and caching artifacts for run2...")
    cache2 = download_all_artifacts_to_cache(
        run_id2, steps2, project, entity,
        cache_dir / run_id2 if cache_dir else None,
        wandb_dir=wandb_dir, offline=offline
    )

    # Filter to steps that actually loaded
    steps1 = [s for s in steps1 if s in cache1]
    steps2 = [s for s in steps2 if s in cache2]
    if not steps1 or not steps2:
        print("Error: No artifacts loaded for one or both runs.", file=sys.stderr)
        return pd.DataFrame()

    # Stack into matrices (use float64 for numerical stability)
    A = np.stack([cache1[s] for s in steps1]).astype(np.float64, copy=False)  # shape (n1, d)
    B = np.stack([cache2[s] for s in steps2]).astype(np.float64, copy=False)  # shape (n2, d)

    # Precompute squared norms
    a2 = (A**2).sum(axis=1)  # (n1,)
    b2 = (B**2).sum(axis=1)  # (n2,)

    # Compute init distances if requested
    init_distances_run1 = {}
    init_distances_run2 = {}
    if include_init_distance:
        init_step1 = min(steps1)
        init_step2 = min(steps2)
        init_weights1 = cache1[init_step1]
        init_weights2 = cache2[init_step2]
        print(f"Using step {init_step1} as init for run1, step {init_step2} as init for run2")

        for i, step in enumerate(steps1):
            init_distances_run1[step] = float(np.linalg.norm(A[i] - init_weights1))
        for i, step in enumerate(steps2):
            init_distances_run2[step] = float(np.linalg.norm(B[i] - init_weights2))

    # Compute true distance (min distance to any point in B) for each point in A
    print("Computing TRUE distances (min distance to run2 for each step in run1)...")
    results = []

    for i, s1 in enumerate(steps1):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing step {i+1}/{len(steps1)} (step {s1})...")

        # Compute distance from A[i] to all points in B
        # d(a, b)^2 = ||a||^2 + ||b||^2 - 2 * a.b
        cross = B @ A[i]  # (n2,)
        d2 = a2[i] + b2 - 2.0 * cross  # (n2,)
        d2 = np.maximum(d2, 0.0)  # numerical stability
        dists = np.sqrt(d2)

        # Find minimum distance and which step in B it corresponds to
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        closest_step_b = steps2[min_idx]

        row = {
            'step': s1,
            'true_distance': float(min_dist),
            'closest_step_run2': closest_step_b,
        }

        if include_init_distance:
            row['distance_run1_init'] = init_distances_run1.get(s1, float('nan'))
            row['distance_run2_init'] = init_distances_run2.get(closest_step_b, float('nan'))

        results.append(row)

    return pd.DataFrame(results)


def compute_true_distances(
    run_id1: str,
    run_id2: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    wandb_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
    block_size: int = 1024,
    steps1: Optional[List[int]] = None,
    steps2: Optional[List[int]] = None,
    output_file: Optional[Path] = None,
    window_size: Optional[int] = None,
    window_mode: str = 'step',
    lr1: Optional[float] = None,
    lr2: Optional[float] = None,
) -> int:
    """
    Compute all pairwise L2 distances between run1 and run2 projected weights.
    Uses block processing and NumPy vectorization for speed and memory safety.

    Saves a long-form CSV with columns: step_run1, step_run2, weight_distance.
    Returns the number of rows written.
    """
    # Discover steps if not provided
    if steps1 is None:
        steps1 = sorted(get_available_steps(run_id1, project, entity, wandb_dir, offline=offline))
    if steps2 is None:
        steps2 = sorted(get_available_steps(run_id2, project, entity, wandb_dir, offline=offline))

    if not steps1 or not steps2:
        print("Error: No artifacts available for one or both runs.", file=sys.stderr)
        return 0

    # Cache/load all artifacts for both runs
    cache1 = download_all_artifacts_to_cache(
        run_id1, steps1, project, entity,
        cache_dir / run_id1 if cache_dir else None,
        wandb_dir=wandb_dir, offline=offline
    )
    cache2 = download_all_artifacts_to_cache(
        run_id2, steps2, project, entity,
        cache_dir / run_id2 if cache_dir else None,
        wandb_dir=wandb_dir, offline=offline
    )

    # Filter to steps that actually loaded
    steps1 = [s for s in steps1 if s in cache1]
    steps2 = [s for s in steps2 if s in cache2]
    if not steps1 or not steps2:
        print("Error: No artifacts loaded for one or both runs.", file=sys.stderr)
        return 0

    # Stack into matrices (use float64 for numerical stability)
    try:
        A = np.stack([cache1[s] for s in steps1]).astype(np.float64, copy=False)  # shape (n1, d)
        B = np.stack([cache2[s] for s in steps2]).astype(np.float64, copy=False)  # shape (n2, d)
    except Exception as e:
        print(f"Error stacking artifacts: {e}", file=sys.stderr)
        return 0

    # Prepare output
    out_path = Path(output_file) if output_file else Path(f"trajectory_distances_true_{run_id1}_{run_id2}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        # Start fresh for true pairwise dump
        try:
            out_path.unlink()
        except Exception:
            pass

    rows_written = 0
    n1 = A.shape[0]
    n2 = B.shape[0]

    if window_size is not None and window_size > 0:
        # Sliding window mode: per A step, compare against a window in B
        half = max(1, window_size // 2)
        # Precompute squared norms for B once
        b2_full = (B**2).sum(axis=1)
        for i, s1 in enumerate(steps1):
            a = A[i]  # (d,)
            a2 = float((a**2).sum())
            # Center index in B
            if window_mode == 'eta' and (lr1 is not None) and (lr2 is not None):
                target_step = int(round(s1 * (lr1 / lr2)))
                j0 = bisect_left(steps2, target_step)
                # bisect_left returns insertion point; pick nearest valid index
                if j0 >= n2:
                    j0 = n2 - 1
                elif j0 > 0 and (abs(steps2[j0] - target_step) > abs(steps2[j0-1] - target_step)):
                    j0 = j0 - 1
            else:
                # step-based alignment: center on matching/nearest step number
                j0 = bisect_left(steps2, s1)
                if j0 >= n2:
                    j0 = n2 - 1
                elif j0 > 0 and (abs(steps2[j0] - s1) > abs(steps2[j0-1] - s1)):
                    j0 = j0 - 1

            j_start = max(0, j0 - half)
            j_end = min(n2, j0 + half + 1)  # slice end is exclusive
            if j_end <= j_start:
                continue

            Bc = B[j_start:j_end]                 # (m, d)
            b2 = b2_full[j_start:j_end]           # (m,)
            cross = Bc @ a                        # (m,)
            # Numerical stabilization: clamp tiny negative values to 0 before sqrt
            d2 = a2 + b2 - 2.0 * cross            # (m,)
            d2 = np.maximum(d2, 0.0)
            dists = np.sqrt(d2)                   # (m,)

            df_chunk = pd.DataFrame({
                'step_run1': np.full(j_end - j_start, s1, dtype=int),
                'step_run2': steps2[j_start:j_end],
                'weight_distance': dists
            })
            try:
                df_chunk.to_csv(out_path, index=False, mode='a', header=(rows_written == 0))
            except Exception as e:
                print(f"Error writing CSV chunk: {e}", file=sys.stderr)
                return rows_written
            rows_written += df_chunk.shape[0]
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n1} A-steps; total rows {rows_written}")
    else:
        # Full pairwise mode: process B in blocks and compare against all A
        # Precompute squared norms for fast pairwise distances
        a2 = (A**2).sum(axis=1)                    # (n1,)
        for j in range(0, n2, block_size):
            Bc = B[j:j+block_size]                # (k, d)
            if Bc.size == 0:
                continue
            b2 = (Bc**2).sum(axis=1)              # (k,)
            # Pairwise distances via expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            cross = A @ Bc.T                      # (n1, k)
            d2 = a2[:, None] + b2[None, :] - 2.0 * cross         # (n1, k)
            d2 = np.maximum(d2, 0.0)
            D = np.sqrt(d2)                       # (n1, k)

            step_run2_chunk = steps2[j:j+block_size]
            # Long-form rows: all pairs in this block
            df_chunk = pd.DataFrame({
                'step_run1': np.repeat(steps1, len(step_run2_chunk)),
                'step_run2': np.tile(step_run2_chunk, len(steps1)),
                'weight_distance': D.reshape(-1)
            })
            # Append chunk
            try:
                df_chunk.to_csv(out_path, index=False, mode='a', header=(rows_written == 0))
            except Exception as e:
                print(f"Error writing CSV chunk: {e}", file=sys.stderr)
                return rows_written
            rows_written += df_chunk.shape[0]
            print(f"  Wrote {df_chunk.shape[0]} rows (total {rows_written})")

    print(f"Saved pairwise distances to: {out_path}")
    return rows_written


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
                       'step_sharpness', 'batch_sharpness', 'gni', 'total_accuracy',
                       'adaptive_batch_sharpness', 'adaptive_batch_sharpness_momentum',
                       'lmax_preconditioned'],
                na_values=['nan'],
                skipinitialspace=True
            )
            # Return first match for now (in practice, might need better matching)
            return df[['step', 'lambda_max']].copy()
        except Exception:
            continue
    
    raise FileNotFoundError(f"Could not find results.txt for run {run_id}")


def compute_init_distance_single_run(
    run_id_or_path: str,
    distance_type: str = 'l2',
    project: Optional[str] = None,
    wandb_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute distance from initialization within a single run.

    For L2 distance: computes ||w(t) - w(0)||_2 using projected weights.
    For test distance: computes ||f(t) - f(0)||_F using test predictions.

    Parameters:
    -----------
    run_id_or_path : str
        Wandb run ID or path to the run folder
    distance_type : str
        'l2' for weight distance or 'test' for test prediction distance
    project : str, optional
        Wandb project name
    wandb_dir : Path, optional
        Base wandb directory for offline runs
    cache_dir : Path, optional
        Directory to cache artifacts
    offline : bool
        Force offline mode (skip wandb API)
    results_dir : Path, optional
        Results directory for finding run folders (used for test distance)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: step, distance_from_init
    """
    if distance_type == 'l2':
        # L2 weight distance from initialization
        run_id = run_id_or_path

        # Get available steps
        steps = sorted(get_available_steps(
            run_id, project, None, wandb_dir, offline=offline
        ))

        if not steps:
            print(f"Warning: No steps found for run {run_id}", file=sys.stderr)
            return pd.DataFrame()

        # Download all artifacts
        weights_cache = download_all_artifacts_to_cache(
            run_id, steps, project, None,
            cache_dir / run_id if cache_dir else None,
            wandb_dir=wandb_dir,
            offline=offline
        )

        if not weights_cache:
            print(f"Warning: No weights loaded for run {run_id}", file=sys.stderr)
            return pd.DataFrame()

        # Find init step (minimum available step)
        init_step = min(weights_cache.keys())
        init_weights = weights_cache[init_step]
        print(f"Using step {init_step} as initialization")

        # Compute distance from init for each step
        results = []
        for step in sorted(weights_cache.keys()):
            weights = weights_cache[step]
            distance = float(np.linalg.norm(weights - init_weights))
            results.append({
                'step': step,
                'distance_from_init': distance
            })

        return pd.DataFrame(results)

    elif distance_type == 'test':
        # Test prediction distance from initialization
        run_path = find_run_directory(run_id_or_path, results_dir)

        if run_path is None:
            print(f"Warning: Could not find run directory for {run_id_or_path}",
                  file=sys.stderr)
            return pd.DataFrame()

        # Load test predictions
        steps, predictions = load_test_predictions(run_path)

        if len(steps) == 0:
            print(f"Warning: No test predictions found in {run_path}",
                  file=sys.stderr)
            return pd.DataFrame()

        # Find init step (minimum step)
        init_idx = 0  # First entry is usually init
        init_step = int(steps[init_idx])
        init_pred = predictions[init_idx]
        print(f"Using step {init_step} as initialization")

        # Compute Frobenius norm distance from init for each step
        results = []
        for i, step in enumerate(steps):
            pred = predictions[i]
            distance = float(np.linalg.norm(pred - init_pred, ord='fro'))
            results.append({
                'step': int(step),
                'distance_from_init': distance
            })

        return pd.DataFrame(results)

    else:
        raise ValueError(f"Unknown distance_type: {distance_type}. Use 'l2' or 'test'.")


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
    parser.add_argument('--offline', action='store_true', help='Force offline mode: skip wandb API and use filesystem only')
    # Pairwise true distance mode
    parser.add_argument('--true', action='store_true', help='Compute all pairwise distances (run1 vs run2)')
    parser.add_argument('--block-size', type=int, default=1024, help='Block size for pairwise distance computation')
    parser.add_argument('--window-size', type=int, default=None, help='Sliding window size in run B per A point (limits comparisons to ~window-size points around aligned index)')
    parser.add_argument('--window-mode', type=str, choices=['step','eta'], default='step', help='Center window using matching step or equivalent step*eta (requires --lr1/--lr2 for eta)')
    parser.add_argument('--init-distance', action='store_true', help='Compute only distance from initialization for each run (outputs distance_run1_init and distance_run2_init, not distance between runs)')
    parser.add_argument('--true-min', action='store_true', help='Compute TRUE distance: for each step in run1, output the minimum distance to any step in run2')
    parser.add_argument('--test-distance', action='store_true', help='Compute test prediction distance (Frobenius norm) instead of weight distance')

    args = parser.parse_args()
    
    if not WANDB_AVAILABLE and not args.offline and not args.test_distance:
        print("Error: wandb library not available. Re-run with --offline to use filesystem-only mode.", file=sys.stderr)
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

    results_dir = Path(args.results_dir).expanduser() if args.results_dir else None

    # Test distance mode - compute Frobenius norm between test predictions
    if args.test_distance:
        if args.output is None or args.output == f"trajectory_distances_{args.run_id1}_{args.run_id2}.csv":
            args.output = f"test_distances_{args.run_id1}_{args.run_id2}.csv"
        output_path = Path(args.output)

        # Find run directories
        run_path1 = find_run_directory(args.run_id1, results_dir)
        run_path2 = find_run_directory(args.run_id2, results_dir)

        if run_path1 is None:
            print(f"Error: Could not find run directory for {args.run_id1}", file=sys.stderr)
            return 1
        if run_path2 is None:
            print(f"Error: Could not find run directory for {args.run_id2}", file=sys.stderr)
            return 1

        print(f"Found run1 at: {run_path1}")
        print(f"Found run2 at: {run_path2}")

        df = compute_test_distances(
            run_path1=run_path1,
            run_path2=run_path2,
            lr1=args.lr1,
            lr2=args.lr2,
            include_init_distance=args.init_distance,
            true_min=args.true_min,
        )

        if df.empty:
            print("No distances computed.", file=sys.stderr)
            return 1

        df.to_csv(output_path, index=False)
        print(f"\nSaved test distances to: {output_path}")
        print(f"Computed test distances for {len(df)} steps")
        return 0

    # Init distance mode (standalone) - only compute distance from initialization for each run
    # Skip this if --true-min or --test-distance is also specified (init-distance becomes a modifier)
    if args.init_distance and not args.true_min and not args.test_distance:
        if args.output is None or args.output == f"trajectory_distances_{args.run_id1}_{args.run_id2}.csv":
            args.output = f"init_distances_{args.run_id1}_{args.run_id2}.csv"
        output_path = Path(args.output)

        print("Computing distance from initialization for each run...")

        # Get available steps for both runs
        steps1 = sorted(get_available_steps(args.run_id1, args.project, args.entity, wandb_dir, offline=args.offline))
        steps2 = sorted(get_available_steps(args.run_id2, args.project, args.entity, wandb_dir, offline=args.offline))
        print(f"Run 1 has {len(steps1)} steps, Run 2 has {len(steps2)} steps")

        # Download all artifacts
        print("Downloading and caching artifacts for run1...")
        weights1_cache = download_all_artifacts_to_cache(
            args.run_id1, steps1, args.project, args.entity,
            cache_dir / args.run_id1 if cache_dir else None,
            wandb_dir=wandb_dir,
            offline=args.offline
        )

        print("Downloading and caching artifacts for run2...")
        weights2_cache = download_all_artifacts_to_cache(
            args.run_id2, steps2, args.project, args.entity,
            cache_dir / args.run_id2 if cache_dir else None,
            wandb_dir=wandb_dir,
            offline=args.offline
        )

        # Find init step (minimum available step) for each run
        init_step1 = min(weights1_cache.keys()) if weights1_cache else None
        init_step2 = min(weights2_cache.keys()) if weights2_cache else None

        if init_step1 is None or init_step2 is None:
            print("Error: Could not find init weights for one or both runs", file=sys.stderr)
            return 1

        init_weights1 = weights1_cache[init_step1]
        init_weights2 = weights2_cache[init_step2]
        print(f"Using step {init_step1} as init for run1, step {init_step2} as init for run2")

        # Compute distance from init for each run
        print("Computing distances from initialization...")
        init_distances_run1 = {}
        init_distances_run2 = {}

        for step, weights in weights1_cache.items():
            init_distances_run1[step] = float(np.linalg.norm(weights - init_weights1))

        for step, weights in weights2_cache.items():
            init_distances_run2[step] = float(np.linalg.norm(weights - init_weights2))

        # Create output DataFrame - use common steps or all steps
        common_steps = sorted(set(init_distances_run1.keys()) & set(init_distances_run2.keys()))
        print(f"Found {len(common_steps)} common steps between runs")

        df = pd.DataFrame({
            'step': common_steps,
            'distance_run1_init': [init_distances_run1[s] for s in common_steps],
            'distance_run2_init': [init_distances_run2[s] for s in common_steps]
        })

        df.to_csv(output_path, index=False)
        print(f"\nSaved init distances to: {output_path}")
        print(f"Computed init distances for {len(common_steps)} steps")
        return 0

    # True-min mode: for each step in run1, compute min distance to any step in run2
    if args.true_min:
        if args.output is None or args.output == f"trajectory_distances_{args.run_id1}_{args.run_id2}.csv":
            args.output = f"true_min_distances_{args.run_id1}_{args.run_id2}.csv"
        output_path = Path(args.output)

        df = compute_true_min_distances(
            run_id1=args.run_id1,
            run_id2=args.run_id2,
            project=args.project,
            entity=args.entity,
            wandb_dir=wandb_dir,
            cache_dir=cache_dir,
            offline=args.offline,
            include_init_distance=args.init_distance,
        )

        if df.empty:
            print("No distances computed.", file=sys.stderr)
            return 1

        df.to_csv(output_path, index=False)
        print(f"\nSaved TRUE distances to: {output_path}")
        print(f"Computed TRUE distances for {len(df)} steps")
        return 0

    # True pairwise mode
    if args.true:
        if args.output is None:
            args.output = f"trajectory_distances_true_{args.run_id1}_{args.run_id2}.csv"
        rows = compute_true_distances(
            run_id1=args.run_id1,
            run_id2=args.run_id2,
            project=args.project,
            entity=args.entity,
            wandb_dir=wandb_dir,
            cache_dir=cache_dir,
            offline=args.offline,
            block_size=args.block_size,
            steps1=None,
            steps2=None,
            output_file=Path(args.output),
            window_size=args.window_size,
            window_mode=args.window_mode,
            lr1=args.lr1,
            lr2=args.lr2,
        )


        if rows == 0:
            print("No pairwise distances computed.", file=sys.stderr)
            return 1
        print(f"Computed {rows} pairwise distance rows")
        print(f"Saved to: {args.output}")
        return 0
    
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
        cache_dir=cache_dir,
        offline=args.offline
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
    else:
        df = pd.DataFrame({
            'step': steps,
            'weight_distance': [distances[s] for s in steps]
        })
    
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

