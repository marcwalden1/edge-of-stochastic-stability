"""Test that --experiment-subdir correctly modifies the output path."""

import argparse
import shutil
import tempfile
from pathlib import Path

from utils.storage import initialize_folders


def test_experiment_subdir():
    """Test that experiment_subdir inserts a subdirectory in the output path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_folder = Path(tmpdir)

        # Create args with experiment_subdir
        args = argparse.Namespace(
            dataset='cifar10',
            model='mlp',
            lr=0.001,
            batch=16,
            gd_noise=0.0,
            experiment_tag=None,
            experiment_subdir='experiment_1'
        )

        runs_folder = initialize_folders(args, results_folder)

        # Verify the path contains experiment_subdir
        assert 'experiment_1' in str(runs_folder), f"Expected 'experiment_1' in path, got {runs_folder}"
        assert runs_folder.exists(), f"Folder was not created: {runs_folder}"
        assert (runs_folder / 'results.txt').exists(), "results.txt was not created"

        # Verify path structure: results/plaintext/cifar10_mlp/experiment_1/{config_name}
        parts = runs_folder.relative_to(results_folder).parts
        assert parts[0] == 'plaintext', f"Expected 'plaintext', got {parts[0]}"
        assert parts[1] == 'cifar10_mlp', f"Expected 'cifar10_mlp', got {parts[1]}"
        assert parts[2] == 'experiment_1', f"Expected 'experiment_1', got {parts[2]}"

        print(f"✓ With --experiment-subdir: {runs_folder}")


def test_without_experiment_subdir():
    """Test backward compatibility - no subdir when experiment_subdir is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_folder = Path(tmpdir)

        # Create args without experiment_subdir
        args = argparse.Namespace(
            dataset='cifar10',
            model='mlp',
            lr=0.001,
            batch=16,
            gd_noise=0.0,
            experiment_tag=None,
            experiment_subdir=None
        )

        runs_folder = initialize_folders(args, results_folder)

        # Verify path structure: results/plaintext/cifar10_mlp/{config_name} (no subdir)
        parts = runs_folder.relative_to(results_folder).parts
        assert parts[0] == 'plaintext', f"Expected 'plaintext', got {parts[0]}"
        assert parts[1] == 'cifar10_mlp', f"Expected 'cifar10_mlp', got {parts[1]}"
        assert len(parts) == 3, f"Expected 3 path components, got {len(parts)}: {parts}"

        print(f"✓ Without --experiment-subdir: {runs_folder}")


def test_without_experiment_subdir_attr():
    """Test backward compatibility - works when experiment_subdir attr doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_folder = Path(tmpdir)

        # Create args without experiment_subdir attribute at all
        args = argparse.Namespace(
            dataset='cifar10',
            model='mlp',
            lr=0.001,
            batch=16,
            gd_noise=0.0,
            experiment_tag=None
        )

        runs_folder = initialize_folders(args, results_folder)

        # Verify path structure: results/plaintext/cifar10_mlp/{config_name} (no subdir)
        parts = runs_folder.relative_to(results_folder).parts
        assert len(parts) == 3, f"Expected 3 path components, got {len(parts)}: {parts}"

        print(f"✓ Without experiment_subdir attr: {runs_folder}")


if __name__ == '__main__':
    test_experiment_subdir()
    test_without_experiment_subdir()
    test_without_experiment_subdir_attr()
    print("\nAll tests passed!")
