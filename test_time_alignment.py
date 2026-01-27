#!/usr/bin/env python3
"""Test the time-alignment feature in compare_test_predictions.py"""

import tempfile
import numpy as np
from pathlib import Path

# Import the functions we want to test
from visualization.compare_test_predictions import (
    extract_lr_from_results,
    compute_frobenius_distances_time_aligned,
)


def test_extract_lr_from_results():
    """Test learning rate extraction from results.txt"""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_path = Path(tmpdir)
        results_file = run_path / 'results.txt'

        # Test standard format
        results_file.write_text("Arguments: Namespace(lr=0.01, batch_size=128)\n")
        assert extract_lr_from_results(run_path) == 0.01

        # Test scientific notation
        results_file.write_text("Arguments: Namespace(lr=1e-3, other=5)\n")
        assert extract_lr_from_results(run_path) == 0.001

        print("✓ extract_lr_from_results works correctly")


def test_compute_frobenius_distances_time_aligned():
    """Test time-aligned distance computation"""
    # Run1: lr=0.01, steps at 0, 100, 200, 300
    # Run2: lr=0.001, steps at 0, 1000, 2000, 3000
    # Time alignment: step1 * 0.01 = step2 * 0.001
    # So step1=100 matches step2=1000 (both at t=1.0)

    steps1 = np.array([0, 100, 200, 300])
    steps2 = np.array([0, 1000, 2000, 3000])

    # Create simple prediction arrays (2 samples, 3 classes)
    preds1 = np.zeros((4, 2, 3))
    preds2 = np.zeros((4, 2, 3))

    # Set different values so we can verify correct matching
    for i in range(4):
        preds1[i] = i * 0.1
        preds2[i] = i * 0.1 + 0.05  # Slight offset

    lr1, lr2 = 0.01, 0.001

    df = compute_frobenius_distances_time_aligned(
        steps1, preds1, lr1, steps2, preds2, lr2
    )

    # Check we got 4 matched pairs
    assert len(df) == 4, f"Expected 4 matches, got {len(df)}"

    # Check columns exist
    assert 'time' in df.columns
    assert 'step_run1' in df.columns
    assert 'step_run2' in df.columns
    assert 'frobenius_distance' in df.columns

    # Check time values (should be 0, 1.0, 2.0, 3.0)
    expected_times = [0.0, 1.0, 2.0, 3.0]
    actual_times = sorted(df['time'].tolist())
    assert actual_times == expected_times, f"Expected times {expected_times}, got {actual_times}"

    # Check step matching
    df_sorted = df.sort_values('time')
    assert df_sorted.iloc[0]['step_run1'] == 0
    assert df_sorted.iloc[0]['step_run2'] == 0
    assert df_sorted.iloc[1]['step_run1'] == 100
    assert df_sorted.iloc[1]['step_run2'] == 1000

    print("✓ compute_frobenius_distances_time_aligned works correctly")


def test_no_matches_raises_error():
    """Test that we get an error when no step pairs match"""
    steps1 = np.array([100, 200])
    steps2 = np.array([150, 250])  # No matches possible
    preds1 = np.zeros((2, 2, 3))
    preds2 = np.zeros((2, 2, 3))

    try:
        compute_frobenius_distances_time_aligned(
            steps1, preds1, 0.01, steps2, preds2, 0.01
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No matching step pairs found" in str(e)
        print("✓ Correctly raises error when no matches found")


if __name__ == '__main__':
    test_extract_lr_from_results()
    test_compute_frobenius_distances_time_aligned()
    test_no_matches_raises_error()
    print("\nAll tests passed!")
