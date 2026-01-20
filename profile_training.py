#!/usr/bin/env python3
"""
Profile training bottlenecks to identify performance issues.
Run this on the same hardware as your training to get accurate measurements.

Usage:
    python profile_training.py [--model mlp|cnn] [--batch-size N] [--num-data N]
"""

import torch
import time
import numpy as np
from pathlib import Path
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data import prepare_dataset
from utils.nets import SquaredLoss, prepare_net, initialize_net
from utils.measure import (
    compute_eigenvalues,
    calculate_averaged_grad_H_grad_step,
    compute_grad_H_grad,
)

DATASET_FOLDER = Path(os.environ.get('DATASETS', '~/datasets')).expanduser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def time_function(func, n_runs=5, warmup=2):
    """Time a function, returning mean and std of execution time."""
    times = []
    for i in range(warmup + n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        if i >= warmup:
            times.append(end - start)
    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser(description="Profile training bottlenecks")
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'])
    parser.add_argument('--batch-size', type=int, default=1876)
    parser.add_argument('--num-data', type=int, default=8192)
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING BOTTLENECK PROFILER")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Setup from args
    batch_size = args.batch_size
    num_data = args.num_data
    model_type = args.model

    print(f"Configuration: model={model_type}, batch_size={batch_size}, num_data={num_data}")
    print()

    # Load data
    print("Loading dataset...")
    X, Y = prepare_dataset(
        dataset='cifar10',
        num_data=num_data,
        classes=[1, 9],
        folder=DATASET_FOLDER,
        seed=111,
    )
    X, Y = X.to(device), Y.to(device)
    print(f"Data shape: X={X.shape}, Y={Y.shape}")

    # Create model
    print("Creating model...")
    net = prepare_net(model_type, X, 'silu')
    initialize_net(net, 0.2, seed=8312)
    net = net.to(device)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    loss_fn = SquaredLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Get a batch
    idx = torch.randperm(len(X))[:batch_size]
    X_batch, Y_batch = X[idx], Y[idx]

    print("=" * 70)
    print("TIMING INDIVIDUAL OPERATIONS")
    print("=" * 70)
    print()

    # 1. Time cuda.empty_cache()
    if torch.cuda.is_available():
        def empty_cache_op():
            torch.cuda.empty_cache()

        mean_t, std_t = time_function(empty_cache_op, n_runs=20, warmup=5)
        print(f"1. torch.cuda.empty_cache():")
        print(f"   Time: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms")
        print(f"   If called 1000x per batch_sharpness: {mean_t*1000:.1f} seconds")
        print()

    # 2. Time a single forward + backward pass (batch)
    def forward_backward_batch():
        optimizer.zero_grad()
        preds = net(X_batch).squeeze(dim=-1)
        loss = loss_fn(preds, Y_batch)
        loss.backward()

    mean_t, std_t = time_function(forward_backward_batch, n_runs=10, warmup=3)
    print(f"2. Single forward+backward (batch_size={batch_size}):")
    print(f"   Time: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms")
    print()

    # 3. Time a single forward + backward pass (full dataset, for lambdamax)
    lmax_subset_size = min(4096, len(X))
    idx_full = torch.randperm(len(X))[:lmax_subset_size]
    X_lmax, Y_lmax = X[idx_full], Y[idx_full]

    def forward_backward_full():
        optimizer.zero_grad()
        preds = net(X_lmax).squeeze(dim=-1)
        loss = loss_fn(preds, Y_lmax)
        loss.backward()

    mean_t, std_t = time_function(forward_backward_full, n_runs=5, warmup=2)
    print(f"3. Single forward+backward (lmax subset={lmax_subset_size}):")
    print(f"   Time: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms")
    print()

    # 4. Time compute_grad_H_grad (single Hessian-vector product)
    def grad_h_grad():
        optimizer.zero_grad()
        preds = net(X_batch).squeeze(dim=-1)
        loss = loss_fn(preds, Y_batch)
        return compute_grad_H_grad(loss, net)

    mean_t, std_t = time_function(grad_h_grad, n_runs=10, warmup=3)
    print(f"4. compute_grad_H_grad (single batch Hessian-vector product):")
    print(f"   Time: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms")
    time_per_ghg = mean_t
    print()

    # 5. Time full batch_sharpness measurement
    print(f"5. Full batch_sharpness measurement (n_estimates=1000, eps=0.005):")
    print("   Running... (this may take a while)")

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    result = calculate_averaged_grad_H_grad_step(
        net, X, Y, loss_fn,
        batch_size=batch_size,
        n_estimates=1000,
        min_estimates=20,
        eps=0.005,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_sharpness_time = time.perf_counter() - start

    print(f"   Time: {batch_sharpness_time:.2f} seconds")
    print(f"   Result: {result:.2f}")
    print()

    # 6. Time lambdamax computation
    print(f"6. Lambda max computation (subset={lmax_subset_size}, LOBPCG):")
    print("   Running... (this may take a while)")

    optimizer.zero_grad()
    preds = net(X_lmax).squeeze(dim=-1)
    loss = loss_fn(preds, Y_lmax)

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    lmax = compute_eigenvalues(
        loss, net,
        k=1,
        max_iterations=100,
        reltol=0.005,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    lambdamax_time = time.perf_counter() - start

    print(f"   Time: {lambdamax_time:.2f} seconds")
    print(f"   Result: {lmax.item():.2f}")
    print()

    # 7. Time a pure training step (no measurements)
    def training_step():
        optimizer.zero_grad()
        preds = net(X_batch).squeeze(dim=-1)
        loss = loss_fn(preds, Y_batch)
        loss.backward()
        optimizer.step()

    mean_t, std_t = time_function(training_step, n_runs=20, warmup=5)
    print(f"7. Pure training step (forward + backward + optimizer.step):")
    print(f"   Time: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms")
    time_per_step = mean_t
    print()

    # Summary and projections
    print("=" * 70)
    print("PROJECTED TRAINING TIME (25,000 steps)")
    print("=" * 70)
    print()

    total_steps = 25000

    # Pure training time
    pure_training = total_steps * time_per_step
    print(f"Pure training (no measurements): {pure_training/60:.1f} minutes")
    print()

    # With batch_sharpness (every 32 steps for batch > 33)
    bs_freq = 32
    bs_calls = total_steps // bs_freq
    bs_total = bs_calls * batch_sharpness_time
    print(f"Batch sharpness overhead:")
    print(f"   Frequency: every {bs_freq} steps")
    print(f"   Calls: {bs_calls}")
    print(f"   Time per call: {batch_sharpness_time:.2f}s")
    print(f"   Total: {bs_total/60:.1f} minutes ({bs_total/3600:.2f} hours)")
    print()

    # With lambdamax (every 64 steps for batch > 33, doubling after 10k)
    # Simplified: assume average frequency of ~100 steps
    lmax_calls_0_10k = 10000 // 64
    lmax_calls_10k_25k = 15000 // 128
    lmax_calls = lmax_calls_0_10k + lmax_calls_10k_25k
    lmax_total = lmax_calls * lambdamax_time
    print(f"Lambda max overhead:")
    print(f"   Frequency: ~64 steps (0-10k), ~128 steps (10k-25k)")
    print(f"   Calls: ~{lmax_calls}")
    print(f"   Time per call: {lambdamax_time:.2f}s")
    print(f"   Total: {lmax_total/60:.1f} minutes ({lmax_total/3600:.2f} hours)")
    print()

    total_time = pure_training + bs_total + lmax_total
    print(f"TOTAL PROJECTED TIME: {total_time/3600:.2f} hours")
    print()

    # Breakdown of batch_sharpness time
    print("=" * 70)
    print("BATCH SHARPNESS BREAKDOWN")
    print("=" * 70)
    print()

    if torch.cuda.is_available():
        # Estimate how much time is spent in empty_cache
        # Based on: empty_cache is called every MC iteration when batch > 128
        empty_cache_time, _ = time_function(empty_cache_op, n_runs=20, warmup=5)

        # Rough estimate: batch_sharpness typically converges in 20-100 samples
        # Let's assume 50 samples on average
        estimated_mc_samples = 50
        empty_cache_overhead = estimated_mc_samples * empty_cache_time

        print(f"If batch_sharpness uses ~{estimated_mc_samples} MC samples on average:")
        print(f"   empty_cache overhead per call: {empty_cache_overhead:.2f}s")
        print(f"   Total empty_cache time ({bs_calls} calls): {bs_calls * empty_cache_overhead / 60:.1f} minutes")
        print(f"   This is {100 * empty_cache_overhead / batch_sharpness_time:.0f}% of batch_sharpness time")
        print()

        # What if we removed empty_cache?
        projected_bs_time_no_cache = batch_sharpness_time - empty_cache_overhead
        projected_bs_total_no_cache = bs_calls * projected_bs_time_no_cache
        projected_total_no_cache = pure_training + projected_bs_total_no_cache + lmax_total

        print(f"PROJECTED TIME WITHOUT empty_cache in loop: {projected_total_no_cache/3600:.2f} hours")
        print(f"   Savings: {(total_time - projected_total_no_cache)/3600:.2f} hours")


if __name__ == '__main__':
    main()
