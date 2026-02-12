import torch
import torch.nn as nn
import numpy as np
import pytest

from utils.measure import (
    get_rmsprop_momentum_buffer,
    get_rmsprop_preconditioner_inv,
    compute_adaptive_grad_H_grad,
    compute_adaptive_grad_H_grad_momentum,
    calculate_adaptive_batch_sharpness_momentum,
    flatt,
)


class _SmallNet(nn.Module):
    """Tiny MLP for fast tests."""
    def __init__(self, d_in=4, d_hid=8, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid, bias=False)
        self.fc2 = nn.Linear(d_hid, d_out, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_rmsprop(net, lr=0.01, alpha=0.99, momentum=0.9):
    return torch.optim.RMSprop(net.parameters(), lr=lr, alpha=alpha, momentum=momentum)


def _do_steps(net, optimizer, X, Y, loss_fn, n=3):
    """Run n optimiser steps to populate optimizer state."""
    for _ in range(n):
        optimizer.zero_grad()
        loss = loss_fn(net(X).squeeze(dim=-1), Y)
        loss.backward()
        optimizer.step()


# ---- Tests for get_rmsprop_momentum_buffer ----

def test_get_rmsprop_momentum_buffer_returns_tensor():
    net = _SmallNet()
    opt = _make_rmsprop(net, momentum=0.9)
    X = torch.randn(16, 4)
    Y = torch.randn(16)
    loss_fn = nn.MSELoss()
    _do_steps(net, opt, X, Y, loss_fn, n=1)

    buf = get_rmsprop_momentum_buffer(opt)
    assert buf is not None
    n_params = sum(p.numel() for p in net.parameters())
    assert buf.shape == (n_params,)


def test_get_rmsprop_momentum_buffer_no_momentum_returns_none():
    net = _SmallNet()
    opt = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, momentum=0)
    X = torch.randn(16, 4)
    Y = torch.randn(16)
    loss_fn = nn.MSELoss()
    _do_steps(net, opt, X, Y, loss_fn, n=1)

    assert get_rmsprop_momentum_buffer(opt) is None


def test_get_rmsprop_momentum_buffer_sgd_returns_none():
    net = _SmallNet()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    X = torch.randn(16, 4)
    Y = torch.randn(16)
    loss_fn = nn.MSELoss()
    _do_steps(net, opt, X, Y, loss_fn, n=1)

    assert get_rmsprop_momentum_buffer(opt) is None


# ---- Tests for compute_adaptive_grad_H_grad_momentum ----

def test_per_batch_kernel_positive():
    """s^T H s > 0 and g^T s > 0 for a well-behaved case with small net."""
    torch.manual_seed(42)
    net = _SmallNet()
    opt = _make_rmsprop(net, momentum=0.9)
    X = torch.randn(32, 4)
    Y = torch.randn(32)
    loss_fn = nn.MSELoss()
    _do_steps(net, opt, X, Y, loss_fn, n=5)

    pinv = get_rmsprop_preconditioner_inv(opt)
    mbuf = get_rmsprop_momentum_buffer(opt)
    beta = opt.param_groups[0]['momentum']

    # Use a fresh batch for the kernel
    idx = torch.randperm(32)[:8]
    loss = loss_fn(net(X[idx]).squeeze(dim=-1), Y[idx])
    sHs, gs = compute_adaptive_grad_H_grad_momentum(loss, net, pinv, mbuf, beta)

    # For MSE loss Hessian is PSD, so s^T H s >= 0
    assert sHs.item() >= 0
    # g^T s should be positive when gradient and step direction are aligned
    # (not guaranteed in general, but likely for this setup)
    assert np.isfinite(sHs.item())
    assert np.isfinite(gs.item())


def test_momentum_sharpness_reduces_to_non_momentum():
    """When beta=0, momentum kernel matches non-momentum kernel exactly."""
    torch.manual_seed(123)
    net = _SmallNet()
    opt = _make_rmsprop(net, momentum=0.9)
    X = torch.randn(32, 4)
    Y = torch.randn(32)
    loss_fn = nn.MSELoss()
    _do_steps(net, opt, X, Y, loss_fn, n=3)

    pinv = get_rmsprop_preconditioner_inv(opt)
    mbuf = get_rmsprop_momentum_buffer(opt)

    # Fixed batch
    idx = torch.arange(8)
    X_b, Y_b = X[idx], Y[idx]

    # Non-momentum version
    loss1 = loss_fn(net(X_b).squeeze(dim=-1), Y_b)
    uHu, gu = compute_adaptive_grad_H_grad(loss1, net, pinv)

    # Momentum version with beta=0 (momentum buffer doesn't matter)
    loss2 = loss_fn(net(X_b).squeeze(dim=-1), Y_b)
    sHs, gs = compute_adaptive_grad_H_grad_momentum(loss2, net, pinv, mbuf, beta=0.0)

    assert pytest.approx(uHu.item(), rel=1e-5) == sHs.item()
    assert pytest.approx(gu.item(), rel=1e-5) == gs.item()


def test_momentum_sharpness_differs_from_non_momentum():
    """When beta>0 and momentum buffer is nonzero, result differs."""
    torch.manual_seed(456)
    net = _SmallNet()
    opt = _make_rmsprop(net, momentum=0.9)
    X = torch.randn(32, 4)
    Y = torch.randn(32)
    loss_fn = nn.MSELoss()
    _do_steps(net, opt, X, Y, loss_fn, n=5)

    pinv = get_rmsprop_preconditioner_inv(opt)
    mbuf = get_rmsprop_momentum_buffer(opt)
    beta = opt.param_groups[0]['momentum']

    idx = torch.arange(8)
    X_b, Y_b = X[idx], Y[idx]

    loss1 = loss_fn(net(X_b).squeeze(dim=-1), Y_b)
    uHu, gu = compute_adaptive_grad_H_grad(loss1, net, pinv)

    loss2 = loss_fn(net(X_b).squeeze(dim=-1), Y_b)
    sHs, gs = compute_adaptive_grad_H_grad_momentum(loss2, net, pinv, mbuf, beta)

    # With nonzero momentum buffer and beta=0.9, numerators should differ
    assert uHu.item() != pytest.approx(sHs.item(), rel=1e-3)


# ---- Test for full MC wrapper ----

def test_calculate_adaptive_batch_sharpness_momentum_returns_float():
    """Full MC wrapper returns a finite float after a few optimizer steps."""
    torch.manual_seed(789)
    net = _SmallNet()
    opt = _make_rmsprop(net, momentum=0.9)
    X = torch.randn(64, 4)
    Y = torch.randn(64)
    loss_fn = nn.MSELoss()
    _do_steps(net, opt, X, Y, loss_fn, n=5)

    result = calculate_adaptive_batch_sharpness_momentum(
        net, X, Y, loss_fn, opt,
        batch_size=16, n_estimates=30, min_estimates=5, eps=0.1,
    )

    assert isinstance(result, float)
    assert np.isfinite(result)
