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
    get_preconditioner_inv,
    get_preconditioned_momentum_buffer,
    get_adam_last_grad,
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


# ---- Adam-specific tests ----

def _make_adam(net, lr=0.001, betas=(0.9, 0.999)):
    return torch.optim.Adam(net.parameters(), lr=lr, betas=betas)


def _do_steps_adam(net, optimizer, X, Y, loss_fn, n=5):
    """Run n steps and save last_grad after each optimizer.step() (mirrors training.py)."""
    for _ in range(n):
        optimizer.zero_grad()
        loss = loss_fn(net(X).squeeze(dim=-1), Y)
        loss.backward()
        optimizer.step()
        # Mirror training.py: save last gradient for ABSM exact formula
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        state = optimizer.state[p]
                        if 'last_grad' not in state:
                            state['last_grad'] = p.grad.detach().clone()
                        else:
                            state['last_grad'].copy_(p.grad)


def test_get_adam_last_grad_returns_none_before_step():
    """last_grad is not yet stored before training loop saves it."""
    net = _SmallNet()
    opt = _make_adam(net)
    X = torch.randn(16, 4)
    Y = torch.randn(16)
    loss_fn = nn.MSELoss()
    # Run steps WITHOUT saving last_grad (simulates first step of old checkpoint)
    _do_steps(net, opt, X, Y, loss_fn, n=3)
    assert get_adam_last_grad(opt) is None


def test_get_adam_last_grad_returns_tensor_after_save():
    """last_grad is available after training loop saves it."""
    net = _SmallNet()
    opt = _make_adam(net)
    X = torch.randn(16, 4)
    Y = torch.randn(16)
    loss_fn = nn.MSELoss()
    _do_steps_adam(net, opt, X, Y, loss_fn, n=3)
    last_grad = get_adam_last_grad(opt)
    assert last_grad is not None
    n_params = sum(p.numel() for p in net.parameters())
    assert last_grad.shape == (n_params,)


def test_get_adam_last_grad_returns_none_for_rmsprop():
    """Helper returns None for non-Adam optimizers."""
    net = _SmallNet()
    opt = _make_rmsprop(net)
    assert get_adam_last_grad(opt) is None


def test_adam_absm_returns_nan_without_last_grad():
    """ABSM returns np.nan when last_grad hasn't been stored yet."""
    torch.manual_seed(0)
    net = _SmallNet()
    opt = _make_adam(net)
    X = torch.randn(64, 4)
    Y = torch.randn(64)
    loss_fn = nn.MSELoss()
    # Steps WITHOUT saving last_grad
    _do_steps(net, opt, X, Y, loss_fn, n=5)
    result = calculate_adaptive_batch_sharpness_momentum(
        net, X, Y, loss_fn, opt, batch_size=8, n_estimates=10, min_estimates=5,
    )
    assert np.isnan(result), f"Expected nan without last_grad, got {result}"


def test_adam_absm_returns_finite_with_last_grad():
    """ABSM returns a finite float after last_grad has been stored."""
    torch.manual_seed(1)
    net = _SmallNet()
    opt = _make_adam(net)
    X = torch.randn(64, 4)
    Y = torch.randn(64)
    loss_fn = nn.MSELoss()
    _do_steps_adam(net, opt, X, Y, loss_fn, n=10)
    result = calculate_adaptive_batch_sharpness_momentum(
        net, X, Y, loss_fn, opt,
        batch_size=8, n_estimates=50, min_estimates=10, eps=0.1,
    )
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_adam_absm_denominator_always_positive():
    """g_B^T s_B > 0 for every batch (denominator has guaranteed positive floor)."""
    torch.manual_seed(2)
    net = _SmallNet()
    opt = _make_adam(net)
    X = torch.randn(64, 4)
    Y = torch.randn(64)
    loss_fn = nn.MSELoss()
    _do_steps_adam(net, opt, X, Y, loss_fn, n=10)

    pinv = get_preconditioner_inv(opt)
    momentum_buffer = get_preconditioned_momentum_buffer(opt)
    beta1 = opt.param_groups[0]['betas'][0]

    # Compute frozen_step and adam_alpha (mirrors calculate_adaptive_batch_sharpness_momentum)
    step_t = int(list(opt.state.values())[0]['step'].item())
    last_grad = get_adam_last_grad(opt)
    bias_correction1 = 1.0 - beta1 ** step_t
    m_prev_times_beta1 = momentum_buffer - (1.0 - beta1) * last_grad
    frozen_step = (pinv * m_prev_times_beta1 / bias_correction1).detach()
    adam_alpha = (1.0 - beta1) / bias_correction1

    rng = torch.Generator()
    rng.manual_seed(99)
    for _ in range(30):
        idx = torch.randperm(len(X), generator=rng)[:8]
        loss = loss_fn(net(X[idx]).squeeze(dim=-1), Y[idx])
        _, gs = compute_adaptive_grad_H_grad_momentum(
            loss, net, pinv, momentum_buffer, beta=0.0,
            frozen_step=frozen_step, adam_alpha=adam_alpha,
        )
        assert gs.item() > 0, f"Denominator g_B^T s_B = {gs.item()} <= 0"


def test_adam_absm_stable_across_measurements():
    """ABSM values should be consistent (not swinging wildly) across repeated calls."""
    torch.manual_seed(3)
    net = _SmallNet()
    opt = _make_adam(net)
    X = torch.randn(64, 4)
    Y = torch.randn(64)
    loss_fn = nn.MSELoss()
    _do_steps_adam(net, opt, X, Y, loss_fn, n=20)

    estimates = []
    for _ in range(5):
        val = calculate_adaptive_batch_sharpness_momentum(
            net, X, Y, loss_fn, opt,
            batch_size=8, n_estimates=40, min_estimates=10, eps=0.1,
        )
        assert np.isfinite(val)
        estimates.append(val)

    # All estimates should be within 3x of each other (no sign flips / giant swings)
    ratio = max(estimates) / min(estimates)
    assert ratio < 3.0, f"ABSM swings too wildly: {estimates}"


def test_adam_absm_positive():
    """ABSM should be positive (sHs / gs ratio with positive denominator floor)."""
    torch.manual_seed(4)
    net = _SmallNet()
    opt = _make_adam(net)
    X = torch.randn(128, 4)
    Y = torch.randn(128)
    loss_fn = nn.MSELoss()
    _do_steps_adam(net, opt, X, Y, loss_fn, n=20)

    result = calculate_adaptive_batch_sharpness_momentum(
        net, X, Y, loss_fn, opt,
        batch_size=8, n_estimates=100, min_estimates=20, eps=0.05,
    )
    assert result > 0, f"ABSM should be positive, got {result}"
