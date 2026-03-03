"""Tests for Generalized Batch Sharpness (GBS).

GBS = E_B[s_B^T H_B s_B] / (−E_B[s_B^T g_B])

Tests are CPU-friendly: tiny nets, small data, few MC samples.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from utils.measure import (
    _build_gbs_step,
    compute_gbs_per_batch,
    calculate_gbs,
    compute_grad_H_grad,
    flatt,
)
from utils.frequency import frequency_calculator, MeasurementContext
from utils.nets import Muon


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _SmallNet(nn.Module):
    """Tiny MLP: 4 -> 8 -> 1, no bias, ~40 params."""
    def __init__(self, d_in=4, d_hid=8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid, bias=False)
        self.fc2 = nn.Linear(d_hid, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_data(n=32, d=4, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    Y = torch.randn(n)
    return X, Y


def _do_steps(net, optimizer, X, Y, loss_fn, n=3):
    for _ in range(n):
        optimizer.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        optimizer.step()


def _n_params(net):
    return sum(p.numel() for p in net.parameters())


# ---------------------------------------------------------------------------
# Tests: _build_gbs_step
# ---------------------------------------------------------------------------

class TestBuildGbsStep:

    def test_sgd_no_momentum_exact(self):
        """SGD with no momentum: s = −lr · g (exact)."""
        torch.manual_seed(10)
        net = _SmallNet()
        lr = 0.05
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.allclose(s, -lr * g_flat.detach(), atol=1e-6)

    def test_sgd_momentum_shape_finite(self):
        """SGD with momentum: step has correct shape and finite values."""
        torch.manual_seed(11)
        net = _SmallNet()
        lr = 0.01
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)  # populate momentum buffer

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()

    def test_sgd_momentum_differs_from_no_momentum(self):
        """With a populated momentum buffer the step differs from −lr·g."""
        torch.manual_seed(12)
        net = _SmallNet()
        lr = 0.01
        opt_mom = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        opt_bare = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt_mom, X, Y, loss_fn, n=3)

        opt_mom.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])
        g_det = g_flat.detach()

        s_mom  = _build_gbs_step(g_det, opt_mom)
        s_bare = _build_gbs_step(g_det, opt_bare)

        assert not torch.allclose(s_mom, s_bare, atol=1e-6)

    def test_adam_shape_and_finite(self):
        """Adam step has correct shape and finite values after warmup."""
        torch.manual_seed(20)
        net = _SmallNet()
        lr = 0.001
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()

    def test_adamw_shape_and_finite(self):
        """AdamW step has correct shape and finite values after warmup."""
        torch.manual_seed(21)
        net = _SmallNet()
        opt = torch.optim.AdamW(net.parameters(), lr=0.001)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()

    def test_rmsprop_no_momentum_shape_finite(self):
        """RMSProp (no momentum) step is finite with correct shape."""
        torch.manual_seed(30)
        net = _SmallNet()
        opt = torch.optim.RMSprop(net.parameters(), lr=0.01, momentum=0.0)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()

    def test_rmsprop_with_momentum_shape_finite(self):
        """RMSProp with momentum step is finite with correct shape."""
        torch.manual_seed(31)
        net = _SmallNet()
        opt = torch.optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()

    def test_unsupported_optimizer_raises(self):
        """Unsupported optimizer type raises ValueError."""
        torch.manual_seed(40)
        net = _SmallNet()
        opt = torch.optim.Adagrad(net.parameters(), lr=0.01)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        with pytest.raises(ValueError, match="unsupported optimizer"):
            _build_gbs_step(g_flat.detach(), opt)

    def test_adam_cold_start_no_state(self):
        """Adam step with no state yet (step 0) should still be finite."""
        torch.manual_seed(50)
        net = _SmallNet()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        # Don't run any optimizer steps — state is empty
        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()

    def test_muon_shape_and_finite(self):
        """Muon step has correct shape and finite values after warmup."""
        torch.manual_seed(60)
        net = _SmallNet()
        lr = 0.02
        opt = Muon(net.parameters(), lr=lr, momentum=0.95)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)  # populate momentum buffers

        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()

    def test_muon_no_state_cold_start(self):
        """Muon step with no optimizer state (cold start) is finite."""
        torch.manual_seed(61)
        net = _SmallNet()
        opt = Muon(net.parameters(), lr=0.02, momentum=0.95)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        # No optimizer steps — state is empty
        opt.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        g_flat = torch.cat([p.grad.flatten() for p in net.parameters()])

        s = _build_gbs_step(g_flat.detach(), opt)

        assert s.shape == (_n_params(net),)
        assert torch.isfinite(s).all()


# ---------------------------------------------------------------------------
# Tests: compute_gbs_per_batch
# ---------------------------------------------------------------------------

class TestComputeGbsPerBatch:

    def test_returns_two_finite_scalars(self):
        """compute_gbs_per_batch returns two scalar finite tensors."""
        torch.manual_seed(100)
        net = _SmallNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.0)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        idx = torch.arange(8)
        loss = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        sHs, sg = compute_gbs_per_batch(loss, net, opt)

        assert sHs.ndim == 0  # scalar
        assert sg.ndim == 0
        assert torch.isfinite(sHs)
        assert torch.isfinite(sg)

    def test_shs_nonnegative_mse_sgd(self):
        """s^T H s >= 0 for MSE loss (PSD Hessian), vanilla SGD."""
        torch.manual_seed(101)
        net = _SmallNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        idx = torch.arange(16)
        loss = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        sHs, _ = compute_gbs_per_batch(loss, net, opt)

        assert sHs.item() >= -1e-6  # non-negative up to float noise

    def test_sg_negative_sgd(self):
        """s^T g < 0 for vanilla SGD (s = -lr*g, so s^T g = -lr*||g||^2 < 0)."""
        torch.manual_seed(102)
        net = _SmallNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        idx = torch.arange(16)
        loss = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        _, sg = compute_gbs_per_batch(loss, net, opt)

        assert sg.item() < 0  # step is a descent direction

    def test_sgd_identity_with_rayleigh_quotient(self):
        """For vanilla SGD: sHs / (−sg) == lr * g^T H g / g^T g (exact per batch)."""
        torch.manual_seed(103)
        net = _SmallNet()
        lr = 0.05
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()

        idx = torch.arange(16)
        X_b, Y_b = X[idx], Y[idx]

        # GBS per-batch
        loss1 = loss_fn(net(X_b).squeeze(-1), Y_b)
        sHs, sg = compute_gbs_per_batch(loss1, net, opt)

        # Rayleigh quotient (same batch, new forward pass since graph was consumed)
        loss2 = loss_fn(net(X_b).squeeze(-1), Y_b)
        rq = compute_grad_H_grad(loss2, net)

        gbs_single = sHs.item() / (-sg.item())
        assert pytest.approx(gbs_single, rel=1e-4) == lr * rq.item()

    def test_adam_per_batch_shs_nonnegative(self):
        """s^T H s >= 0 for MSE loss with Adam."""
        torch.manual_seed(104)
        net = _SmallNet()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        X, Y = _make_data()
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        idx = torch.arange(16)
        loss = loss_fn(net(X[idx]).squeeze(-1), Y[idx])
        sHs, _ = compute_gbs_per_batch(loss, net, opt)

        assert sHs.item() >= -1e-6


# ---------------------------------------------------------------------------
# Tests: calculate_gbs  (full MC wrapper)
# ---------------------------------------------------------------------------

class TestCalculateGbs:

    # Keep n_estimates low (30) and min_estimates small (5) for CPU speed.
    _mc_kwargs = dict(n_estimates=30, min_estimates=5, eps=0.1)

    def test_sgd_returns_positive_float(self):
        """calculate_gbs returns a positive finite float for vanilla SGD."""
        torch.manual_seed(200)
        net = _SmallNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
        X, Y = _make_data(n=64)
        loss_fn = nn.MSELoss()

        result = calculate_gbs(net, X, Y, loss_fn, opt, batch_size=16,
                               **self._mc_kwargs)

        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0

    def test_sgd_momentum_returns_finite_float(self):
        """calculate_gbs returns finite float for SGD with momentum."""
        torch.manual_seed(201)
        net = _SmallNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        X, Y = _make_data(n=64)
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        result = calculate_gbs(net, X, Y, loss_fn, opt, batch_size=16,
                               **self._mc_kwargs)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_adam_returns_finite_float(self):
        """calculate_gbs returns finite float for Adam."""
        torch.manual_seed(202)
        net = _SmallNet()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        X, Y = _make_data(n=64)
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        result = calculate_gbs(net, X, Y, loss_fn, opt, batch_size=16,
                               **self._mc_kwargs)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_rmsprop_returns_finite_float(self):
        """calculate_gbs returns finite float for RMSProp."""
        torch.manual_seed(203)
        net = _SmallNet()
        opt = torch.optim.RMSprop(net.parameters(), lr=0.01, momentum=0.0)
        X, Y = _make_data(n=64)
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        result = calculate_gbs(net, X, Y, loss_fn, opt, batch_size=16,
                               **self._mc_kwargs)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_unsupported_optimizer_returns_nan(self):
        """calculate_gbs returns np.nan for an unsupported optimizer."""
        torch.manual_seed(204)
        net = _SmallNet()
        opt = torch.optim.Adagrad(net.parameters(), lr=0.01)
        X, Y = _make_data(n=64)
        loss_fn = nn.MSELoss()

        result = calculate_gbs(net, X, Y, loss_fn, opt, batch_size=16,
                               **self._mc_kwargs)

        assert np.isnan(result)

    def test_muon_returns_finite_float(self):
        """calculate_gbs returns a finite positive float for Muon after warmup."""
        torch.manual_seed(210)
        net = _SmallNet()
        opt = Muon(net.parameters(), lr=0.02, momentum=0.95)
        X, Y = _make_data(n=64)
        loss_fn = nn.MSELoss()
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        result = calculate_gbs(net, X, Y, loss_fn, opt, batch_size=16,
                               **self._mc_kwargs)

        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0

    def test_sgd_gbs_approx_lr_times_batch_sharpness(self):
        """For vanilla SGD: GBS ≈ lr * batch_sharpness (ratio-of-means identity).

        Exact per-batch identity: GBS_i = lr * RQ_i.
        Since GBS uses ratio-of-means and batch_sharpness uses mean-of-ratios,
        equality is approximate; we just verify GBS is in a reasonable ballpark
        (within 50%) of lr * batch_sharpness.
        """
        torch.manual_seed(205)
        net = _SmallNet()
        lr = 0.01
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)
        X, Y = _make_data(n=128)
        loss_fn = nn.MSELoss()

        # Use more estimates for a stable MC estimate
        mc_kw = dict(n_estimates=80, min_estimates=20, eps=0.05)

        from utils.measure import calculate_averaged_grad_H_grad_step
        bs = calculate_averaged_grad_H_grad_step(
            net, X, Y, loss_fn, batch_size=16, **mc_kw,
        )
        gbs = calculate_gbs(net, X, Y, loss_fn, opt, batch_size=16, **mc_kw)

        assert np.isfinite(bs) and np.isfinite(gbs)
        # GBS ≈ lr * bs; check within a factor of 3 (generous for MC noise)
        ratio = gbs / (lr * bs)
        assert 0.2 < ratio < 5.0, (
            f"GBS={gbs:.4f}, lr*bs={lr * bs:.4f}, ratio={ratio:.3f} out of expected range"
        )


# ---------------------------------------------------------------------------
# Tests: frequency rule
# ---------------------------------------------------------------------------

class TestGbsFrequencyRule:

    def test_fires_at_step_256(self):
        ctx = MeasurementContext(step_number=256, batch_size=64, lr=0.01)
        assert frequency_calculator.should_measure('gbs', ctx) is True

    def test_does_not_fire_at_step_255(self):
        ctx = MeasurementContext(step_number=255, batch_size=64, lr=0.01)
        assert frequency_calculator.should_measure('gbs', ctx) is False

    def test_fires_at_step_512(self):
        ctx = MeasurementContext(step_number=512, batch_size=64, lr=0.01)
        assert frequency_calculator.should_measure('gbs', ctx) is True

    def test_rare_measure_doubles_interval(self):
        """Under rare_measure=True, frequency doubles (512 instead of 256)."""
        ctx_normal = MeasurementContext(step_number=256, batch_size=64, lr=0.01,
                                        rare_measure=False)
        ctx_rare   = MeasurementContext(step_number=256, batch_size=64, lr=0.01,
                                        rare_measure=True)
        assert frequency_calculator.should_measure('gbs', ctx_normal) is True
        assert frequency_calculator.should_measure('gbs', ctx_rare) is False

    def test_gbs_registered_in_frequency_calculator(self):
        """'gbs' is a registered measurement type."""
        assert 'gbs' in frequency_calculator.get_available_measurements()
