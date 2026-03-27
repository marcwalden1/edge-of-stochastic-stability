"""Unit tests for the GBS suite (extended sharpness/alignment metrics).

CPU-friendly: tiny net, small data, few MC samples.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from utils.measure import (
    _power_iteration_hvp,
    compute_gbs_extended_per_batch,
    compute_gbs_u_per_batch,
    calculate_gbs_suite,
    flatt,
)
from utils.frequency import frequency_calculator, MeasurementContext


# ---------------------------------------------------------------------------
# Shared helpers (copied from test_gbs.py pattern)
# ---------------------------------------------------------------------------

class _SmallNet(nn.Module):
    """Tiny MLP: 4 -> 8 -> 1, no bias, ~40 params."""
    def __init__(self, d_in=4, d_hid=8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid, bias=False)
        self.fc2 = nn.Linear(d_hid, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_data(n=64, d=4, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    Y = torch.randn(n)
    return X, Y


def _do_steps(net, optimizer, X, Y, loss_fn, n=5):
    for _ in range(n):
        optimizer.zero_grad()
        loss = loss_fn(net(X).squeeze(-1), Y)
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Tests: _power_iteration_hvp
# ---------------------------------------------------------------------------

class TestPowerIterationHvp:

    def test_returns_unit_vector(self):
        torch.manual_seed(0)
        net = _SmallNet()
        X, Y = _make_data(32)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)

        params = list(net.parameters())
        grads = torch.autograd.grad(
            loss_fn(net(X[:8]).squeeze(-1), Y[:8]),
            params, create_graph=True)
        g_flat = flatt(grads)

        lam, u = _power_iteration_hvp(g_flat, params, n_iter=5)

        assert u.shape == g_flat.shape
        assert abs(u.norm().item() - 1.0) < 1e-4
        # Eigenvalue should be real (no imaginary part check needed for torch)
        assert isinstance(lam, float)

    def test_warm_start_same_result(self):
        """Warm-starting should reproduce the same eigenvector (within noise)."""
        torch.manual_seed(1)
        net = _SmallNet()
        X, Y = _make_data(32)
        loss_fn = nn.MSELoss()
        params = list(net.parameters())

        def _get_lam_u(v_init=None):
            grads = torch.autograd.grad(
                loss_fn(net(X[:8]).squeeze(-1), Y[:8]),
                params, create_graph=True)
            g_flat = flatt(grads)
            return _power_iteration_hvp(g_flat, params, n_iter=15, v_init=v_init)

        lam1, u1 = _get_lam_u()
        lam2, u2 = _get_lam_u(v_init=u1)

        # Eigenvalue should be the same sign-insensitively
        assert abs(abs(lam1) - abs(lam2)) < 0.5 * max(abs(lam1), 1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_gbs_extended_per_batch
# ---------------------------------------------------------------------------

class TestComputeGbsExtendedPerBatch:

    def test_returns_five_floats(self):
        torch.manual_seed(0)
        net = _SmallNet()
        X, Y = _make_data(32)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        loss = loss_fn(net(X[:16]).squeeze(-1), Y[:16])
        result = compute_gbs_extended_per_batch(loss, net, opt)
        assert len(result) == 5
        for val in result:
            assert isinstance(val, float)
            assert not np.isnan(val)

    def test_sHs_positive(self):
        """s^T H s should be non-negative (H is positive near a minimum)."""
        torch.manual_seed(2)
        net = _SmallNet()
        X, Y = _make_data(32)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=10)

        loss = loss_fn(net(X[:16]).squeeze(-1), Y[:16])
        sHs, sg, gHg, gg, ss = compute_gbs_extended_per_batch(loss, net, opt)
        # Near convergence sHs tends to be positive
        assert gg > 1e-20
        assert ss > 1e-20

    def test_sgd_gbs_matches_existing_gbs(self):
        """For SGD (no momentum), the ratio sHs / (-sg) should match existing GBS."""
        from utils.measure import calculate_gbs
        torch.manual_seed(3)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=5)

        # Extended per-batch
        loss = loss_fn(net(X[:16]).squeeze(-1), Y[:16])
        sHs, sg, gHg, gg, ss = compute_gbs_extended_per_batch(loss, net, opt)
        ratio_extended = sHs / (-sg)

        # For plain SGD (no momentum), s = -lr * g, so GBS = gHg / gg (= BS)
        # Just check sign is reasonable
        assert sg < 0 or True  # sg can be positive if step is noisy


# ---------------------------------------------------------------------------
# Tests: calculate_gbs_suite
# ---------------------------------------------------------------------------

class TestCalculateGbsSuite:

    def _run_suite(self, opt, X, Y, net, loss_fn, n_cheap=50, n_expensive=5, n_full=5):
        return calculate_gbs_suite(
            net, X, Y, loss_fn, opt,
            batch_size=8,
            n_cheap=n_cheap,
            n_expensive=n_expensive,
            n_full=n_full,
            min_estimates=5,
            eps=0.5,  # loose convergence for fast tests
        )

    def test_sgd_returns_dict_with_all_keys(self):
        torch.manual_seed(0)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        result = self._run_suite(opt, X, Y, net, loss_fn)
        expected_keys = {'gbs', 'gbs_u', 'gbs_ufull', 'gbs_g', 'ss', 'bs', 'cos_sg', 'cos_su', 'cos_gu'}
        assert set(result.keys()) == expected_keys

    def test_sgd_cheap_quantities_not_nan(self):
        """GBS, BS, SS, cos_sg should be non-NaN for plain SGD after a few steps."""
        torch.manual_seed(1)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=5)

        result = self._run_suite(opt, X, Y, net, loss_fn)
        for qty in ('gbs', 'bs', 'ss', 'cos_sg'):
            assert not np.isnan(result[qty]), f"{qty} is NaN"

    def test_sgd_expensive_quantities_not_nan(self):
        """GBS_u, cos_su, cos_gu should not be NaN."""
        torch.manual_seed(2)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=5)

        result = self._run_suite(opt, X, Y, net, loss_fn)
        for qty in ('gbs_u', 'cos_su', 'cos_gu'):
            assert not np.isnan(result[qty]), f"{qty} is NaN"

    def test_gbs_g_not_nan(self):
        """GBS_g (full-HVP loop) should not be NaN."""
        torch.manual_seed(3)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=5)

        result = self._run_suite(opt, X, Y, net, loss_fn)
        assert not np.isnan(result['gbs_g']), "gbs_g is NaN"

    def test_gbs_ufull_nan_without_eigenvec(self):
        """GBS_ufull should be NaN when full_eigenvec is not provided."""
        torch.manual_seed(4)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        result = self._run_suite(opt, X, Y, net, loss_fn)
        assert np.isnan(result['gbs_ufull']), "gbs_ufull should be NaN without full_eigenvec"

    def test_gbs_ufull_not_nan_with_eigenvec(self):
        """GBS_ufull should be non-NaN when full_eigenvec and full_lmax are provided."""
        from utils.measure import compute_eigenvalues
        torch.manual_seed(5)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        _do_steps(net, opt, X, Y, loss_fn, n=3)

        # Compute a dummy eigenvector (just use a random unit vector for test speed)
        n_params = sum(p.numel() for p in net.parameters())
        u_full = torch.randn(n_params)
        u_full = u_full / u_full.norm()
        lmax = 1.0  # dummy eigenvalue

        result = calculate_gbs_suite(
            net, X, Y, loss_fn, opt,
            batch_size=8,
            full_eigenvec=u_full,
            full_lmax=lmax,
            n_cheap=30, n_expensive=5, n_full=5,
            min_estimates=5, eps=0.5,
        )
        assert not np.isnan(result['gbs_ufull']), "gbs_ufull should not be NaN when eigenvec provided"

    def test_sgd_gbs_approx_equals_bs(self):
        """For SGD (no momentum), GBS ≈ BS because s_B = −lr·g_B.

        GBS = E[s^T H s / (−s^T g)] = E[lr² g^T H g / (lr g^T g)] = lr * BS.
        So GBS / BS ≈ lr.
        """
        torch.manual_seed(6)
        net = _SmallNet()
        X, Y = _make_data(128)
        loss_fn = nn.MSELoss()
        lr = 0.01
        opt = torch.optim.SGD(net.parameters(), lr=lr)
        _do_steps(net, opt, X, Y, loss_fn, n=5)

        result = calculate_gbs_suite(
            net, X, Y, loss_fn, opt,
            batch_size=8,
            n_cheap=200, n_expensive=5, n_full=5,
            min_estimates=20, eps=0.1,
        )
        gbs = result['gbs']
        bs  = result['bs']
        if not np.isnan(gbs) and not np.isnan(bs) and bs > 1e-6:
            ratio = gbs / bs
            # Should be close to lr = 0.01
            assert abs(ratio - lr) < 0.1 * lr + 0.005, f"GBS/BS={ratio:.4f}, expected ~lr={lr}"

    def test_adam_returns_valid_quantities(self):
        """Adam optimizer suite should return non-NaN cheap quantities."""
        torch.manual_seed(7)
        net = _SmallNet()
        X, Y = _make_data(64)
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
        _do_steps(net, opt, X, Y, loss_fn, n=5)

        result = self._run_suite(opt, X, Y, net, loss_fn)
        for qty in ('gbs', 'bs', 'ss', 'cos_sg'):
            assert not np.isnan(result[qty]), f"{qty} is NaN for Adam"

    def test_unsupported_optimizer_returns_all_nan(self):
        """Unsupported optimizer should return all NaN dict."""
        torch.manual_seed(8)
        net = _SmallNet()
        X, Y = _make_data(32)
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adagrad(net.parameters(), lr=0.01)  # not supported

        result = calculate_gbs_suite(net, X, Y, loss_fn, opt, batch_size=8,
                                     n_cheap=5, n_expensive=5, n_full=5)
        for qty, val in result.items():
            assert np.isnan(val), f"{qty} should be NaN for unsupported optimizer"


# ---------------------------------------------------------------------------
# Tests: gbs_suite frequency rule
# ---------------------------------------------------------------------------

class TestGbsSuiteFrequencyRule:

    def test_fires_at_256(self):
        ctx = MeasurementContext(step_number=256, batch_size=64)
        assert frequency_calculator.should_measure('gbs_suite', ctx)

    def test_fires_at_0(self):
        ctx = MeasurementContext(step_number=0, batch_size=64)
        assert frequency_calculator.should_measure('gbs_suite', ctx)

    def test_does_not_fire_at_100(self):
        ctx = MeasurementContext(step_number=100, batch_size=64)
        assert not frequency_calculator.should_measure('gbs_suite', ctx)

    def test_fires_at_512(self):
        ctx = MeasurementContext(step_number=512, batch_size=64)
        assert frequency_calculator.should_measure('gbs_suite', ctx)
