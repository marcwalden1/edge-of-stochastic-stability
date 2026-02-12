"""Tests for preconditioned lambda_max(P^{-1}H) computation."""

import torch
import torch.nn as nn
import numpy as np
import pytest

from utils.measure import (
    get_preconditioner_diag,
    create_preconditioned_hessian_vector_product,
    compute_preconditioned_eigenvalues,
    create_hessian_vector_product,
    compute_eigenvalues,
    EigenvectorCache,
    param_length,
)


def _make_small_net_and_loss(hidden=8, seed=42):
    """Helper: 2-input, 1-output MLP with MSE loss on small data."""
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(2, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, 1, bias=False),
    )
    X = torch.randn(16, 2)
    Y = torch.randn(16, 1)
    loss_fn = nn.MSELoss()
    return net, X, Y, loss_fn


# ------------------------------------------------------------------ #
# 1. get_preconditioner_diag — RMSProp
# ------------------------------------------------------------------ #
class TestGetPreconditionerDiagRMSProp:
    def test_rmsprop_matches_manual(self):
        net, X, Y, loss_fn = _make_small_net_and_loss()
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-8)

        # One step to populate state
        optimizer.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        optimizer.step()

        p_diag = get_preconditioner_diag(optimizer)
        assert p_diag is not None

        # Manual computation
        eps = 1e-8
        manual_parts = []
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                manual_parts.append((state['square_avg'].sqrt() + eps).flatten().detach())
        manual = torch.cat(manual_parts)

        assert torch.allclose(p_diag, manual, atol=1e-7)


# ------------------------------------------------------------------ #
# 2. get_preconditioner_diag — Adam (bias-corrected)
# ------------------------------------------------------------------ #
class TestGetPreconditionerDiagAdam:
    def test_adam_bias_correction(self):
        net, X, Y, loss_fn = _make_small_net_and_loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        # Two steps
        for _ in range(2):
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()

        p_diag = get_preconditioner_diag(optimizer)
        assert p_diag is not None

        # Manual computation with bias correction
        eps = 1e-8
        beta2 = 0.999
        manual_parts = []
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                step = state['step']
                if isinstance(step, torch.Tensor):
                    step = step.item()
                bias_correction2 = 1 - beta2 ** step
                v_hat = state['exp_avg_sq'] / bias_correction2
                manual_parts.append((v_hat.sqrt() + eps).flatten().detach())
        manual = torch.cat(manual_parts)

        assert torch.allclose(p_diag, manual, atol=1e-7)


# ------------------------------------------------------------------ #
# 3. get_preconditioner_diag — SGD returns None
# ------------------------------------------------------------------ #
class TestGetPreconditionerDiagSGD:
    def test_sgd_returns_none(self):
        net, X, Y, loss_fn = _make_small_net_and_loss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        optimizer.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        optimizer.step()

        assert get_preconditioner_diag(optimizer) is None


# ------------------------------------------------------------------ #
# 4. get_preconditioner_diag — uninitialized returns None
# ------------------------------------------------------------------ #
class TestGetPreconditionerDiagUninitialized:
    def test_uninitialized_returns_none(self):
        net, X, Y, loss_fn = _make_small_net_and_loss()
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
        # No optimizer.step() called
        assert get_preconditioner_diag(optimizer) is None


# ------------------------------------------------------------------ #
# 5. Preconditioned HVP symmetry: v^T A(w) == w^T A(v)
# ------------------------------------------------------------------ #
class TestPreconditionedHVPSymmetry:
    def test_symmetry(self):
        torch.manual_seed(123)
        net, X, Y, loss_fn = _make_small_net_and_loss()
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99)

        optimizer.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        optimizer.step()

        p_diag = get_preconditioner_diag(optimizer)
        p_inv_sqrt = (1.0 / torch.sqrt(p_diag)).clamp(max=1e6)

        # Recompute loss with graph
        optimizer.zero_grad()
        preds = net(X)
        loss = loss_fn(preds, Y)

        matvec = create_preconditioned_hessian_vector_product(loss, net, p_inv_sqrt)

        n = param_length(net)
        v = torch.randn(n)
        w = torch.randn(n)

        Av = matvec(v)
        Aw = matvec(w)

        lhs = torch.dot(v, Aw)
        rhs = torch.dot(w, Av)

        assert torch.allclose(lhs, rhs, atol=1e-4), f"Symmetry violated: {lhs.item()} vs {rhs.item()}"


# ------------------------------------------------------------------ #
# 6. Preconditioned lmax matches explicit dense computation
# ------------------------------------------------------------------ #
class TestPreconditionedLmaxMatchesExplicit:
    def test_matches_dense(self):
        torch.manual_seed(77)
        # Very small net for dense computation
        net = nn.Linear(2, 1, bias=False)
        X = torch.randn(8, 2)
        Y = torch.randn(8, 1)
        loss_fn = nn.MSELoss()

        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99)
        optimizer.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        optimizer.step()

        p_diag = get_preconditioner_diag(optimizer)
        p_inv_sqrt = (1.0 / torch.sqrt(p_diag)).clamp(max=1e6)

        # Compute loss with graph
        optimizer.zero_grad()
        preds = net(X)
        loss = loss_fn(preds, Y)

        # Build dense P^{-1/2} H P^{-1/2}
        n = param_length(net)
        hvp = create_hessian_vector_product(loss, net)

        H_dense = torch.zeros(n, n)
        for i in range(n):
            e_i = torch.zeros(n)
            e_i[i] = 1.0
            H_dense[:, i] = hvp(e_i)

        P_inv_sqrt_diag = torch.diag(p_inv_sqrt)
        M = P_inv_sqrt_diag @ H_dense @ P_inv_sqrt_diag

        eigvals_dense = torch.linalg.eigvalsh(M)
        lmax_dense = eigvals_dense[-1].item()

        # LOBPCG version
        optimizer.zero_grad()
        preds2 = net(X)
        loss2 = loss_fn(preds2, Y)

        lmax_lobpcg = compute_preconditioned_eigenvalues(
            loss2, net, p_inv_sqrt, k=1, max_iterations=200, reltol=1e-4
        )

        assert abs(lmax_lobpcg.item() - lmax_dense) / max(abs(lmax_dense), 1e-8) < 0.05, \
            f"LOBPCG {lmax_lobpcg.item():.6f} vs dense {lmax_dense:.6f}"


# ------------------------------------------------------------------ #
# 7. Identity preconditioner → matches standard lambda_max(H)
# ------------------------------------------------------------------ #
class TestIdentityPreconditioner:
    def test_identity_matches_standard_lmax(self):
        torch.manual_seed(99)
        net, X, Y, loss_fn = _make_small_net_and_loss(hidden=4)

        # Compute standard lmax
        preds = net(X)
        loss = loss_fn(preds, Y)
        cache1 = EigenvectorCache(max_eigenvectors=5)
        lmax_standard = compute_eigenvalues(
            loss, net, k=1, max_iterations=200, reltol=1e-4,
            eigenvector_cache=cache1, return_eigenvectors=False,
        )

        # Compute preconditioned lmax with p_inv_sqrt = ones (identity preconditioner)
        preds2 = net(X)
        loss2 = loss_fn(preds2, Y)
        n = param_length(net)
        p_inv_sqrt = torch.ones(n)

        cache2 = EigenvectorCache(max_eigenvectors=5)
        lmax_preconditioned = compute_preconditioned_eigenvalues(
            loss2, net, p_inv_sqrt, k=1, max_iterations=200, reltol=1e-4,
            eigenvector_cache=cache2, return_eigenvectors=False,
        )

        rel_err = abs(lmax_preconditioned.item() - lmax_standard.item()) / max(abs(lmax_standard.item()), 1e-8)
        assert rel_err < 0.05, \
            f"Identity preconditioner: {lmax_preconditioned.item():.6f} vs standard: {lmax_standard.item():.6f} (rel err {rel_err:.4f})"
