"""Muon optimizer: MomentUm Orthogonalized by Newton-Schulz.

Reference: Kosson et al. / Keller Jordan (2024).

For 2D+ parameters the momentum update is orthogonalized via quintic
Newton-Schulz iterations before being applied as the weight update.
1D parameters (biases, LayerNorm scales) fall back to plain SGD+momentum.
"""

import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5,
                                 eps: float = 1e-7) -> torch.Tensor:
    """Return an approximately orthogonal matrix near G via Newton-Schulz.

    Applies 5 iterations of the quintic Newton-Schulz update:
        X <- a*X + b*X*X^T*X + c*(X*X^T)^2*X
    with coefficients (a, b, c) chosen to map singular values toward 1.

    Args:
        G: 2-D tensor of shape (r, c) with r <= c (transposed internally if not).
        steps: Number of Newton-Schulz iterations (default 5).
        eps: Small value added to Frobenius norm for numerical stability.

    Returns:
        Tensor of the same shape as G with (approximately) orthonormal rows.
    """
    assert G.ndim == 2, "zeropower_via_newtonschulz5 requires a 2-D tensor"

    # Work with the "tall" orientation (rows >= cols) for numerical reasons;
    # transpose if needed and flip back at the end.
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T  # now shape (c, r) with c >= r

    # Normalize so singular values start near 1.
    X = G / (G.norm(p='fro') + eps)

    # Quintic Newton-Schulz coefficients (from the Muon paper).
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(steps):
        A = X @ X.T           # (r, r)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X


class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-Schulz.

    For parameters with ndim >= 2, the effective gradient (momentum buffer,
    optionally Nesterov) is reshaped to 2-D, orthogonalized via
    Newton-Schulz, then applied as the update.  1-D parameters (biases,
    norm scales) receive a plain momentum update (no orthogonalization).

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default 0.02).
        momentum: Momentum coefficient (default 0.95).
        nesterov: Use Nesterov momentum (default True).
        ns_steps: Newton-Schulz iteration count (default 5).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g)

                if nesterov:
                    g_eff = g + mom * buf
                else:
                    g_eff = buf

                if p.ndim >= 2:
                    shape = p.shape
                    g_2d = g_eff.view(shape[0], -1)
                    update = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                    update = update.view(shape)
                else:
                    update = g_eff

                p.data.add_(update, alpha=-lr)

        return loss
