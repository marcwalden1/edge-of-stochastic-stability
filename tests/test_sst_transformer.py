"""
Tests for SST-2 transformer replication of Damian et al. (arXiv:2209.15594).
Covers: data loading, architecture, loss, training loop integration.
"""
import sys
sys.path.insert(0, '/n/home06/mwalden/edge-of-stochastic-stability')

import torch
import torch.nn as nn
import numpy as np
import pytest

from utils.nets import SSTTransformer, SSTTransformerBlock, LogisticLoss, get_model_presets, prepare_net


# ─────────────────────────────────────────────────────
# 1. SSTTransformer architecture tests
# ─────────────────────────────────────────────────────

def test_sst_transformer_output_shape():
    net = SSTTransformer()
    x = torch.randint(0, 33278, (8, 64))
    out = net(x)
    assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"

def test_sst_transformer_head_zero_init():
    """Head weight and bias must be zero-initialized (matches Damian kernel_init=zeros)."""
    net = SSTTransformer()
    assert torch.all(net.head.weight == 0), "Head weight not zero-initialized"
    assert torch.all(net.head.bias == 0), "Head bias not zero-initialized"

def test_sst_transformer_head_zero_init_means_zero_output():
    """With zero-init head, all outputs should be exactly zero at init."""
    net = SSTTransformer()
    x = torch.randint(0, 33278, (4, 64))
    out = net(x)
    assert torch.all(out == 0), f"Expected all-zero output at init, got max={out.abs().max().item()}"

def test_sst_transformer_no_causal_mask():
    """
    Bidirectional: output should depend on all tokens.
    Verify by checking that shuffling distant tokens changes the output
    (impossible with a causal mask that only sees position 0).
    """
    net = SSTTransformer()
    net.eval()
    torch.manual_seed(0)
    for p in net.parameters():
        nn.init.normal_(p, std=0.1)

    x = torch.randint(0, 33278, (1, 64))
    x_permuted = x.clone()
    x_permuted[0, 1], x_permuted[0, 63] = x_permuted[0, 63].item(), x_permuted[0, 1].item()

    with torch.no_grad():
        out_orig = net(x)
        out_perm = net(x_permuted)

    assert not torch.allclose(out_orig, out_perm, atol=1e-6), \
        "Output unchanged after permuting tokens — possible causal masking bug"

def test_sst_transformer_seq_len_flexibility():
    """Model should handle variable sequence lengths up to its max."""
    net = SSTTransformer(seq_len=64)
    x32 = torch.randint(0, 33278, (2, 32))
    x64 = torch.randint(0, 33278, (2, 64))
    assert net(x32).shape == (2, 1)
    assert net(x64).shape == (2, 1)

def test_sst_transformer_hyperparams_match_damian():
    """Verify exact architecture dimensions match Damian et al."""
    net = SSTTransformer()
    assert net.tok_emb.num_embeddings == 33278
    assert net.tok_emb.embedding_dim == 64
    assert net.pos_emb.num_embeddings == 64
    assert len(net.blocks) == 2
    block = net.blocks[0]
    assert block.attn.num_heads == 2
    assert block.ff1.in_features == 64 and block.ff1.out_features == 64  # no FF expansion
    assert net.head.out_features == 1

def test_sst_transformer_parameter_count():
    net = SSTTransformer()
    n_params = sum(p.numel() for p in net.parameters())
    assert 2_000_000 < n_params < 3_000_000, f"Unexpected param count: {n_params:,}"

def test_sst_transformer_gradients_flow_through_head():
    """
    At zero-init, head.weight gets a gradient (it's the first parameter that sees signal).
    Embeddings and FF layers get zero gradient at step 0 by chain rule (head.weight=0
    means ∂loss/∂(pre-head) = 0), but they receive gradients after any weight update.
    """
    net = SSTTransformer()
    x = torch.randint(0, 33278, (4, 64))
    labels = torch.tensor([1., -1., 1., -1.])
    loss_fn = LogisticLoss()

    # Step 0: head gets gradient, pre-head layers do not (zero-init head)
    out = net(x)
    loss = loss_fn(out, labels)
    loss.backward()
    assert net.head.weight.grad is not None and net.head.weight.grad.norm() > 0, \
        "Head weight should get nonzero gradient at step 0"

    # After one SGD step, head.weight != 0, so gradients flow everywhere
    with torch.no_grad():
        net.head.weight -= 0.1 * net.head.weight.grad

    net.zero_grad()
    out = net(x)
    loss = loss_fn(out, labels)
    loss.backward()

    # Now ALL parameters should have non-zero gradients
    for name, p in net.named_parameters():
        assert p.grad is not None, f"No gradient for {name} after step 1"
        assert not torch.all(p.grad == 0), f"All-zero gradient for {name} after step 1"

def test_preset_matches_damian():
    presets = get_model_presets()
    assert 'sst_transformer' in presets
    p = presets['sst_transformer']['params']
    assert p['vocab_size'] == 33278
    assert p['seq_len'] == 64
    assert p['d_model'] == 64
    assert p['n_heads'] == 2
    assert p['n_layers'] == 2
    assert p['n_classes'] == 1

def test_prepare_net_sst():
    net = prepare_net('sst_transformer', {
        'vocab_size': 33278, 'seq_len': 64, 'd_model': 64,
        'n_heads': 2, 'n_layers': 2, 'n_classes': 1, 'output_dim': 1
    })
    assert isinstance(net, SSTTransformer)


# ─────────────────────────────────────────────────────
# 2. LogisticLoss tests
# ─────────────────────────────────────────────────────

def test_logistic_loss_at_zero_init():
    """At init (all-zero output), logistic loss = log(2) ≈ 0.6931."""
    loss_fn = LogisticLoss()
    out = torch.zeros(4, 1)
    labels = torch.tensor([1., -1., 1., -1.])
    loss = loss_fn(out, labels)
    assert abs(loss.item() - np.log(2)) < 1e-5, f"Expected log(2)={np.log(2):.4f}, got {loss.item():.4f}"

def test_logistic_loss_correct_direction():
    """Loss decreases when predictions align with labels."""
    loss_fn = LogisticLoss()
    labels = torch.tensor([1., -1., 1., -1.])
    good_preds = torch.tensor([[5.], [-5.], [5.], [-5.]])
    bad_preds  = torch.tensor([[-5.], [5.], [-5.], [5.]])
    assert loss_fn(good_preds, labels) < loss_fn(bad_preds, labels)

def test_logistic_loss_handles_squeezed_input():
    """LogisticLoss should accept both (B,1) and (B,) shaped inputs identically."""
    loss_fn = LogisticLoss()
    labels = torch.tensor([1., -1.])
    loss1 = loss_fn(torch.tensor([[0.5], [-0.5]]), labels)
    loss2 = loss_fn(torch.tensor([0.5, -0.5]), labels)
    assert torch.isclose(loss1, loss2), "Loss differs between (B,1) and (B,) inputs"

def test_logistic_loss_matches_damian_formula():
    """L = mean(-log_sigmoid(f * y)) — exact Damian criterion."""
    import torch.nn.functional as F
    loss_fn = LogisticLoss()
    preds  = torch.tensor([[1.2], [-0.8], [0.3]])
    labels = torch.tensor([1., -1., 1.])
    expected = -F.logsigmoid(preds.squeeze(-1) * labels).mean()
    assert torch.isclose(loss_fn(preds, labels), expected)

def test_logistic_loss_differentiable():
    """Loss should be differentiable (gradient check)."""
    loss_fn = LogisticLoss()
    preds  = torch.tensor([[0.5], [-0.3]], requires_grad=True)
    labels = torch.tensor([1., -1.])
    loss = loss_fn(preds, labels)
    loss.backward()
    assert preds.grad is not None and preds.grad.norm() > 0


# ─────────────────────────────────────────────────────
# 3. Dataset preset / dispatcher tests
# ─────────────────────────────────────────────────────

from utils.data import get_dataset_presets

def test_sst2_in_dataset_presets():
    presets = get_dataset_presets()
    assert 'sst2' in presets
    assert presets['sst2']['input_dim'] == 64
    assert presets['sst2']['output_dim'] == 1

def test_prepare_dataset_has_sst2_branch():
    import inspect
    from utils.data import prepare_dataset
    src = inspect.getsource(prepare_dataset)
    assert "sst2" in src and "prepare_sst2" in src


# ─────────────────────────────────────────────────────
# 4. Training loop integration
# ─────────────────────────────────────────────────────

def test_training_loop_squeeze_and_loss():
    """
    Simulate training.py: preds = net(X).squeeze(-1); loss = loss_fn(preds, Y).
    At zero-init: loss = log(2), head.weight gets gradient.
    """
    net = SSTTransformer()
    loss_fn = LogisticLoss()
    X = torch.randint(0, 33278, (8, 64))
    Y = torch.tensor([1., -1., 1., -1., 1., -1., 1., -1.])

    preds = net(X).squeeze(dim=-1)
    assert preds.shape == (8,)
    loss = loss_fn(preds, Y)
    assert abs(loss.item() - np.log(2)) < 1e-4, f"Expected log(2), got {loss.item():.4f}"
    loss.backward()
    assert net.head.weight.grad.norm() > 0

def test_sgd_step_reduces_loss():
    """A few SGD steps should reduce the loss (basic training sanity check)."""
    torch.manual_seed(42)
    net = SSTTransformer()
    loss_fn = LogisticLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    X = torch.randint(0, 33278, (16, 64))
    Y = torch.tensor([1., -1.] * 8)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        preds = net(X).squeeze(-1)
        loss = loss_fn(preds, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ─────────────────────────────────────────────────────
# 5. No regression on existing models / losses
# ─────────────────────────────────────────────────────

def test_existing_gpt_unaffected():
    from utils.nets import GPT
    net = GPT(vocab_size=65, seq_len=128, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    x = torch.randint(0, 65, (2, 128))
    out = net(x)
    assert out.shape == (2 * 128, 65)

def test_existing_language_ce_loss_unaffected():
    from utils.nets import LanguageCELoss
    loss_fn = LanguageCELoss()
    logits  = torch.randn(256, 65)
    targets = torch.randint(0, 65, (2, 128))
    assert loss_fn(logits, targets).item() > 0

def test_existing_squared_loss_unaffected():
    from utils.nets import SquaredLoss
    loss_fn = SquaredLoss()
    preds   = torch.randn(16, 10)
    targets = torch.randn(16, 10)
    assert loss_fn(preds, targets).item() >= 0

def test_existing_model_presets_unchanged():
    presets = get_model_presets()
    for key in ['mlp', 'cnn', 'resnet', 'vit', 'gpt_s', 'gpt', 'gpt_l']:
        assert key in presets, f"Preset '{key}' was removed"

def test_existing_dataset_presets_unchanged():
    presets = get_dataset_presets()
    for key in ['cifar10', 'cifar10_2cls', 'svhn', 'fmnist', 'imagenet32', 'shakespeare']:
        assert key in presets, f"Preset '{key}' was removed"

def test_training_py_wired_correctly():
    """Verify training.py has logistic loss wired in (parse-level check)."""
    with open('/n/home06/mwalden/edge-of-stochastic-stability/training.py') as f:
        src = f.read()
    lines = src.split('\n')
    assert any('LogisticLoss' in l for l in lines[:50]), "LogisticLoss not imported in training.py"
    assert "'logistic'" in src, "--loss choices missing 'logistic'"
    assert "LogisticLoss()" in src, "LogisticLoss() not instantiated in training.py"

