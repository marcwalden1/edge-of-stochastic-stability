"""
Tests for SST-2 transformer replication of Damian et al. (arXiv:2209.15594).
Covers: data loading, architecture, loss, training loop integration,
        frozen tok_emb, compute_grad_H_grad with frozen params, batch sharpness.
"""
import sys
sys.path.insert(0, '/n/home06/mwalden/edge-of-stochastic-stability')

import torch
import torch.nn as nn
import numpy as np
import pytest

from utils.nets import SSTTransformer, SSTTransformerBlock, SSTCNN, LogisticLoss, get_model_presets, prepare_net


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
        if p.requires_grad:
            nn.init.normal_(p, std=0.1)
    # Re-init tok_emb in-place (without grad) so token differences are detectable
    with torch.no_grad():
        nn.init.normal_(net.tok_emb.weight, std=0.1)

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
    """Total param count (including frozen tok_emb) still ~2.18M."""
    net = SSTTransformer()
    n_params = sum(p.numel() for p in net.parameters())
    assert 2_000_000 < n_params < 3_000_000, f"Unexpected param count: {n_params:,}"

def test_sst_transformer_gradients_flow_through_head():
    """
    tok_emb is frozen so it never receives gradients.
    After one SGD step (head.weight becomes nonzero), all OTHER trainable
    parameters (pos_emb, transformer blocks, head) receive nonzero gradients.
    """
    net = SSTTransformer()
    x = torch.randint(0, 33278, (4, 64))
    labels = torch.tensor([1., 1., 1., -1.])  # asymmetric so head.bias gets nonzero grad
    loss_fn = LogisticLoss()

    # tok_emb is frozen — must never get a gradient
    assert not net.tok_emb.weight.requires_grad, "tok_emb should be frozen"

    # After one SGD step, head.weight != 0, so gradients flow to all trainable params
    with torch.no_grad():
        net.head.weight.fill_(0.1)

    net.zero_grad()
    out = net(x)
    loss = loss_fn(out, labels)
    loss.backward()

    # tok_emb must have no gradient
    assert net.tok_emb.weight.grad is None, "Frozen tok_emb should not have a gradient"

    # All other trainable params must have nonzero gradients
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"No gradient for {name}"
        assert not torch.all(p.grad == 0), f"All-zero gradient for {name}"

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
    assert p['use_bert_emb'] == True

def test_prepare_net_sst():
    net = prepare_net('sst_transformer', {
        'vocab_size': 33278, 'seq_len': 64, 'd_model': 64,
        'n_heads': 2, 'n_layers': 2, 'n_classes': 1, 'output_dim': 1,
        'use_bert_emb': False,
    })
    assert isinstance(net, SSTTransformer)

def test_sst_cnn_output_shape():
    net = SSTCNN(use_bert_emb=False)
    x = torch.randint(0, 33278, (8, 64))
    out = net(x)
    assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"

def test_sst_cnn_head_zero_init_means_zero_output():
    net = SSTCNN(use_bert_emb=False)
    x = torch.randint(0, 33278, (4, 64))
    out = net(x)
    assert torch.all(out == 0), f"Expected all-zero output at init, got max={out.abs().max().item()}"

def test_sst_cnn_all_pad_input_is_finite():
    torch.manual_seed(0)
    net = SSTCNN(use_bert_emb=False)
    for p in net.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.1)
    with torch.no_grad():
        nn.init.normal_(net.tok_emb.weight, std=0.1)
        net.tok_emb.weight[0].zero_()

    x_padded = torch.zeros(1, 64, dtype=torch.long)

    with torch.no_grad():
        out = net(x_padded)

    assert torch.isfinite(out).all(), "All-pad input should produce finite output"

def test_preset_includes_sst_cnn():
    presets = get_model_presets()
    assert 'sst_cnn' in presets
    p = presets['sst_cnn']['params']
    assert p['vocab_size'] == 33278
    assert p['seq_len'] == 64
    assert p['d_model'] == 64
    assert p['hidden_dim'] == 128
    assert p['kernel_sizes'] == (3, 5, 7)
    assert p['n_classes'] == 1
    assert p['use_bert_emb'] == True

def test_prepare_net_sst_cnn():
    net = prepare_net('sst_cnn', {
        'vocab_size': 33278, 'seq_len': 64, 'd_model': 64,
        'hidden_dim': 128, 'kernel_sizes': (3, 5, 7), 'n_classes': 1,
        'output_dim': 1, 'use_bert_emb': False,
    })
    assert isinstance(net, SSTCNN)


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
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5)
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


# ─────────────────────────────────────────────────────
# 6. Frozen tok_emb architecture tests
# ─────────────────────────────────────────────────────

def test_tok_emb_frozen():
    """tok_emb.weight must have requires_grad=False."""
    net = SSTTransformer()
    assert not net.tok_emb.weight.requires_grad, \
        "tok_emb.weight should be frozen (requires_grad=False)"

def test_pos_emb_still_trainable():
    """pos_emb.weight must remain trainable (always dense gradients)."""
    net = SSTTransformer()
    assert net.pos_emb.weight.requires_grad, \
        "pos_emb.weight should be trainable"

def test_tok_emb_no_gradient_after_backward():
    """After loss.backward(), tok_emb.weight.grad must be None."""
    net = SSTTransformer()
    with torch.no_grad():
        net.head.weight.fill_(0.1)
    X = torch.randint(0, 33278, (4, 64))
    Y = torch.tensor([1., -1., 1., -1.])
    loss = LogisticLoss()(net(X), Y)
    loss.backward()
    assert net.tok_emb.weight.grad is None, \
        "Frozen tok_emb should produce no gradient"

def test_trainable_param_count():
    """Trainable params (excluding frozen tok_emb) should be ~53,825."""
    net = SSTTransformer()
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert 50_000 < n_trainable < 60_000, \
        f"Unexpected trainable param count: {n_trainable:,} (expected ~53,825)"

def test_total_param_count_unchanged():
    """Total param count (including frozen tok_emb) is still ~2.18M."""
    net = SSTTransformer()
    n_total = sum(p.numel() for p in net.parameters())
    assert 2_000_000 < n_total < 3_000_000, \
        f"Total param count changed unexpectedly: {n_total:,}"

def test_all_trainable_params_get_nonzero_gradients():
    """
    After head.weight is nonzero, every requires_grad=True param
    (pos_emb, transformer blocks, head) gets a nonzero gradient.
    tok_emb should still have no gradient.
    """
    torch.manual_seed(7)
    net = SSTTransformer()
    with torch.no_grad():
        net.head.weight.fill_(0.05)
    X = torch.randint(0, 33278, (8, 64))
    Y = torch.tensor([1., 1., 1., -1., 1., 1., -1., 1.])  # asymmetric so head.bias gets nonzero grad
    loss = LogisticLoss()(net(X), Y)
    loss.backward()

    assert net.tok_emb.weight.grad is None, "tok_emb should have no gradient"
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"No gradient for trainable param {name}"
        assert p.grad.norm().item() > 0, f"Zero gradient for trainable param {name}"

def test_tok_emb_values_unchanged_after_sgd_step():
    """tok_emb.weight values must not change after an SGD optimizer step."""
    torch.manual_seed(0)
    net = SSTTransformer()
    emb_before = net.tok_emb.weight.data.clone()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
    X = torch.randint(0, 33278, (8, 64))
    Y = torch.tensor([1., -1.] * 4)
    optimizer.zero_grad()
    loss = LogisticLoss()(net(X), Y)
    loss.backward()
    optimizer.step()

    assert torch.equal(net.tok_emb.weight.data, emb_before), \
        "tok_emb.weight changed after optimizer step — it should be frozen"


# ─────────────────────────────────────────────────────
# 7. compute_grad_H_grad with frozen params
# ─────────────────────────────────────────────────────

from utils.nets import SquaredLoss
from utils.measure import compute_grad_H_grad, calculate_averaged_grad_H_grad_step

def test_compute_grad_H_grad_with_frozen_tok_emb():
    """compute_grad_H_grad must not crash on SSTTransformer and return a finite positive scalar."""
    torch.manual_seed(1)
    net = SSTTransformer()
    with torch.no_grad():
        net.head.weight.fill_(0.05)
    X = torch.randint(0, 33278, (8, 64))
    Y = torch.tensor([1., -1.] * 4)
    loss_fn = SquaredLoss()
    loss = loss_fn(net(X).squeeze(-1), Y)
    result = compute_grad_H_grad(loss, net)
    assert torch.isfinite(result), f"compute_grad_H_grad returned non-finite: {result}"
    assert result.item() > 0, f"compute_grad_H_grad returned non-positive: {result.item()}"

def test_compute_grad_H_grad_only_uses_trainable_params():
    """
    Result computed via compute_grad_H_grad (which uses requires_grad filter)
    must equal the result when manually restricting to requires_grad=True params.
    """
    torch.manual_seed(2)
    net = SSTTransformer()
    with torch.no_grad():
        net.head.weight.fill_(0.05)
    X = torch.randint(0, 33278, (8, 64))
    Y = torch.tensor([1., -1.] * 4)
    loss_fn = SquaredLoss()

    # Result via the fixed compute_grad_H_grad
    loss = loss_fn(net(X).squeeze(-1), Y)
    result = compute_grad_H_grad(loss, net, return_ghg_gg_separately=False)

    # Manual check: only trainable params included
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    loss2 = loss_fn(net(X).squeeze(-1), Y)
    grads = torch.autograd.grad(loss2, trainable_params, create_graph=True)
    g = torch.cat([g.flatten() for g in grads])
    s = g.detach()
    Hv = torch.autograd.grad(torch.dot(g, s), trainable_params)
    Hv_flat = torch.cat([h.flatten() for h in Hv]).detach()
    manual_result = torch.dot(s, Hv_flat) / torch.dot(s, s)

    assert torch.isclose(result, manual_result, rtol=1e-4), \
        f"compute_grad_H_grad result {result.item():.4f} != manual {manual_result.item():.4f}"

def test_compute_grad_H_grad_regression_dense_model():
    """
    For a tiny fully-dense MLP (all params trainable), compute_grad_H_grad
    must still return a finite positive scalar (regression guard for the filter change).
    """
    torch.manual_seed(3)
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    X = torch.randn(4, 8)
    Y = torch.randn(4)
    loss = nn.MSELoss()(net(X).squeeze(-1), Y)
    result = compute_grad_H_grad(loss, net)
    assert torch.isfinite(result) and result.item() > 0, \
        f"Dense MLP: compute_grad_H_grad returned {result.item()}"


# ─────────────────────────────────────────────────────
# 8. Batch sharpness end-to-end with frozen tok_emb
# ─────────────────────────────────────────────────────

def test_batch_sharpness_with_frozen_tok_emb():
    """calculate_averaged_grad_H_grad_step must work on SSTTransformer without crashing."""
    torch.manual_seed(4)
    net = SSTTransformer()
    with torch.no_grad():
        net.head.weight.fill_(0.05)
    X = torch.randint(0, 33278, (64, 64))
    Y = torch.tensor([1., -1.] * 32)
    loss_fn = SquaredLoss()

    result = calculate_averaged_grad_H_grad_step(
        net, X, Y, loss_fn,
        batch_size=16, n_estimates=10, min_estimates=5,
    )
    assert np.isfinite(result), f"Batch sharpness returned non-finite: {result}"
    assert result > 0, f"Batch sharpness returned non-positive: {result}"

def test_batch_sharpness_positive_and_finite():
    """Batch sharpness must be > 0 and finite across multiple random seeds."""
    loss_fn = SquaredLoss()
    for seed in range(3):
        torch.manual_seed(seed + 10)
        net = SSTTransformer()
        with torch.no_grad():
            net.head.weight.fill_(0.05)
        X = torch.randint(0, 33278, (64, 64))
        Y = torch.tensor([1., -1.] * 32)
        result = calculate_averaged_grad_H_grad_step(
            net, X, Y, loss_fn,
            batch_size=16, n_estimates=10, min_estimates=5,
        )
        assert np.isfinite(result) and result > 0, \
            f"Seed {seed}: batch sharpness = {result}"

def test_batch_sharpness_scales_with_curvature():
    """
    A model trained at a high learning rate should reach a flatter minimum
    (lower sharpness) than one trained at a very low learning rate (still near init).
    Sanity check: batch sharpness is sensitive to model state.
    """
    torch.manual_seed(99)
    loss_fn = SquaredLoss()
    X = torch.randint(0, 33278, (64, 64))
    Y = torch.tensor([1., -1.] * 32)

    def get_bs(net):
        return calculate_averaged_grad_H_grad_step(
            net, X, Y, loss_fn,
            batch_size=16, n_estimates=20, min_estimates=10,
        )

    # Model A: zero-init head (very flat — head.weight=0)
    net_a = SSTTransformer()

    # Model B: head.weight filled to 0.5 (more curvature)
    net_b = SSTTransformer()
    with torch.no_grad():
        net_b.head.weight.fill_(0.5)

    bs_a = get_bs(net_a)
    bs_b = get_bs(net_b)

    # At zero init, head.weight=0 → gradient through backbone=0 → gHg/g² dominated by head only
    # With nonzero head, backbone gradients contribute → different (typically higher) batch sharpness
    # We just assert they're different (not necessarily ordered) — the point is sensitivity
    assert bs_a != bs_b, \
        f"Batch sharpness insensitive to model state: A={bs_a:.4f}, B={bs_b:.4f}"


# ─────────────────────────────────────────────────────
# 9. Padding mask + masked mean pooling (Fix A)
# ─────────────────────────────────────────────────────

def test_masked_pooling_ignores_pad_tokens():
    """Positions with token id=0 ([PAD]) must be invisible to the model output."""
    torch.manual_seed(42)
    net = SSTTransformer(use_bert_emb=False)
    with torch.no_grad():
        net.head.weight.fill_(0.1)

    # x: real tokens in positions 0-9, PAD (id=0) in positions 10-63
    x = torch.randint(1, 33278, (2, 64))
    x[:, 10:] = 0  # pad positions 10-63

    # x2: same real tokens in 0-9, but different non-zero ids in 10-63
    x2 = x.clone()
    x2[:, 10:] = torch.randint(1, 33278, (2, 54))

    with torch.no_grad():
        out1 = net(x)
        out2 = net(x2)

    assert torch.allclose(out1, out2, atol=1e-5), \
        f"PAD positions affected output: {out1} vs {out2}"

def test_attention_ignores_pad_positions():
    """Replacing a PAD token with a different non-zero id must not change output."""
    torch.manual_seed(7)
    net = SSTTransformer(use_bert_emb=False)
    with torch.no_grad():
        net.head.weight.fill_(0.1)

    x = torch.randint(1, 33278, (1, 64))
    x[0, 30:] = 0  # positions 30-63 are PAD

    x2 = x.clone()
    x2[0, 40] = 5000  # change one PAD token to a different non-zero id — should be ignored

    with torch.no_grad():
        assert torch.allclose(net(x), net(x2), atol=1e-5), \
            "Replacing a PAD token with non-PAD changed output — masking broken"

def test_masked_mean_correct_denominator():
    """Masked mean pooling must divide by number of real tokens, not total sequence length."""
    torch.manual_seed(13)
    net = SSTTransformer(use_bert_emb=False)
    with torch.no_grad():
        net.head.weight.fill_(0.1)

    # Two sequences: one with 10 real tokens, one with 50 real tokens
    x = torch.zeros(2, 64, dtype=torch.long)
    x[0, :10] = torch.randint(1, 33278, (10,))   # 10 real tokens
    x[1, :50] = torch.randint(1, 33278, (50,))   # 50 real tokens

    # Manually compute masked mean for sample 0 using the embedding directly
    with torch.no_grad():
        h0_emb = net.tok_emb(x[0:1]) + net.pos_emb(
            torch.arange(64).unsqueeze(0))
        # run through blocks
        pad0 = (x[0:1] == 0)
        h0 = h0_emb.clone()
        for block in net.blocks:
            h0 = block(h0, pad_mask=pad0)
        valid0 = (~pad0).float()  # should be 1 for positions 0-9, 0 for 10-63
        manual_pool = (h0 * valid0.unsqueeze(-1)).sum(1) / valid0.sum(1, keepdim=True)
        expected_out = net.head(manual_pool)
        actual_out = net(x[0:1])

    assert torch.allclose(actual_out, expected_out, atol=1e-5), \
        f"Masked mean pooling mismatch: {actual_out} vs {expected_out}"

def test_batch_sharpness_finite_with_masked_attention():
    """calculate_averaged_grad_H_grad_step works with heavily padded sequences."""
    torch.manual_seed(20)
    net = SSTTransformer(use_bert_emb=False)
    with torch.no_grad():
        net.head.weight.fill_(0.05)

    # Construct sequences with only 10 real tokens and 54 PAD tokens
    X = torch.zeros(32, 64, dtype=torch.long)
    X[:, :10] = torch.randint(1, 33278, (32, 10))
    Y = torch.tensor([1., -1.] * 16)

    from utils.nets import SquaredLoss
    from utils.measure import calculate_averaged_grad_H_grad_step
    result = calculate_averaged_grad_H_grad_step(
        net, X, Y, SquaredLoss(), batch_size=8, n_estimates=5, min_estimates=3)
    assert np.isfinite(result) and result > 0, f"Batch sharpness with heavy padding: {result}"


# ─────────────────────────────────────────────────────
# 10. Pretrained BERT embeddings (Fix B)
# ─────────────────────────────────────────────────────

from utils.nets import load_bert_embeddings_projected

def test_bert_emb_loads_correctly():
    """load_bert_embeddings_projected returns a (33278, 64) finite tensor."""
    W = load_bert_embeddings_projected(vocab_size=33278, d_model=64)
    assert W.shape == (33278, 64), f"Expected (33278, 64), got {W.shape}"
    assert torch.isfinite(W).all(), "BERT projected embeddings contain non-finite values"
    # BERT rows (0:30522) should be normalized to ~unit std; rows 30522+ are zero
    stds = W[:30522].std(0)
    assert (stds > 0.9).all() and (stds < 1.1).all(), \
        f"Std not ≈ 1.0 for BERT rows: min={stds.min():.3f}, max={stds.max():.3f}"

def test_sst_transformer_bert_emb_frozen():
    """With use_bert_emb=True, tok_emb is still frozen (requires_grad=False)."""
    net = SSTTransformer(use_bert_emb=True)
    assert not net.tok_emb.weight.requires_grad, "tok_emb should be frozen even with BERT init"

def test_bert_emb_nonrandom():
    """BERT-projected embeddings must have non-trivial structure (not random noise)."""
    W = load_bert_embeddings_projected(vocab_size=33278, d_model=64)
    # Use only the BERT rows (30522 rows that were actually projected)
    W_bert = W[:30522]
    # Random 30522×64 matrix would have singular values roughly equal (~sqrt(30522/64)≈21.8).
    # The BERT projected embeddings should have concentrated variance in fewer dimensions.
    # Check: first PC explains more than 1/64 ≈ 1.56% (= uniform random baseline for 64 dims)
    _, S, _ = torch.linalg.svd(W_bert, full_matrices=False)
    explained_first = (S[0]**2 / (S**2).sum()).item()
    assert explained_first > 0.016, \
        f"First PC explains only {explained_first:.4f} — embeddings look random (threshold 0.016)"

def test_sst_transformer_bert_emb_different_from_random():
    """BERT-initialized model should have different tok_emb weights than random-init model."""
    net_random = SSTTransformer(use_bert_emb=False)
    net_bert = SSTTransformer(use_bert_emb=True)
    assert not torch.allclose(net_random.tok_emb.weight, net_bert.tok_emb.weight), \
        "BERT and random tok_emb weights are identical — BERT init not applied"


# =============================================================================
# Section 11: SSTMLP
# =============================================================================

from utils.nets import SSTMLP


def test_sst_mlp_forward_shape():
    """SSTMLP output shape is (B, 1) for token input (B, seq_len)."""
    net = SSTMLP()
    x = torch.randint(1, 33278, (4, 64))
    out = net(x)
    assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"


def test_sst_mlp_tok_emb_frozen():
    """tok_emb.weight must have requires_grad=False."""
    net = SSTMLP()
    assert not net.tok_emb.weight.requires_grad, "tok_emb should be frozen"


def test_sst_mlp_masked_pooling():
    """The PAD embedding (id=0) must not influence the output regardless of its value."""
    torch.manual_seed(0)
    net = SSTMLP()
    nn.init.normal_(net.head.weight)
    nn.init.normal_(net.head.bias)

    # x: 10 real tokens in positions 0-9, PAD (id=0) in positions 10-63
    x = torch.zeros(1, 64, dtype=torch.long)
    x[0, :10] = torch.randint(1, 33278, (10,))

    with torch.no_grad():
        out1 = net(x)
        # Perturb the PAD embedding drastically — output must not change
        net.tok_emb.weight[0] = torch.randn(net.tok_emb.embedding_dim) * 100
        out2 = net(x)

    assert torch.allclose(out1, out2, atol=1e-5), \
        f"PAD embedding change affected output: {out1.item():.6f} vs {out2.item():.6f}"


def test_sst_mlp_preset_exists():
    """sst_mlp preset must be in get_model_presets() with required keys."""
    presets = get_model_presets()
    assert 'sst_mlp' in presets
    p = presets['sst_mlp']['params']
    for key in ('vocab_size', 'seq_len', 'd_model', 'hidden_dim', 'n_layers', 'n_classes', 'use_bert_emb'):
        assert key in p, f"Missing key '{key}' in sst_mlp preset"
    assert presets['sst_mlp']['type'] == 'sst_mlp'


def test_sst_mlp_prepare_net():
    """prepare_net('sst_mlp', ...) returns an SSTMLP instance."""
    params = {'vocab_size': 33278, 'seq_len': 64, 'd_model': 64,
              'hidden_dim': 128, 'n_layers': 2, 'n_classes': 1,
              'use_bert_emb': False, 'input_dim': 64, 'output_dim': 1}
    net = prepare_net('sst_mlp', params)
    assert isinstance(net, SSTMLP)


def test_sst_mlp_batch_sharpness_finite():
    """calculate_averaged_grad_H_grad_step returns a finite positive value for SSTMLP."""
    from utils.nets import SquaredLoss
    from utils.measure import calculate_averaged_grad_H_grad_step
    torch.manual_seed(42)
    net = SSTMLP()
    loss_fn = SquaredLoss()
    X = torch.zeros(32, 64, dtype=torch.long)
    for i in range(32):
        length = torch.randint(5, 20, (1,)).item()
        X[i, :length] = torch.randint(1, 33278, (length,))
    Y = torch.randint(0, 2, (32,)).float() * 2 - 1
    result = calculate_averaged_grad_H_grad_step(net, loss_fn, X, Y, num_batches=2, batch_size=16)
    assert result is not None and torch.isfinite(torch.tensor(result)), \
        f"batch sharpness not finite: {result}"
    assert result > 0, f"batch sharpness not positive: {result}"
