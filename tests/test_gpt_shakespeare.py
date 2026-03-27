#!/usr/bin/env python3
"""Tests for GPT model and Shakespeare dataset implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from utils.nets import (
    GPT, GPTBlock, LanguageCELoss,
    get_model_presets, prepare_net, initialize_net, initialize_gpt,
)
from utils.data import get_dataset_presets


# ---------------------------------------------------------------------------
# LanguageCELoss
# ---------------------------------------------------------------------------

def test_language_ce_loss_shape():
    print("test_language_ce_loss_shape ... ", end="")
    loss_fn = LanguageCELoss()
    B, S, V = 4, 8, 65
    logits = torch.randn(B * S, V)
    targets = torch.randint(0, V, (B, S))
    loss = loss_fn(logits, targets)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() > 0
    print("OK")


def test_language_ce_loss_values():
    print("test_language_ce_loss_values ... ", end="")
    loss_fn = LanguageCELoss()
    # Perfect predictions should give ~0 loss
    B, S, V = 2, 4, 65
    targets = torch.randint(0, V, (B, S))
    logits = torch.full((B * S, V), -1e9)
    logits[range(B * S), targets.reshape(-1)] = 1e9
    loss = loss_fn(logits, targets)
    assert loss.item() < 1e-3, f"Expected near-zero loss, got {loss.item()}"
    print("OK")


# ---------------------------------------------------------------------------
# GPTBlock
# ---------------------------------------------------------------------------

def test_gpt_block_shape():
    print("test_gpt_block_shape ... ", end="")
    B, S, d = 2, 16, 64
    block = GPTBlock(d_model=d, n_heads=4, d_ff=256, seq_len=S)
    x = torch.randn(B, S, d)
    out = block(x)
    assert out.shape == (B, S, d), f"Expected {(B, S, d)}, got {out.shape}"
    print("OK")


def test_gpt_block_causal_mask():
    """Each position should only attend to positions <= itself."""
    print("test_gpt_block_causal_mask ... ", end="")
    block = GPTBlock(d_model=64, n_heads=4, d_ff=256, seq_len=32)
    assert block.causal_mask.shape == (32, 32)
    # Upper triangle (excluding diagonal) should be True (masked)
    mask = block.causal_mask
    for i in range(32):
        for j in range(32):
            if j > i:
                assert mask[i, j], f"Expected mask[{i},{j}] to be True"
            else:
                assert not mask[i, j], f"Expected mask[{i},{j}] to be False"
    print("OK")


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

def test_gpt_forward_shape():
    print("test_gpt_forward_shape ... ", end="")
    B, S, V = 3, 32, 65
    net = GPT(vocab_size=V, seq_len=S, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    x = torch.randint(0, V, (B, S))
    logits = net(x)
    assert logits.shape == (B * S, V), f"Expected {(B*S, V)}, got {logits.shape}"
    print("OK")


def test_gpt_no_nan():
    print("test_gpt_no_nan ... ", end="")
    net = GPT(vocab_size=65, seq_len=32, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    x = torch.randint(0, 65, (4, 32))
    logits = net(x)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"
    print("OK")


def test_gpt_gradients_flow():
    print("test_gpt_gradients_flow ... ", end="")
    net = GPT(vocab_size=65, seq_len=16, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    loss_fn = LanguageCELoss()
    x = torch.randint(0, 65, (2, 16))
    y = torch.randint(0, 65, (2, 16))
    logits = net(x)
    loss = loss_fn(logits, y)
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    print("OK")


def test_gpt_causal_autoregressive():
    """Output at position i must not depend on positions > i."""
    print("test_gpt_causal_autoregressive ... ", end="")
    torch.manual_seed(42)
    V, S, d = 65, 8, 64
    net = GPT(vocab_size=V, seq_len=S, d_model=d, n_heads=4, n_layers=2, d_ff=256)
    net.eval()

    x = torch.randint(0, V, (1, S))
    x2 = x.clone()
    x2[0, S // 2:] = torch.randint(0, V, (S - S // 2,))  # change second half

    with torch.no_grad():
        out1 = net(x).reshape(1, S, V)
        out2 = net(x2).reshape(1, S, V)

    # First half of positions should be identical
    assert torch.allclose(out1[0, :S//2], out2[0, :S//2], atol=1e-5), \
        "Causal masking broken: positions in first half differ"
    print("OK")


# ---------------------------------------------------------------------------
# Presets and prepare_net
# ---------------------------------------------------------------------------

def test_model_presets_gpt():
    print("test_model_presets_gpt ... ", end="")
    presets = get_model_presets()
    for name in ('gpt_s', 'gpt', 'gpt_l'):
        assert name in presets, f"Missing preset: {name}"
        assert presets[name]['type'] == 'gpt'
        assert 'params' in presets[name]
        p = presets[name]['params']
        for key in ('d_model', 'n_heads', 'n_layers', 'd_ff', 'seq_len'):
            assert key in p, f"Missing key '{key}' in preset '{name}'"
    print("OK")


def test_prepare_net_gpt():
    print("test_prepare_net_gpt ... ", end="")
    presets = get_model_presets()
    for name in ('gpt_s', 'gpt', 'gpt_l'):
        params = dict(presets[name]['params'])
        params['output_dim'] = 65
        params['input_dim'] = 128  # set by training.py, unused by GPT
        net = prepare_net('gpt', params)
        assert isinstance(net, GPT)
        x = torch.randint(0, 65, (2, 128))
        out = net(x)
        assert out.shape == (2 * 128, 65)
    print("OK")


# ---------------------------------------------------------------------------
# initialize_gpt
# ---------------------------------------------------------------------------

def test_initialize_gpt():
    print("test_initialize_gpt ... ", end="")
    net = GPT(vocab_size=65, seq_len=32, d_model=64, n_heads=4, n_layers=4, d_ff=256)
    initialize_gpt(net, scale=0.5)
    # Head std should be ~0.02 * 0.5 = 0.01
    head_std = net.head.weight.std().item()
    assert 0.001 < head_std < 0.1, f"Unexpected head std: {head_std}"
    # Embedding std should be ~0.02
    emb_std = net.tok_emb.weight.std().item()
    assert 0.001 < emb_std < 0.1, f"Unexpected tok_emb std: {emb_std}"
    print("OK")


def test_initialize_net_gpt():
    print("test_initialize_net_gpt ... ", end="")
    net = GPT(vocab_size=65, seq_len=32, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    # Should not raise
    initialize_net(net, scale=0.2, seed=42)
    print("OK")


# ---------------------------------------------------------------------------
# Dataset presets
# ---------------------------------------------------------------------------

def test_shakespeare_dataset_preset():
    print("test_shakespeare_dataset_preset ... ", end="")
    presets = get_dataset_presets()
    assert 'shakespeare' in presets
    assert presets['shakespeare']['input_dim'] == 128
    assert presets['shakespeare']['output_dim'] == 65
    print("OK")


# ---------------------------------------------------------------------------
# prepare_shakespeare (windowing logic, no download needed)
# ---------------------------------------------------------------------------

def test_shakespeare_windowing_logic():
    """Test the non-overlapping window logic directly."""
    print("test_shakespeare_windowing_logic ... ", end="")
    seq_len = 8
    # Simulate tokenized text of length 100
    data = torch.arange(100, dtype=torch.long)

    n_seq = len(data) // (seq_len + 1)
    X = torch.stack([data[i * seq_len: i * seq_len + seq_len] for i in range(n_seq)])
    Y = torch.stack([data[i * seq_len + 1: i * seq_len + seq_len + 1] for i in range(n_seq)])

    assert X.shape == (n_seq, seq_len)
    assert Y.shape == (n_seq, seq_len)
    assert X.dtype == torch.long
    # Y is X shifted by 1
    assert torch.all(Y == X + 1), "Y should be X shifted by 1"
    print("OK")


# ---------------------------------------------------------------------------
# Second-order autograd (Hessian-vector product)
# ---------------------------------------------------------------------------

def test_gpt_second_order_autograd():
    """Check that second-order autograd works (required for lambda_max)."""
    print("test_gpt_second_order_autograd ... ", end="")
    net = GPT(vocab_size=65, seq_len=8, d_model=32, n_heads=2, n_layers=2, d_ff=64)
    loss_fn = LanguageCELoss()
    x = torch.randint(0, 65, (2, 8))
    y = torch.randint(0, 65, (2, 8))

    logits = net(x)
    loss = loss_fn(logits, y)

    # First-order grads with create_graph=True
    params = [p for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Second-order: grad of sum(grads) w.r.t. params
    grad_norm = sum(g.pow(2).sum() for g in grads)
    grads2 = torch.autograd.grad(grad_norm, params)
    assert all(g is not None for g in grads2), "Second-order grad returned None"
    assert all(not torch.isnan(g).any() for g in grads2), "NaN in second-order grads"
    print("OK")


# ---------------------------------------------------------------------------
# Flash/mem-efficient sdp disabled
# ---------------------------------------------------------------------------

def test_flash_sdp_disabled():
    print("test_flash_sdp_disabled ... ", end="")
    # Instantiating GPT should disable flash/mem-efficient sdp
    _net = GPT(vocab_size=65, seq_len=8, d_model=32, n_heads=2, n_layers=1, d_ff=64)
    assert not torch.backends.cuda.flash_sdp_enabled(), \
        "Flash SDP should be disabled after GPT.__init__"
    assert not torch.backends.cuda.mem_efficient_sdp_enabled(), \
        "Mem-efficient SDP should be disabled after GPT.__init__"
    print("OK")


# ---------------------------------------------------------------------------
# End-to-end: loss + backward with all three presets
# ---------------------------------------------------------------------------

def test_gpt_presets_end_to_end():
    print("test_gpt_presets_end_to_end ... ", end="")
    presets = get_model_presets()
    loss_fn = LanguageCELoss()
    for name in ('gpt_s', 'gpt', 'gpt_l'):
        params = dict(presets[name]['params'])
        params['output_dim'] = 65
        params['input_dim'] = 128
        net = prepare_net('gpt', params)
        initialize_net(net, scale=0.2, seed=0)
        S = params['seq_len']
        x = torch.randint(0, 65, (2, S))
        y = torch.randint(0, 65, (2, S))
        logits = net(x)
        loss = loss_fn(logits, y)
        loss.backward()
        assert not torch.isnan(loss), f"NaN loss for preset {name}"
    print("OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_language_ce_loss_shape,
        test_language_ce_loss_values,
        test_gpt_block_shape,
        test_gpt_block_causal_mask,
        test_gpt_forward_shape,
        test_gpt_no_nan,
        test_gpt_gradients_flow,
        test_gpt_causal_autoregressive,
        test_model_presets_gpt,
        test_prepare_net_gpt,
        test_initialize_gpt,
        test_initialize_net_gpt,
        test_shakespeare_dataset_preset,
        test_shakespeare_windowing_logic,
        test_gpt_second_order_autograd,
        test_flash_sdp_disabled,
        test_gpt_presets_end_to_end,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {e}")
            failed.append(t.__name__)

    print()
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
