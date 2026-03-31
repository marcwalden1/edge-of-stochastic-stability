import torch as T
import torch
import torch.nn as nn
import timm
import math

from utils.resnet_new import resnet14, ResNet
from utils.resnet_bn import resnet10 as resnet10_bn, ResNet as ResNetBN
import torch.nn.functional as F

from einops import rearrange

from utils.data import get_dataset_presets
from typing import Union
from pathlib import Path
from contextlib import contextmanager



def get_model_presets():
    model_presets = {
        'linear': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2
            }
        },
        'linear_s': {
            'type': 'linear',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1,
                'bias': True
            }
        },
        'linear_l': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4
            }
        },
        'lin_tiny': {
            'type': 'linear',
            'params': {
                'hidden_dim': 2,
                'n_layers': 1
            }
        },
        'mlp': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2,
                'activation': 'relu'
            }
        },
        'mlp2': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 2
            }
        },
        'mlp3': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 3
            }
        },
        'mlp_s': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1
            }
        },
        'mlp_l': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4
            }
        },
        'cnn': {
            'type': 'cnn',
            'params': {
                'hidden_dim': 512,
                'activation': 'relu'
            }
        },
        'resnet': {
            'type': 'resnet',
            'params': {},
        },
        'resnet_bn': {
            'type': 'resnet_bn',
            'params': {},
        },
        'vit': {
            'type': 'vit',
            'params': {
                'img_size': 32,
                'patch_size': 4,
                'embed_dim': 64,
                'depth': 2,
                'num_heads': 2,
                'mlp_ratio': 4.0,
            }
        },
        'gpt_s': {
            'type': 'gpt',
            'params': {'d_model': 64, 'n_heads': 4, 'n_layers': 4, 'd_ff': 256, 'seq_len': 128},
        },
        'gpt': {
            'type': 'gpt',
            'params': {'d_model': 128, 'n_heads': 4, 'n_layers': 6, 'd_ff': 512, 'seq_len': 128},
        },
        'gpt_l': {
            'type': 'gpt',
            'params': {'d_model': 192, 'n_heads': 6, 'n_layers': 8, 'd_ff': 768, 'seq_len': 128},
        },
        'sst_transformer': {
            'type': 'sst_transformer',
            'params': {
                'vocab_size': 33278,
                'seq_len': 64,
                'd_model': 64,
                'n_heads': 2,
                'n_layers': 2,
                'n_classes': 1,
                'use_bert_emb': True,
            }
        },
        'sst_mlp': {
            'type': 'sst_mlp',
            'params': {
                'vocab_size': 33278,
                'seq_len': 64,
                'd_model': 64,
                'hidden_dim': 128,
                'n_layers': 2,
                'n_classes': 1,
                'use_bert_emb': True,
            }
        },
        'sst_cnn': {
            'type': 'sst_cnn',
            'params': {
                'vocab_size': 33278,
                'seq_len': 64,
                'd_model': 64,
                'hidden_dim': 128,
                'kernel_sizes': (3, 5, 7),
                'n_classes': 1,
                'use_bert_emb': True,
            }
        },
        'sst_lstm': {
            'type': 'sst_lstm',
            'params': {
                'vocab_size': 33278,
                'seq_len': 64,
                'd_model': 64,
                'hidden_dim': 128,
                'n_layers': 1,
                'n_classes': 1,
                'use_bert_emb': True,
            }
        },
        'sst_ssm': {
            'type': 'sst_ssm',
            'params': {
                'vocab_size': 33278,
                'seq_len': 64,
                'd_model': 64,
                'd_inner': 64,
                'd_state': 16,
                'n_classes': 1,
                'use_bert_emb': True,
            }
        },
    }
    return model_presets





class SquaredLoss(nn.modules.loss._Loss):
    '''
    Basically MSE, but doesn't average over the dimensions.
    With added support for sampling_vector (aka weighting of the samples, aka mask) the samples! 
    Used to do GD with noise to simulate SGD
    '''
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 ) -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: T.Tensor, target: T.Tensor,
                sampling_vector: T.Tensor = None,
                reduction: str = None
                ) -> T.Tensor:
        '''
        THE MASK NEEDS TO BE "NORMALIZED" - i.e. with expected value of 1/n*unit_vector, 
        and NOT just unit_vector, without the normalization
        this is because without mask there is averaging happening!
        Inherently, this is to perform Jacobian-vector products
        '''

        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape for the loss to operate as expected.\nDid you forget to squeeze the output?")
        

        if sampling_vector is not None:
            total_L2 = F.mse_loss(input, target, reduction='none') 
            
            if len(target.shape) != 1:
                loss_per_sample = total_L2.sum(dim=-1) # shape = (batch_size,)
            else:
                loss_per_sample = total_L2

            assert len(loss_per_sample.shape) == 1
            sampled_loss = T.dot(loss_per_sample, sampling_vector) # L \dot \omega - where \omega is the sampling vector
            return sampled_loss
        
        # if len(target.shape) != 1:
        #     # used to 
        #     multiplier = input.size(-1)
        # else:
        #     multiplier = 1.
        
        total_L2 =  F.mse_loss(input, target, reduction='none') # shape = (batch_size, num_classes)
        if len(target.shape) != 1:
            loss_per_sample = total_L2.sum(dim=-1) # shape = (batch_size,)
        else:
            loss_per_sample = total_L2

        
        if reduction is not None:
            if reduction == 'none':
                return loss_per_sample

            raise ValueError(f"Are you sure you want to use reduction={reduction}? Double-check what you doing - maybe use self.reduction variable at __init__ instead?\n")
        
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        if self.reduction == 'sum':
            return loss_per_sample.sum()
        # we are not introducing this just as a safety
        # if self.reduction == 'none':
        #     return loss_per_sample
        
        raise ValueError("Unknown reduction type")



class LanguageCELoss(nn.Module):
    def forward(self, input, target):
        # input: (B*T, vocab), target: (B, T) -> (B*T,)
        return F.cross_entropy(input, target.reshape(-1))


class LogisticLoss(nn.Module):
    """Binary logistic loss for {-1, +1} labels.

    Replicates Damian et al. (arXiv:2209.15594) SST-2 criterion:
        L = mean(-log_sigmoid(output * label))
    where output is (B, 1) and label is (B,) float {-1, +1}.
    """
    def forward(self, input, target):
        # input: (B, 1) or (B,), target: (B,) float {-1, +1}
        return -F.logsigmoid(input.squeeze(-1) * target).mean()


_BERT_PROJ_CACHE_PATHS = [
    "/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/bert_emb_proj64.pt",  # Harvard
    "~/bert_emb_proj64.pt",                                             # MIT / other clusters
]

def load_bert_embeddings_projected(vocab_size=33278, d_model=64):
    """Load bert-base-uncased word embeddings projected to d_model dims via SVD.

    Loads from a precomputed cache file if available (fast). Otherwise computes
    from scratch: SVD of BERT's 30522×768 embedding matrix, top-d_model components,
    normalized to zero mean / unit std. Remaining vocab rows (30522:vocab_size) are zero
    (those token ids never appear in SST-2 data).

    Returns a (vocab_size, d_model) tensor, detached, ready to copy into Embedding.weight.
    """
    import os
    if d_model == 64 and vocab_size == 33278:
        for cache_path in _BERT_PROJ_CACHE_PATHS:
            cache_path = os.path.expanduser(cache_path)
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=True)
    from transformers import AutoModel
    model = AutoModel.from_pretrained("bert-base-uncased")
    W = model.embeddings.word_embeddings.weight.data.float().cpu()  # (30522, 768)
    bert_vocab = W.shape[0]
    del model
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    W_proj = U[:, :d_model] * S[:d_model].unsqueeze(0)   # (30522, d_model)
    W_proj = (W_proj - W_proj.mean(0)) / W_proj.std(0).clamp(min=1e-6)
    out = torch.zeros(vocab_size, d_model)
    out[:bert_vocab] = W_proj
    out[0].zero_()  # Treat token id 0 as PAD so masked positions contribute nothing.
    return out.detach()


class SSTTransformerBlock(nn.Module):
    """Bidirectional (encoder-style) transformer block with post-norm.

    Matches Damian et al. (arXiv:2209.15594) models/transformer.py exactly:
      x = LayerNorm(x + MultiHeadAttn(x))          # full attention, no causal mask
      x = LayerNorm(x + Linear(GELU(Linear(x))))   # FF with no hidden expansion
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Disable flash/efficient attention so second-order gradients work
        T.backends.cuda.enable_flash_sdp(False)
        T.backends.cuda.enable_mem_efficient_sdp(False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model, bias=False)
        self.ff2 = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        # Full bidirectional attention — no causal mask
        # pad_mask: (B, S) bool, True = PAD position to ignore (PyTorch key_padding_mask convention)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff2(F.gelu(self.ff1(x))))
        return x


class SSTTransformer(nn.Module):
    """Small bidirectional transformer classifier for SST-2.

    Replicates Damian et al. (arXiv:2209.15594) models/transformer.py exactly:
      - vocab_size=33278 (bert-base-uncased), d_model=64, n_heads=2, n_layers=2, seq_len=64
      - Token + positional embeddings (both learned)
      - n_layers SSTTransformerBlocks (bidirectional, post-norm)
      - Mean pool over sequence -> linear head (zero-initialized weight)
    Intended for use with LogisticLoss and {-1, +1} labels.
    """
    def __init__(self, vocab_size=33278, seq_len=64, d_model=64, n_heads=2, n_layers=2, n_classes=1, use_bert_emb=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_bert_emb:
            self.tok_emb.weight.data.copy_(load_bert_embeddings_projected(vocab_size, d_model))
        self.tok_emb.weight.requires_grad_(False)  # frozen: sparse per-batch gradients would break batch sharpness
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([SSTTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, n_classes, bias=True)
        # Zero-initialize head weight and bias (matches Damian's kernel_init=zeros)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, S) long token ids; id=0 is [PAD]
        B, S = x.shape
        pad_mask = (x == 0)  # (B, S), True for [PAD] positions
        pos = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, pad_mask=pad_mask)
        # masked mean pool — only over real (non-PAD) positions
        valid = (~pad_mask).float()  # (B, S)
        h = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(h)  # (B, n_classes)


class SSTMLP(nn.Module):
    """Bag-of-words MLP classifier for SST-2.

    Frozen BERT-projected tok_emb → masked mean pool → 2-layer ReLU MLP → linear head.
    Simpler alternative to SSTTransformer for comparing EoS dynamics across architectures.
    """
    def __init__(self, vocab_size=33278, seq_len=64, d_model=64,
                 hidden_dim=128, n_layers=2, n_classes=1, use_bert_emb=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_bert_emb:
            self.tok_emb.weight.data.copy_(load_bert_embeddings_projected(vocab_size, d_model))
        self.tok_emb.weight.requires_grad_(False)
        layers = []
        in_dim = d_model
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_classes, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, S) long token ids; id=0 is [PAD]
        B, S = x.shape
        pad_mask = (x == 0)               # (B, S), True for [PAD] positions
        h = self.tok_emb(x)               # (B, S, d_model)
        valid = (~pad_mask).float()        # (B, S)
        h = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        h = self.mlp(h)
        return self.head(h)               # (B, n_classes)


class SSTCNN(nn.Module):
    """Text CNN classifier for SST-2 using frozen BERT-projected token embeddings."""
    def __init__(self, vocab_size=33278, seq_len=64, d_model=64,
                 hidden_dim=128, kernel_sizes=(3, 5, 7), n_classes=1,
                 use_bert_emb=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_bert_emb:
            self.tok_emb.weight.data.copy_(load_bert_embeddings_projected(vocab_size, d_model))
        self.tok_emb.weight.requires_grad_(False)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, hidden_dim, kernel_size=k, bias=True, padding=k // 2)
            for k in kernel_sizes
        ])
        self.head = nn.Linear(hidden_dim * len(kernel_sizes), n_classes, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, S) long token ids; id=0 is [PAD]
        pad_mask = (x == 0)
        h = self.tok_emb(x).transpose(1, 2)  # (B, d_model, S)
        valid = (~pad_mask).unsqueeze(1)     # (B, 1, S)
        pooled = []
        for conv in self.convs:
            z = F.relu(conv(h))
            # Exclude padded positions from max pooling.
            z = z.masked_fill(~valid, torch.finfo(z.dtype).min)
            all_pad = (~valid).all(dim=-1, keepdim=True)
            z = torch.where(all_pad, torch.zeros_like(z), z)
            pooled.append(z.max(dim=-1).values)
        return self.head(torch.cat(pooled, dim=-1))


class SSTLSTM(nn.Module):
    """LSTM classifier for SST-2 with frozen BERT-projected token embeddings.

    Frozen tok_emb → LSTM → masked mean pool over non-PAD positions → linear head.
    No LayerNorm, no dropout. Zero-init head for EoS analysis.
    """
    def __init__(self, vocab_size=33278, seq_len=64, d_model=64,
                 hidden_dim=128, n_layers=1, n_classes=1, use_bert_emb=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_bert_emb:
            self.tok_emb.weight.data.copy_(load_bert_embeddings_projected(vocab_size, d_model))
        self.tok_emb.weight.requires_grad_(False)
        self.lstm = nn.LSTM(d_model, hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_classes, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, S) long token ids; id=0 is [PAD]
        pad_mask = (x == 0)                                        # (B, S)
        h = self.tok_emb(x)                                        # (B, S, d_model)
        out, _ = self.lstm(h)                                      # (B, S, hidden_dim)
        valid = (~pad_mask).float()                                 # (B, S)
        pooled = (out * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(pooled)                                    # (B, n_classes)


class SSTSSM(nn.Module):
    """Minimal selective state-space model (Mamba-inspired) for SST-2 classification.

    Pure PyTorch implementation for full autograd / HVP compatibility.
    No LayerNorm, no dropout. Zero-init head for EoS analysis.

    Per time step t:
      u_t  = in_proj(x_t)                          (B, d_inner)
      Δ_t  = softplus(dt_proj(x_proj[Δ](u_t)))     (B, d_inner)  input-dependent step
      B_t  = x_proj[B](u_t)                        (B, d_state)  selective input
      C_t  = x_proj[C](u_t)                        (B, d_state)  selective output
      A    = -exp(A_log)                            (d_inner, d_state)  stable diagonal
      dA_t = exp(Δ_t[...,None] * A)               ZOH discretisation
      dB_t = Δ_t[...,None] * B_t[:,None,:]
      h_t  = dA_t * h_{t-1} + dB_t * u_t[...,None]
      y_t  = (C_t[:,None,:] * h_t).sum(-1) + D * u_t
      out_t = out_proj(y_t)                        (B, d_model)
    Pool out_t over non-PAD positions → head.
    """
    def __init__(self, vocab_size=33278, seq_len=64, d_model=64,
                 d_inner=64, d_state=16, n_classes=1, use_bert_emb=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_bert_emb:
            self.tok_emb.weight.data.copy_(load_bert_embeddings_projected(vocab_size, d_model))
        self.tok_emb.weight.requires_grad_(False)

        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, d_inner, bias=False)
        # x_proj produces [Δ (1), B (d_state), C (d_state)] per token
        self.x_proj   = nn.Linear(d_inner, 1 + 2 * d_state, bias=False)
        # dt_proj expands scalar Δ to d_inner channels
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        # A_log: log-spaced negative eigenvalues, shape (d_inner, d_state)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A_init.clone()))
        # D: skip connection (one per d_inner channel)
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.head = nn.Linear(d_model, n_classes, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Initialise dt_proj bias so softplus(bias) ≈ 0.1
        nn.init.constant_(self.dt_proj.bias, math.log(math.expm1(0.1)))

    def forward(self, x):
        # x: (B, S) long token ids; id=0 is [PAD]
        B, S = x.shape
        pad_mask = (x == 0)                                        # (B, S)
        emb = self.tok_emb(x)                                      # (B, S, d_model)

        A = -torch.exp(self.A_log)                                 # (d_inner, d_state)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=emb.dtype)

        outputs = []
        for t in range(S):
            u = self.in_proj(emb[:, t, :])                         # (B, d_inner)
            xz = self.x_proj(u)                                    # (B, 1+2*d_state)
            dt_raw, B_t, C_t = xz.split([1, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt_raw))                  # (B, d_inner)
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))      # (B, d_inner, d_state)
            dB = dt.unsqueeze(-1) * B_t.unsqueeze(1)               # (B, d_inner, d_state)
            h = dA * h + dB * u.unsqueeze(-1)                      # (B, d_inner, d_state)
            y = (C_t.unsqueeze(1) * h).sum(-1) + self.D * u        # (B, d_inner)
            outputs.append(self.out_proj(y))                        # (B, d_model)

        out = torch.stack(outputs, dim=1)                          # (B, S, d_model)
        valid = (~pad_mask).float()                                 # (B, S)
        pooled = (out * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(pooled)                                    # (B, n_classes)


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len, use_layer_norm=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # LayerNorm omitted by default for EoS sharpness analysis; pass use_layer_norm=True to restore
        self.ln1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

        # Causal mask registered as buffer for device safety
        mask = T.triu(T.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        # Additive causal mask (NOT is_causal=True, to avoid flash dispatch)
        qkv = self.qkv_proj(self.ln1(x)).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)
        # Additive causal mask (NOT is_causal=True, to avoid flash dispatch)
        attn = attn.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.out_proj(out)

        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, n_layers, d_ff, use_layer_norm=False):
        super().__init__()
        # Disable efficient/flash attention so second-order gradients work
        T.backends.cuda.enable_flash_sdp(False)
        T.backends.cuda.enable_mem_efficient_sdp(False)
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, seq_len, use_layer_norm=use_layer_norm)
            for _ in range(n_layers)
        ])
        # LayerNorm omitted by default for EoS sharpness analysis; pass use_layer_norm=True to restore
        self.ln_f = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # x: (B, S) long
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))  # (B, S, vocab_size)
        return logits.reshape(B * S, -1)  # (B*S, vocab_size)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, activation: str = 'relu'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        self.activation = activation

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'silu':
                x = F.silu(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
        x = self.layers[-1](x)
        return x

    def __repr__(self):
        return f"MLP({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim})"


class CNN(nn.Module):
    def __init__(self, fc_hidden_dim, output_dim, activation: str = 'relu'):
        super(CNN, self).__init__()
        self.fc_hidden_dim = fc_hidden_dim
        self.activation = activation
        # self.conv1 = nn.Conv2d(3, 64, 3, 1)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1)
        act = nn.ReLU() if self.activation == 'relu' else nn.SiLU() if self.activation == 'silu' else None
        if act is None:
            raise ValueError(f"Unsupported activation: {self.activation}")
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1), # 64*30*30
            act,
            nn.Conv2d(64, 64, 3, 1), # 64*28*28
            act,
            nn.MaxPool2d(2, 2), # 64, 14

            nn.Conv2d(64, 128, 3, 1), # 128, 12
            act,
            nn.MaxPool2d(2, 2), # 128, 6
        )
        self.fcs = nn.Sequential(
            nn.Linear(128*6*6, fc_hidden_dim, bias=True),
            act,
            nn.Linear(fc_hidden_dim, output_dim, bias=True)
        )
        # self.fc1 = nn.Linear(128*6*6, width, bias=False)
        # self.fc2 = nn.Linear(width, 1, bias=False)
        # self.apply(_weights_init)

    def forward(self, x):
        x = self.convs(x)
        x = rearrange(x, 'b c w h -> b (c w h)')
        x = self.fcs(x)
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=6,
                 num_heads=3, mlp_ratio=4.0, num_classes=10, use_layer_norm=False):
        super().__init__()
        # Disable efficient/flash attention so second-order gradients work
        T.backends.cuda.enable_flash_sdp(False)
        T.backends.cuda.enable_mem_efficient_sdp(False)
        # LayerNorm omitted by default for EoS sharpness analysis; pass use_layer_norm=True to restore
        norm_layer = nn.LayerNorm if use_layer_norm else nn.Identity
        self.model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            norm_layer=norm_layer,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, bias=True):
        super(Linear, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"Linear({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim})"


def prepare_net(model_type: str,
                params: dict
                ):
    if model_type == 'linear':
        net = Linear(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'], params['bias'])
    
    if model_type == 'mlp':
        net = MLP(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'], activation=params.get('activation','relu'))
    
    if model_type == 'cnn':
        net = CNN(params['hidden_dim'], params['output_dim'], activation=params.get('activation','relu'))
    
    if model_type == 'resnet':
        net = resnet14()
    
    if model_type == 'resnet_bn':
        raise "Not implemented - you are still using old resnet_bn"
        net = resnet10_bn()

    if model_type == 'vit':
        net = ViT(
            img_size=params.get('img_size', 32),
            patch_size=params.get('patch_size', 4),
            embed_dim=params.get('embed_dim', 192),
            depth=params.get('depth', 6),
            num_heads=params.get('num_heads', 3),
            mlp_ratio=params.get('mlp_ratio', 4.0),
            num_classes=params['output_dim'],
            use_layer_norm=params.get('use_layer_norm', False),
        )

    if model_type == 'gpt':
        net = GPT(
            vocab_size=params['output_dim'],
            seq_len=params.get('seq_len', 128),
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            d_ff=params['d_ff'],
            use_layer_norm=params.get('use_layer_norm', False),
        )

    if model_type == 'sst_transformer':
        net = SSTTransformer(
            vocab_size=params.get('vocab_size', 33278),
            seq_len=params.get('seq_len', 64),
            d_model=params.get('d_model', 64),
            n_heads=params.get('n_heads', 2),
            n_layers=params.get('n_layers', 2),
            n_classes=params.get('n_classes', 1),
            use_bert_emb=params.get('use_bert_emb', False),
        )

    if model_type == 'sst_mlp':
        net = SSTMLP(
            vocab_size=params.get('vocab_size', 33278),
            seq_len=params.get('seq_len', 64),
            d_model=params.get('d_model', 64),
            hidden_dim=params.get('hidden_dim', 128),
            n_layers=params.get('n_layers', 2),
            n_classes=params.get('n_classes', 1),
            use_bert_emb=params.get('use_bert_emb', False),
        )

    if model_type == 'sst_cnn':
        net = SSTCNN(
            vocab_size=params.get('vocab_size', 33278),
            seq_len=params.get('seq_len', 64),
            d_model=params.get('d_model', 64),
            hidden_dim=params.get('hidden_dim', 128),
            kernel_sizes=params.get('kernel_sizes', (3, 5, 7)),
            n_classes=params.get('n_classes', 1),
            use_bert_emb=params.get('use_bert_emb', False),
        )

    if model_type == 'sst_lstm':
        net = SSTLSTM(
            vocab_size=params.get('vocab_size', 33278),
            seq_len=params.get('seq_len', 64),
            d_model=params.get('d_model', 64),
            hidden_dim=params.get('hidden_dim', 128),
            n_layers=params.get('n_layers', 1),
            n_classes=params.get('n_classes', 1),
            use_bert_emb=params.get('use_bert_emb', False),
        )

    if model_type == 'sst_ssm':
        net = SSTSSM(
            vocab_size=params.get('vocab_size', 33278),
            seq_len=params.get('seq_len', 64),
            d_model=params.get('d_model', 64),
            d_inner=params.get('d_inner', 64),
            d_state=params.get('d_state', 16),
            n_classes=params.get('n_classes', 1),
            use_bert_emb=params.get('use_bert_emb', False),
        )

    return net

def prepare_net_dataset_specific(model_name: str,
                                 dataset: str,
                                 ):
    '''
    Returns the model specific to the procided dataset
    '''
    model_presets = get_model_presets()
    params = model_presets[model_name]['params']
    model_type = model_presets[model_name]['type']

    dataset_presets = get_dataset_presets()
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']

    net = prepare_net(model_type, params)

    return net

    

def initialize_mlp(net, scale=None):
    if scale is None:
        scale=1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data = m.weight.data * scale
            nn.init.zeros_(m.bias)


def initialize_cnn(net, scale=None):
    if scale is None:
        scale = 1.0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def initialize_resnet_old(net, scale=0.01):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.weight.data *= 0.1
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    net.fc.weight.data *= scale

def initialize_resnet(net, scale=None):
    if scale is None:
        scale = 0.01
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, torch.nn.Linear):
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    T.nn.init.normal_(net.linear.weight, std=scale)


def initialize_resnet_bn(net, scale=0.1):
    for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    # initialize fc layer
    nn.init.kaiming_normal_(net.fc.weight, mode='fan_out', nonlinearity='relu')
    # self.fc.weight.data.mul_(0.1)
    nn.init.constant_(net.fc.bias, 0)

    # # Zero-initialize the last BN in each residual branch,
    # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    # for m in net.modules():
    #     if isinstance(m, BasicBlock):
    #         pass
    #         nn.init.constant_(m.bn2.weight, 0)

    # custom scale
    net.fc.weight.data *= scale


def initialize_gpt(net, scale=None, residual_scaling=False):
    if scale is None:
        scale = 1.0
    # scale < 1 reduces initial sharpness to allow progressive sharpening toward EoS.
    # residual_scaling applies GPT-2 style 1/sqrt(2*n_layers) to residual output projections.
    std_base = 0.02 * scale
    std_residual = std_base / math.sqrt(2 * net.n_layers) if residual_scaling else std_base
    nn.init.normal_(net.tok_emb.weight, std=std_base)
    nn.init.normal_(net.pos_emb.weight, std=std_base)
    for block in net.blocks:
        nn.init.normal_(block.qkv_proj.weight, std=std_base)
        nn.init.normal_(block.ff1.weight, std=std_base)
        # Residual output projections (out_proj, ff2) optionally get depth scaling
        nn.init.normal_(block.out_proj.weight, std=std_residual)
        nn.init.normal_(block.ff2.weight, std=std_residual)
    nn.init.normal_(net.head.weight, std=std_base)


def initialize_vit(net, scale=1.0, residual_scaling=False):
    # scale < 1 reduces initial sharpness to allow progressive sharpening toward EoS.
    # residual_scaling applies GPT-2 style 1/sqrt(2*depth) to residual output projections
    # (attn.proj and mlp.fc2), keeping signal O(1) through all layers at init.
    std_base = 0.02 * scale
    std_residual = std_base / math.sqrt(2 * len(net.model.blocks)) if residual_scaling else std_base

    for m in net.model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, std=std_base)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    if residual_scaling:
        for block in net.model.blocks:
            nn.init.normal_(block.attn.proj.weight, std=std_residual)
            nn.init.normal_(block.mlp.fc2.weight, std=std_residual)

    if hasattr(net.model, 'pos_embed') and net.model.pos_embed is not None:
        nn.init.normal_(net.model.pos_embed, std=std_base)
    if hasattr(net.model, 'cls_token') and net.model.cls_token is not None:
        nn.init.normal_(net.model.cls_token, std=std_base)


def freeze_layernorm(net):
    """Freeze LayerNorm weight/bias (requires_grad=False) so LN normalizes without learning."""
    for m in net.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = False


def initialize_linear(net, scale=None):
    if scale is None:
        scale=1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data = m.weight.data * scale
            nn.init.zeros_(m.bias)


@contextmanager
def temp_seed(seed):
    '''
    Temporarily sets the seed for the random number generator
    This is a context
    '''
    if seed is None:
        yield
        return
        
    state = T.get_rng_state()
    T.manual_seed(seed)
    if T.cuda.is_available():
        cuda_state = T.cuda.get_rng_state()
        T.cuda.manual_seed(seed)
    
    try:
        yield
    finally:
        T.set_rng_state(state)
        if T.cuda.is_available():
            T.cuda.set_rng_state(cuda_state)


def initialize_net(net, scale=None, seed=None, residual_scaling=False):

    with temp_seed(seed):
        if isinstance(net, Linear):
            initialize_linear(net, scale=scale)
        elif isinstance(net, MLP):
            initialize_mlp(net, scale=scale)
        elif isinstance(net, ResNet):
            initialize_resnet(net, scale=scale)
        elif isinstance(net, ResNetBN):
            initialize_resnet_bn(net, scale=scale)
        elif isinstance(net, CNN):
            initialize_cnn(net, scale=scale)
        elif isinstance(net, ViT):
            initialize_vit(net, scale=scale if scale is not None else 1.0, residual_scaling=residual_scaling)
        elif isinstance(net, GPT):
            initialize_gpt(net, scale=scale, residual_scaling=residual_scaling)
        elif isinstance(net, SSTTransformer):
            # Use default PyTorch init for embeddings/attention/FF layers.
            # Head is already zero-initialized in SSTTransformer.__init__,
            # matching Damian et al. (arXiv:2209.15594) kernel_init=zeros.
            pass
        elif isinstance(net, (SSTMLP, SSTCNN)):
            # Use default PyTorch init for the feature extractor.
            # The classifier head is already zero-initialized in __init__.
            pass
        else:
            raise ValueError("Unknown net type")


# ---------------------------------------------------------------------------
# Muon optimizer (MomentUm Orthogonalized by Newton-Schulz)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5,
                                 eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2, "zeropower_via_newtonschulz5 requires a 2-D tensor"
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T
    X = G / (G.norm(p='fro') + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-Schulz.

    2D+ params get Newton-Schulz orthogonalized momentum; 1D params
    (biases, norms) fall back to plain SGD+momentum.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, nesterov, ns_steps = (
                group['lr'], group['momentum'], group['nesterov'], group['ns_steps'])
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g)
                g_eff = g + mom * buf if nesterov else buf
                if p.ndim >= 2:
                    shape = p.shape
                    update = zeropower_via_newtonschulz5(
                        g_eff.view(shape[0], -1), steps=ns_steps).view(shape)
                else:
                    update = g_eff
                p.data.add_(update, alpha=-lr)
        return loss


def prepare_optimizer(net, lr, momentum, adam, nesterov: bool = False,
                      rmsprop_alpha: float = None, rmsprop_momentum: float = None,
                      muon: bool = False, muon_momentum: float = 0.95,
                      muon_ns_steps: int = 5):
    if muon:
        return Muon(net.parameters(), lr=lr, momentum=muon_momentum,
                    ns_steps=muon_ns_steps)
    if rmsprop_alpha is not None:
        kwargs = dict(lr=lr, alpha=rmsprop_alpha)
        if rmsprop_momentum is not None:
            kwargs['momentum'] = rmsprop_momentum
        return T.optim.RMSprop(net.parameters(), **kwargs)
    if adam:
        if momentum is not None:
            raise ValueError("Momentum is not supported for Adam, just because. Change the code if you need to change the params in Adam")
        # adam is (beta1, beta2) when Adam is used
        betas = adam if isinstance(adam, tuple) else (0.9, 0.99)
        return T.optim.Adam(net.parameters(), lr=lr, betas=betas)
    
    if momentum is not None:
        # PyTorch expects the keyword 'nesterov' (lowercase) and requires momentum > 0
        return T.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=bool(nesterov))

        
    
    return T.optim.SGD(net.parameters(), lr=lr, momentum=0)



def get_path_of_last_net(path: Union[str, Path], not_final=False):
    path = Path(path)
    if path.is_dir():
        files = list(path.glob('*.pt'))
        if 'net_final.pt' in [file.name for file in files]:
            return path / 'net_final.pt'
        if len(files) == 0:
            return None
        files.sort(key=lambda x: x.stat().st_mtime)

        return files[-1]
    else:
        return path
