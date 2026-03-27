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
