"""The kernel-writing tasks: a PyTorch reference, how to make its inputs, and shapes.

Each reference is a short function whose source goes into the prompt verbatim. The
agent must write `kernel(*inputs)` returning the same tensor. `dev_shape` is the one
shape a naive harness checks; `shapes` are what the rigorous harness checks and times,
deliberately including sizes that are not powers of two, a single row, and the very
wide or very narrow cases kernels tuned on one shape tend to get wrong.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


def softmax(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def layernorm(x, w, b):
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), 1e-5).to(x.dtype)


def rmsnorm(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)


def add_rmsnorm(x, r, w):
    h = x.float() + r.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)


def bias_gelu(x, b):
    return F.gelu(x.float() + b.float(), approximate="tanh").to(x.dtype)


def swiglu(x):
    n = x.shape[-1] // 2
    return (F.silu(x[..., :n].float()) * x[..., n:].float()).to(x.dtype)


def row_mean(x):
    return x.float().mean(dim=-1).to(x.dtype)


def cross_entropy(logits, target):
    return F.cross_entropy(logits.float(), target, reduction="none")


def rope(x, cos, sin):
    # x: [tokens, heads, dim]; cos, sin: [tokens, dim // 2]; rotate-half convention.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    c, s = cos[:, None, :].float(), sin[:, None, :].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def causal_softmax(scores):
    t = scores.shape[-1]
    mask = torch.ones(t, t, dtype=torch.bool, device=scores.device).triu(1)
    return torch.softmax(scores.float().masked_fill(mask, float("-inf")), dim=-1).to(scores.dtype)


def matmul_bias_relu(a, b, bias):
    return torch.relu(a.float() @ b.float() + bias.float()).to(a.dtype)


def transpose(x):
    return x.t().contiguous()


@dataclass
class Task:
    name: str
    description: str
    reference: Callable
    make_inputs: Callable          # (shape, generator, device, dtype) -> tuple of tensors
    dev_shape: tuple
    shapes: list = field(default_factory=list)
    atol: float = 1e-2
    rtol: float = 1e-2

    @property
    def source(self) -> str:
        return inspect.getsource(self.reference)


def _randn(shape, g, device, dtype, scale=1.0):
    return (torch.randn(shape, generator=g, device=device) * scale).to(dtype)


TASKS = {t.name: t for t in [
    Task("softmax", "Row-wise softmax over the last dimension of a 2-D tensor.", softmax,
         lambda s, g, d, t: (_randn(s, g, d, t, 3.0),),
         (1024, 1024), [(1024, 1024), (4096, 4096), (1000, 3001), (1, 50257), (8192, 128)],
         atol=2e-3, rtol=1e-2),
    Task("layernorm", "LayerNorm over the last dimension of a 2-D tensor, with weight and bias, eps=1e-5.",
         layernorm,
         lambda s, g, d, t: (_randn(s, g, d, t), _randn(s[-1:], g, d, t, 0.5) + 1, _randn(s[-1:], g, d, t, 0.1)),
         (1024, 1024), [(1024, 1024), (4096, 4096), (1000, 3001), (1, 8192), (8192, 128)],
         atol=1e-2, rtol=1e-2),
    Task("rmsnorm", "RMSNorm over the last dimension of a 2-D tensor with a weight vector, eps=1e-6.",
         rmsnorm,
         lambda s, g, d, t: (_randn(s, g, d, t), _randn(s[-1:], g, d, t, 0.5) + 1),
         (1024, 1024), [(1024, 1024), (4096, 4096), (1000, 3001), (1, 8192), (8192, 128)],
         atol=1e-2, rtol=1e-2),
    Task("add_rmsnorm", "Add a residual to a 2-D tensor, then RMSNorm the sum over the last dimension, eps=1e-6.",
         add_rmsnorm,
         lambda s, g, d, t: (_randn(s, g, d, t), _randn(s, g, d, t), _randn(s[-1:], g, d, t, 0.5) + 1),
         (1024, 1024), [(1024, 1024), (4096, 4096), (1000, 3001), (1, 8192), (8192, 128)],
         atol=1e-2, rtol=1e-2),
    Task("bias_gelu", "Add a bias vector to each row of a 2-D tensor, then tanh-approximate GELU.", bias_gelu,
         lambda s, g, d, t: (_randn(s, g, d, t, 2.0), _randn(s[-1:], g, d, t)),
         (1024, 1024), [(1024, 1024), (4096, 4096), (1000, 3001), (1, 50257), (8192, 128)],
         atol=1e-2, rtol=1e-2),
    Task("swiglu", "SwiGLU: split the last dimension in half, silu(first half) * second half.", swiglu,
         lambda s, g, d, t: (_randn(s, g, d, t, 2.0),),
         (1024, 2048), [(1024, 2048), (4096, 8192), (1000, 6002), (1, 12288), (8192, 256)],
         atol=1e-2, rtol=1e-2),
    Task("row_mean", "Mean over the last dimension of a 2-D tensor, giving one value per row.", row_mean,
         lambda s, g, d, t: (_randn(s, g, d, t),),
         (1024, 1024), [(1024, 1024), (4096, 4096), (1000, 3001), (1, 50257), (8192, 128)],
         atol=2e-3, rtol=1e-2),
    Task("cross_entropy", "Per-row cross-entropy loss (no reduction) of logits [M, V] against int64 targets [M]; "
         "returns float32 [M].", cross_entropy,
         lambda s, g, d, t: (_randn(s, g, d, t, 2.0),
                             torch.randint(0, s[-1], (s[0],), generator=g, device=d)),
         (1024, 32000), [(1024, 32000), (512, 50257), (3, 151936), (4096, 1000)],
         atol=1e-3, rtol=1e-3),
    Task("rope", "Rotary position embedding, rotate-half convention: x [tokens, heads, dim], cos and sin "
         "[tokens, dim // 2].", rope,
         lambda s, g, d, t: (_randn(s, g, d, t), _randn((s[0], s[2] // 2), g, d, t).cos(),
                             _randn((s[0], s[2] // 2), g, d, t).sin()),
         (1024, 32, 128), [(1024, 32, 128), (4096, 32, 128), (1000, 7, 64), (1, 32, 128)],
         atol=1e-2, rtol=1e-2),
    Task("causal_softmax", "Causal softmax over attention scores [batch, T, T]: positions above the diagonal "
         "are masked out, softmax over the last dimension.", causal_softmax,
         lambda s, g, d, t: (_randn(s, g, d, t, 3.0),),
         (4, 1024, 1024), [(4, 1024, 1024), (1, 4096, 4096), (3, 777, 777), (16, 128, 128)],
         atol=2e-3, rtol=1e-2),
    Task("matmul_bias_relu", "relu(a @ b + bias) for a [M, K], b [K, N], bias [N].", matmul_bias_relu,
         lambda s, g, d, t: (_randn((s[0], s[1]), g, d, t, 0.1), _randn((s[1], s[2]), g, d, t, 0.1),
                             _randn((s[2],), g, d, t, 0.1)),
         (1024, 1024, 1024), [(1024, 1024, 1024), (2048, 2048, 2048), (1000, 777, 1500), (16, 4096, 4096)],
         atol=2e-2, rtol=2e-2),
    Task("transpose", "Transpose a 2-D tensor and return it contiguous.", transpose,
         lambda s, g, d, t: (_randn(s, g, d, t),),
         (1024, 1024), [(1024, 1024), (4096, 4096), (1000, 3001), (8192, 64)],
         atol=0.0, rtol=0.0),
]}
