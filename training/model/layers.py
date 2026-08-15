"""Model layers: RMSNorm, RoPE, causal multi-head attention, MLP (Architecture §4.2).

No bias terms anywhere. Each component will be ported 1:1 to Candle in
server/src/model/layers.rs — keep the math here explicit and unadorned.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization (pre-norm), no bias, with a learnable per-channel scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms.to(x.dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (Su et al. 2021), split-half convention.

    Frequencies are 1 / 10000^(2i / dim) for i in [0, dim/2). Applied to Q and K
    inside attention as a per-position rotation of each (x1, x2) pair:
        x1' = x1*cos(t) - x2*sin(t),  x2' = x1*sin(t) + x2*cos(t)
    """

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # (max_seq_len, dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension of x in halves. x: (B, H, T, D)."""
        seq_len = x.size(2)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)


class CausalSelfAttention(nn.Module):
    """Scaled dot-product attention with a causal mask, applied to every position."""

    def __init__(self, n_embd: int, n_head: int, context_length: int, rope: RotaryEmbedding):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.rope = rope
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        causal = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = att.softmax(dim=-1)
        y = att @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """Linear -> GELU -> Linear, standard 4x expansion."""

    def __init__(self, n_embd: int, ffn_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, ffn_dim, bias=False)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(ffn_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))
