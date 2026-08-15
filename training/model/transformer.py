"""The full decoder-only transformer (Architecture §4.2).

Layout per block: RMSNorm -> causal self-attention (RoPE on Q/K) -> residual,
then RMSNorm -> MLP -> residual. Final RMSNorm precedes the tied output head.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .layers import MLP, CausalSelfAttention, RMSNorm, RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.context_length, rope)
        self.norm2 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = MLP(config.n_embd, config.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AnuTransformer(nn.Module):
    """Decoder-only transformer with tied input embedding / output head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, self.rope) for _ in range(config.n_layer)]
        )
        self.final_rmsnorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) -> logits: (B, T, vocab_size)."""
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_rmsnorm(x)
        return self.lm_head(x)

    def compute_loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over next-token prediction for every position."""
        logits = self(input_ids)
        return nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size), targets.view(-1)
        )


def num_parameters(model: nn.Module) -> int:
    """Total parameter count, deduplicating the tied embedding/output head."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total
