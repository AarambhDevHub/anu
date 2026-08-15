"""Model configuration — the single source of truth (Architecture §4.1).

This dataclass must be mirrored by hand in server/src/model/config.rs.
Any change here requires the same change there (and vice versa).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 12_000
    context_length: int = 512
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    ffn_dim: int = 2_048
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def num_params_estimate(self) -> int:
        """Deduplicated (tied) parameter estimate, dominated by the embedding."""
        embedding = self.vocab_size * self.n_embd
        rms_norms = (self.n_layer * 2 + 1) * self.n_embd  # per-block x2 + final
        per_block = (
            3 * self.n_embd * self.n_embd  # q, k, v projections
            + self.n_embd * self.n_embd  # attention output projection
            + 2 * self.n_embd * self.ffn_dim  # MLP (up + down)
        )
        return embedding + rms_norms + per_block * self.n_layer

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    @classmethod
    def from_json(cls, text: str) -> ModelConfig:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_json_file(cls, path: str) -> ModelConfig:
        with open(path) as f:
            return cls.from_json(f.read())
