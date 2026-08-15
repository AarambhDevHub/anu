"""PyTorch Dataset over packed token bins.

Each item is a contiguous context window drawn from a random offset in the
packed uint16 array (nanoGPT-style): the random offset is the shuffle, and
packed-not-padded layout keeps the GPU busy with dense windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, bin_path: str | Path, context_length: int, seed: int | None = None):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.rng = np.random.default_rng(seed)
        self.max_offset = max(0, len(self.data) - context_length - 1)

    def __len__(self) -> int:
        return self.max_offset

    def __getitem__(self, index: int):
        offset = int(self.rng.integers(0, self.max_offset + 1))
        x = self.data[offset : offset + self.context_length]
        y = self.data[offset + 1 : offset + self.context_length + 1]
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def load_meta(meta_path: str | Path) -> dict:
    with open(meta_path) as f:
        return json.load(f)
