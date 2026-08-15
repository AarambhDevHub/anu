"""Download TinyStories, tokenize it, and pack it into fixed-length sequences.

Pipeline (per Architecture §6):
  1. Download TinyStoriesV2-GPT4-train.txt / TinyStoriesV2-GPT4-valid.txt
     from the Hugging Face Hub (cached in training/data/raw/).
  2. Tokenize every story with weights/tokenizer.json.
  3. Concatenate the token streams, separating stories with <|endoftext|>,
     and write one contiguous uint16 array per split (train.bin / valid.bin).

The Dataset then serves fixed-length context windows from random offsets of
these bins (packed, not padded), which is the GPU-efficient layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ID = "roneneldan/TinyStories"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "training" / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENIZED_DIR = DATA_DIR / "tokenized"
TOKENIZER_PATH = PROJECT_ROOT / "weights" / "tokenizer.json"
DEFAULT_CONTEXT_LENGTH = 512


def download_raw(split: str) -> Path:
    """Download the raw corpus file for a split, returning its local path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(REPO_ID, FILES[split], repo_type="dataset")
    return Path(path)


def iter_stories(raw_path: Path, max_stories: int | None = None):
    ds = load_dataset("text", data_files=[str(raw_path)], split="train", streaming=True)
    for i, row in enumerate(ds):
        text = row["text"]
        if text:
            yield text
        if max_stories is not None and i + 1 >= max_stories:
            break


def pack_stories(stories, tokenizer: Tokenizer, out_path: Path) -> int:
    """Tokenize an iterable of stories and pack them into a single uint16 array.

    Stories are concatenated with the <|endoftext|> token as a separator.
    Returns the total number of tokens written.
    """
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("tokenizer has no <|endoftext|> token")
    if eos_id > 65535:
        raise ValueError("vocab too large for uint16 packing")

    capacity = 1 << 22
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(capacity,))
    pos = 0
    with tqdm(desc=f"tokenizing {out_path.name}", unit="story") as pbar:
        for text in stories:
            ids = tokenizer.encode(text).ids
            if pos + len(ids) + 1 > arr.shape[0]:
                while capacity <= pos + len(ids) + 1:
                    capacity *= 2
                arr = np.memmap(out_path, dtype=np.uint16, mode="r+", shape=(capacity,))
            arr[pos : pos + len(ids)] = ids
            pos += len(ids)
            arr[pos] = eos_id
            pos += 1
            pbar.update(1)
    final = np.memmap(out_path, dtype=np.uint16, mode="r+", shape=(pos,))
    final.flush()
    return pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Limit corpus size (useful for local smoke runs).")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    meta = {"context_length": args.context_length, "vocab_size": tokenizer.get_vocab_size()}
    for split in args.splits:
        raw_path = download_raw(split)
        out_path = TOKENIZED_DIR / f"{split}.bin"
        num_tokens = pack_stories(iter_stories(raw_path, args.max_stories), tokenizer, out_path)
        meta[split] = {"num_tokens": num_tokens}
        print(f"{split}: {num_tokens:,} tokens -> {out_path}")

    with open(TOKENIZED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {TOKENIZED_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
