"""Phase 1 on Kaggle: full-corpus tokenizer + packed dataset.

Self-contained script: create a Kaggle notebook (Python, CPU is fine —
tokenizer training and tokenization are CPU work), paste this into one
cell, and run. Torch is preinstalled on Kaggle; this script needs only
datasets/tokenizers/numpy/tqdm (see kaggle-requirements.txt).

Outputs written to /kaggle/working (auto-downloaded as notebook output):
  tokenizer.json     — BPE, vocab 12000, trained on the full corpus
  train.bin/valid.bin — packed uint16 token arrays
  meta.json           — token counts + config

Optional flag: --retrain 0 to reuse an uploaded weights/tokenizer.json
and only run the tokenization/packing step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

REPO_ID = "roneneldan/TinyStories"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 12_000
CONTEXT_LENGTH = 512
OUT_DIR = Path("/kaggle/working")
LOCAL_FALLBACK = Path("../working")  # so the script also runs outside Kaggle


def out_dir() -> Path:
    d = OUT_DIR if OUT_DIR.exists() else LOCAL_FALLBACK
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_raw(split: str) -> Path:
    return Path(hf_hub_download(REPO_ID, FILES[split], repo_type="dataset"))


def iter_stories(raw_path: Path):
    ds = load_dataset("text", data_files=[str(raw_path)], split="train", streaming=True)
    for row in ds:
        if row["text"]:
            yield row["text"]


def train_tokenizer(raw_path: Path, out: Path) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_stories(raw_path), trainer=trainer)
    tokenizer.save(str(out))
    return tokenizer


def pack_stories(stories, tokenizer: Tokenizer, out_path: Path) -> int:
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    capacity = 1 << 24
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
    parser.add_argument("--retrain", type=int, default=1,
                        help="0 = reuse an existing tokenizer.json instead of retraining")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    # parse_known_args: ignore Jupyter/Colab kernel args (e.g. `-f kernel.json`)
    args, _ = parser.parse_known_args()
    out = out_dir()

    if args.retrain:
        print("training tokenizer on full corpus...")
        tokenizer = train_tokenizer(download_raw("train"), out / "tokenizer.json")
    else:
        src = Path("tokenizer.json")
        if not src.exists():
            sys.exit("tokenizer.json not found in working dir (upload it first)")
        tokenizer = Tokenizer.from_file(str(src))

    meta = {"context_length": CONTEXT_LENGTH, "vocab_size": tokenizer.get_vocab_size()}
    for split in args.splits:
        raw_path = download_raw(split)
        num_tokens = pack_stories(iter_stories(raw_path), tokenizer, out / f"{split}.bin")
        meta[split] = {"num_tokens": num_tokens}
        print(f"{split}: {num_tokens:,} tokens")

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done. outputs in {out}: tokenizer.json, train.bin, valid.bin, meta.json")


if __name__ == "__main__":
    main()
