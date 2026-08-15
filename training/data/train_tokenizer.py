"""Train a BPE tokenizer on TinyStories and save it as weights/tokenizer.json.

Dataset: roneneldan/TinyStories, TinyStoriesV2-GPT4-train.txt (CDLA-Sharing-1.0,
see arXiv:2305.07759). BPE is byte-level so encode -> decode is lossless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.prepare_dataset import download_raw  # noqa: E402

REPO_ID = "roneneldan/TinyStories"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / "tokenizer.json"


def iter_stories(max_stories: int | None = None):
    """Yield story texts from the TinyStoriesV2-GPT4-train.txt download."""
    path = download_raw("train")
    ds = load_dataset(
        "text", data_files=[str(path)], split="train", streaming=True, trust_remote_code=False
    )
    for i, row in enumerate(ds):
        text = row["text"]
        if text:
            yield text
        if max_stories is not None and i + 1 >= max_stories:
            break


def build_tokenizer(vocab_size: int, max_stories: int | None) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_stories(max_stories), trainer=trainer)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=12_000)
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Limit corpus size (useful for local smoke runs).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write tokenizer.json")
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.vocab_size, args.max_stories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    print(f"vocab_size={tokenizer.get_vocab_size()} saved to {args.output}")


if __name__ == "__main__":
    main()
