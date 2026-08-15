# Anu — Training Half

PyTorch training side of Anu. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync --dev
```

## Commands

```bash
uv run python data/train_tokenizer.py    # builds tokenizer.json (full corpus — heavy, run on Kaggle)
uv run python data/prepare_dataset.py    # downloads + tokenizes TinyStories (heavy — run on Kaggle)
uv run python data/prepare_dataset.py --max-stories 5000   # local smoke run
uv run python train.py                   # trains with checkpoint/resume (Phase 3, Kaggle)
uv run python export.py                  # exports safetensors weights (Phase 4)
uv run pytest                            # tests
uv run ruff check .                      # lint
```

## Layout

- `model/` — ModelConfig, RMSNorm, RoPE, attention, MLP, transformer
- `data/` — tokenizer + dataset pipeline (`train_tokenizer.py`, `prepare_dataset.py`, `dataset.py`)
- `kaggle/` — self-contained notebooks/scripts for heavy runs on Kaggle (see `kaggle/README.md`)
- `train.py`, `export.py`, `sample.py` — training loop, weight export, reference generations

The production `weights/tokenizer.json` was trained on the full
`TinyStoriesV2-GPT4-train.txt` corpus (12k vocab, byte-level BPE, lossless
round-trip for any UTF-8).