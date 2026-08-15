# Anu — Training Half

PyTorch training side of Anu. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync --dev
```

## Commands

```bash
uv run python data/train_tokenizer.py    # builds tokenizer.json (Phase 1)
uv run python data/prepare_dataset.py    # downloads + tokenizes TinyStories (Phase 1)
uv run python train.py                   # trains with checkpoint/resume (Phase 3)
uv run python export.py                  # exports safetensors weights (Phase 4)
uv run pytest                            # tests
uv run ruff check .                      # lint
```

## Layout

- `model/` — ModelConfig, RMSNorm, RoPE, attention, MLP, transformer
- `data/` — tokenizer + dataset pipeline
- `train.py`, `export.py`, `sample.py` — training loop, weight export, reference generations