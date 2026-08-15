# Phase 1 on Kaggle

All heavy data work (full-corpus BPE training, tokenizing ~2.12M stories)
runs on Kaggle, not locally. This is a self-contained script — no repo
imports, so it works when pasted into a blank notebook.

## Steps

1. Create a notebook at [kaggle.com](https://kaggle.com) (New Notebook).
   **CPU accelerator is sufficient** for this phase — tokenizer training and
   tokenization are CPU work. GPU only matters from Phase 3 onward.
2. In the first cell:
   `!pip install -q datasets huggingface_hub numpy tokenizers tqdm`
   (also available as `training/kaggle-requirements.txt`)
3. Paste the contents of `training/kaggle/phase1_data_pipeline.py` into a
   second cell and run. It downloads `TinyStoriesV2-GPT4-train.txt` /
   `TinyStoriesV2-GPT4-valid.txt` (~2.2 GB total), trains the 12k-vocab BPE
   tokenizer, and packs both splits into uint16 token bins.
4. Outputs land in `/kaggle/working` and are auto-downloaded when the
   notebook run finishes:
   - `tokenizer.json` → copy to repo `weights/tokenizer.json`
   - `train.bin`, `valid.bin`, `meta.json` → copy to `training/data/tokenized/`
5. Locally, verify the artifacts:
   `uv run pytest` (includes a real-tokenizer round-trip test)
   `uv run python data/prepare_dataset.py --max-stories 100 --splits valid` (smoke re-pack)

## Variants

- Already have `weights/tokenizer.json` (e.g. the one trained locally) and
  only need bins? Upload it to the notebook and run with `--retrain 0`.
- Dataset config mirrors `training/data/prepare_dataset.py`; keep both in
  sync if you change pack format or context length.

---

# Phase 3 on Kaggle (model training)

**GPU accelerator required** (T4 or better). Self-contained script:
`training/kaggle/phase3_train.py` — mirrors `training/train.py` + the model
modules; keep in sync when changing either.

## Setup

1. Create a notebook with **GPU** accelerator.
2. First cell: `!pip install -q tokenizers numpy tqdm` (torch is preinstalled).
3. Add the Phase 1 outputs as an input dataset. Two options:
   - Create a dataset version "anu-data" from the Phase 1 notebook's
     downloaded outputs (`train.bin`, `valid.bin`, `meta.json`,
     `tokenizer.json`) and Add input it to the notebook, or
   - Drop the files into `/kaggle/input/anu-data/` via the notebook's
     input section.
4. Paste `training/kaggle/phase3_train.py` into a second cell and run.

## Session budget

- ~31M-param model, batch 32 x 512 tokens = 16,384 tokens/step.
- `TOTAL_STEPS = 25_000` ≈ one epoch of the ~470M-token corpus; on a T4
  expect ~4-6h total, typically split across 2 sessions.
- Speedup knobs: training uses the fused attention (`use_sdpa`) and a
  low-precision cross-entropy (no fp32 logits upcast); the log prints a
  per-phase breakdown (`fwd+loss / bwd / opt` ms) every 25 steps so you can
  see where time goes.

## Resume across sessions (12h Kaggle cap)

1. When a session ends (or you stop it), the latest checkpoint +
   `metrics.jsonl` are in `/kaggle/working/checkpoints/` and auto-download.
2. Upload those checkpoints back into the "anu-data" input dataset
   (new version: keep `train.bin`/`valid.bin`/`meta.json`/`tokenizer.json`
   plus the `checkpoints/` folder).
3. Rerun the notebook: it auto-detects the latest `ckpt-*.pt` and continues
   exactly where it left off (model + optimizer + LR schedule state).

## Verifying progress

- `metrics.jsonl` logs train loss, val loss, and sampled generations.
- Val loss should trend down across checkpoints; samples should get
  progressively more readable.