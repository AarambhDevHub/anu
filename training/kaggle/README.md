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