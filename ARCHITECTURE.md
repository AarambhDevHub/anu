# Anu — Architecture

*A small decoder-only LLM, trained from scratch in PyTorch and served through an OpenAI-compatible API in Rust.*

> **Naming note:** "Anu" is a placeholder (Sanskrit for "atom" — smallest particle). Rename freely; every reference below is a simple find-replace.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Repository Structure](#3-repository-structure)
4. [Model Architecture (PyTorch)](#4-model-architecture-pytorch)
5. [Tokenizer Design](#5-tokenizer-design)
6. [Training Pipeline Architecture](#6-training-pipeline-architecture)
7. [Weight Export & Interop Contract](#7-weight-export--interop-contract)
8. [Rust Inference Architecture (Candle)](#8-rust-inference-architecture-candle)
9. [Server Architecture (Axum, OpenAI-Compatible API)](#9-server-architecture-axum-openai-compatible-api)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Design Trade-offs Summary](#11-design-trade-offs-summary)

---

## 1. Overview

Anu is a single project with two halves that share one contract — a `safetensors` weight file, a `config.json`, and a `tokenizer.json`:

- **Training half (Python / PyTorch):** define a small decoder-only transformer, train it from scratch on a small dataset using a Kaggle GPU, and publish the resulting weights openly on Hugging Face Hub.
- **Serving half (Rust / Candle / Axum):** load those exact weights into a from-scratch Rust implementation of the same architecture, and expose it behind an OpenAI-compatible HTTP API.

**Goals, in priority order:**

1. A trained model whose weights are public and independently runnable by anyone.
2. A Rust server that proves production backend engineering, not just "called an LLM API."
3. A coherent demo — the model should visibly produce readable text, not gibberish.
4. Everything finishable within a 1–2 week window on free-tier compute.

**Non-goals:** instruction-following/chat behavior (this is a base language model unless a later stretch phase adds fine-tuning), state-of-the-art benchmark scores, multi-GPU or distributed training.

---

## 2. Design Philosophy

| Principle | Rationale | Trade-off Accepted |
|---|---|---|
| Small enough to train on free Kaggle GPU-hours | Timeline is 1–2 weeks; no budget for rented multi-GPU clusters | Model will not be competitive on general knowledge — that's not the point |
| One shared contract (safetensors + config.json + tokenizer.json) between Python and Rust | Removes the two halves' only real coupling risk — mismatched weights/tokenizer | Requires strict discipline: architecture in Rust must mirror Python exactly |
| Standard, well-tested architecture choices (RoPE, RMSNorm, tied embeddings) | Every "clever" architectural choice is a debugging risk at small scale and short timeline | Less novel than a custom architecture; that novelty isn't the goal here |
| Dataset: `roneneldan/TinyStories` (GPT-4-only subset) chosen for demo coherence over thematic branding | A small, restricted-vocabulary synthetic dataset proven to let tiny models (1M-33M params in the original paper) produce fluent output; code/docs corpora are harder to model well at this scale | Less "on-brand" out of the box; Rust-flavored data can be mixed in as a stretch phase once the core recipe works |
| API-hosted-style architecture, self-hosted weights | The Rust server is judged on serving engineering, not on having invented a new model architecture | The model itself is intentionally simple — all the engineering differentiation lives in the Rust half |
| Checkpoint-and-resume training from day one | Kaggle free tier caps sessions at 12 hours and ~30 GPU-hours/week | Adds moderate upfront engineering before "real" training starts |

---

## 3. Repository Structure

A single monorepo, so one link covers the whole story for recruiters/interviewers:

```
anu/
├── training/                 # Python / PyTorch
│   ├── data/
│   │   ├── prepare_dataset.py
│   │   └── train_tokenizer.py
│   ├── model/
│   │   ├── config.py         # ModelConfig dataclass — single source of truth
│   │   ├── layers.py         # RMSNorm, RoPE, Attention, MLP
│   │   └── transformer.py    # Full decoder-only model
│   ├── train.py              # Training loop, checkpoint/resume
│   ├── export.py             # Checkpoint → safetensors + config.json
│   └── sample.py             # Reference generations for parity testing
│
├── server/                   # Rust
│   ├── Cargo.toml
│   ├── src/
│   │   ├── model/
│   │   │   ├── config.rs     # Mirrors training/model/config.py exactly
│   │   │   ├── layers.rs     # RMSNorm, RoPE, Attention, MLP
│   │   │   └── transformer.rs
│   │   ├── generation.rs     # KV-cache, sampling (temperature/top-k/top-p)
│   │   ├── api/
│   │   │   ├── completions.rs
│   │   │   ├── chat.rs
│   │   │   └── models.rs
│   │   ├── middleware.rs     # rate limiting, auth, logging
│   │   └── main.rs
│   └── tests/
│       └── parity.rs         # Compares output against training/sample.py output
│
├── weights/                  # config.json, tokenizer.json (safetensors published to HF Hub, not committed)
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md
```

---

## 4. Model Architecture (PyTorch)

### 4.1 Configuration

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 12,000 | Custom BPE, sized for a small dataset — see §5 |
| `context_length` | 512 | Keeps attention cost and memory low for Kaggle GPUs |
| `n_layer` | 8 | |
| `n_embd` (d_model) | 512 | |
| `n_head` | 8 | head_dim = 64 |
| `ffn_dim` | 2048 | 4× d_model, standard GELU MLP (not SwiGLU — fewer moving parts to debug) |
| Positional encoding | RoPE (rotary) | Applied to Q/K inside attention; no learned position embedding table |
| Normalization | RMSNorm, pre-norm | Norm before attention and before MLP in each block |
| Weight tying | Input embedding ↔ output head | Saves ~6M params, standard for small models |
| **Approx. total params** | **~31M** | Dominated by the embedding/output matrix (12,000 × 512) |

This config lives in exactly one place conceptually (`training/model/config.py`) and is mirrored by hand in `server/src/model/config.rs`. Any change to one requires the same change to the other — this is the single most important discipline point in the whole project.

### 4.2 Component Design

- **Embedding layer:** `vocab_size × n_embd` lookup table, tied with the final output projection.
- **Transformer block (×8):**
  1. RMSNorm → Multi-head self-attention (causal mask, RoPE on Q/K) → residual add
  2. RMSNorm → MLP (Linear → GELU → Linear) → residual add
- **Attention:** standard scaled dot-product attention, causal mask, no bias terms (simplifies both the PyTorch and Candle implementations and removes a class of key-naming bugs when porting).
- **Output head:** final RMSNorm → tied linear projection to vocab logits.

---

## 5. Tokenizer Design

- Built with Hugging Face `tokenizers` — a BPE tokenizer, **trained on your own dataset**, not a repurposed GPT-2/Llama tokenizer.
- `vocab_size = 12,000` — small enough to match the small dataset (an oversized vocabulary wastes embedding capacity that a 31M-param model can't spare).
- Output artifact: a single `tokenizer.json` file.
- **Critical property:** `tokenizers` is itself written in Rust with Python bindings — the identical `tokenizer.json` loads byte-for-byte the same way in `training/` (via the Python bindings) and in `server/` (via the Rust crate directly). This removes an entire category of train/serve mismatch bugs.

---

## 6. Training Pipeline Architecture

- **Data:** `roneneldan/TinyStories` from Hugging Face Hub — specifically the `TinyStoriesV2-GPT4-train.txt` file (GPT-4-only stories, cleaner than the mixed GPT-3.5/GPT-4 original). ~2.12M short stories, ~470-500M tokens, ~2.23GB raw text — lines up almost exactly with the training budget, so most/all of it can be used rather than hand-carving a subset. Matching `TinyStories-valid.txt` is used for validation loss. License: CDLA-Sharing-1.0 — keep attribution to the TinyStories paper (arXiv:2305.07759) in the eventual model card. A Rust-flavored data slice can be mixed in later as a branding stretch goal once the base recipe is validated (see Roadmap Phase 9+).
- **Batching:** fixed-length sequences of `context_length` tokens, packed/concatenated with document separators rather than padded, for GPU efficiency.
- **Optimizer:** AdamW, cosine LR schedule with linear warmup.
- **Precision:** mixed precision (bf16/fp16) to fit Kaggle's GPU memory and speed up training.
- **Checkpoint/resume — the most important piece of engineering here:**
  - Save a checkpoint every N steps (model state, optimizer state, step count, LR schedule state) to a Kaggle output dataset.
  - On notebook start, detect and resume from the latest checkpoint automatically.
  - This is what makes training safely span multiple 12-hour Kaggle sessions without losing progress.
- **Monitoring:** log training/validation loss per checkpoint; periodically sample generations from the model to track qualitative improvement (loss curves alone can be misleading at this scale).

---

## 7. Weight Export & Interop Contract

This is the seam between the two halves of the project — treat it as a strict contract:

| Artifact | Format | Produced by | Consumed by |
|---|---|---|---|
| Model weights | `.safetensors` | `training/export.py` | `server/src/model` (Candle) |
| Architecture config | `config.json` | `training/export.py` | Both — Python re-checks it, Rust hardcodes matching struct |
| Tokenizer | `tokenizer.json` | `training/data/train_tokenizer.py` | Both, identically |
| Reference generations | plain text/JSON, fixed seed + prompts | `training/sample.py` | `server/tests/parity.rs` |

**Never export as a pickled `.pt`/`.bin` state dict as the primary artifact** — safetensors is what Candle loads natively, and it avoids Rust ever needing to deserialize a Python pickle.

Before leaving the training phase, `sample.py` generates completions for a fixed set of prompts at a fixed seed and saves them. This is the ground truth the Rust port is checked against in Phase 5 of the roadmap.

---

## 8. Rust Inference Architecture (Candle)

- **Model definition:** hand-written Candle module mirroring `training/model/` layer-for-layer — same RMSNorm, same RoPE implementation, same block structure. No dynamic config loading of architecture shape from JSON for v1; hardcode the struct to match `config.json` and assert on load that the file's values agree (fail loudly on drift, don't silently ignore it).
- **Weight loading:** `candle_core::safetensors::load` into the model's parameter tensors, with explicit tensor-name mapping from the PyTorch state-dict naming to the Candle module's naming — this mapping is the single most common source of silent bugs when porting, so write it explicitly rather than relying on name matching.
- **KV-cache:** implemented for autoregressive generation so each new token's forward pass is O(1) in sequence position rather than recomputing the full context every step — this is what makes the server usably fast.
- **Sampling:** temperature, top-k, and top-p (nucleus) sampling, selectable per request.
- **Parity test:** `server/tests/parity.rs` loads the same fixed prompts used in `sample.py` and asserts the Rust greedy-decoded output matches the saved Python reference. This is the correctness gate before any server code is built on top.

---

## 9. Server Architecture (Axum, OpenAI-Compatible API)

| Endpoint | Behavior |
|---|---|
| `GET /v1/models` | Lists the single served model and its metadata |
| `POST /v1/completions` | Raw text completion; streams tokens via Server-Sent Events when `stream: true` |
| `POST /v1/chat/completions` | Optional — wraps a simple prompt template around the base model. Since Anu is a base model (not instruction-tuned), this continues text rather than reliably following instructions; document that limitation clearly rather than overselling it |

**Middleware / cross-cutting concerns:**
- Rate limiting (`tower-governor`)
- Structured logging and tracing (`tracing` crate)
- API-key auth (simple bearer-token check)
- Request timeouts and input validation (reject sequences over `context_length`)
- Consistent OpenAI-style JSON error responses with correct HTTP status codes

**Concurrency model:** the loaded model sits behind a request queue; a single GPU/CPU worker processes generation requests sequentially (or with simple batching if time allows in a stretch phase) — Tokio handles concurrent HTTP connections while generation itself is serialized against the one loaded model instance.

---

## 10. Deployment Architecture

- Multi-stage `Dockerfile`: build stage compiles the Rust server, runtime stage is a minimal image containing only the binary + `config.json` + `tokenizer.json` (weights pulled from HF Hub at startup or baked into the image).
- GitHub Actions CI: run `cargo test` (including the parity test) and build the Docker image on every push.
- Deploy target: Fly.io, Railway, or Shuttle — anywhere that gives a public HTTPS endpoint without managing raw infrastructure.
- Health-check endpoint for the deploy platform to monitor.

---

## 11. Design Trade-offs Summary

| Decision | Chosen | Alternative Considered | Why Chosen |
|---|---|---|---|
| Positional encoding | RoPE | Learned absolute positions | Modern standard, no extra params, generalizes slightly better at this scale |
| Normalization | RMSNorm | LayerNorm | Simpler, fewer params, standard in modern small LLMs |
| MLP | Plain GELU, 4× | SwiGLU (gated) | Fewer moving parts to debug under time pressure; SwiGLU's gain is marginal at 31M params |
| Repo layout | Single monorepo | Two separate repos | One link for recruiters; simpler for a 1–2 week solo project |
| Dataset | `roneneldan/TinyStories` (TinyStoriesV2-GPT4-train.txt) | Rust-themed corpus | Coherent demo output matters more for portfolio impact than thematic branding; Rust data is a stretch-goal add-on |
| Batching in server | Sequential (single-flight) | True dynamic batching | Dynamic batching is real engineering value but risks the timeline; sequential is honest and simple, batching can be a documented "future work" item |
