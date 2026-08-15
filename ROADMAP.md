# Anu — Roadmap

Target: **10–12 days** (fits a 1–2 week window). Each phase lists duration, hardware, tasks, tests, milestone, and the git tag to cut when the phase is done.

---

## Phase 0 — Project Setup & Planning

- **Duration:** 0.5 day
- **Hardware:** Local CPU
- **Tasks:**
  - Scaffold the monorepo structure (see Architecture §3)
  - Initialize `training/` (Python/uv or venv) and `server/` (Cargo) as sibling projects
  - Create Kaggle account/notebook, verify GPU access
  - Dataset confirmed: `roneneldan/TinyStories`, using `TinyStoriesV2-GPT4-train.txt` (~2.12M stories, ~470-500M tokens, CDLA-Sharing-1.0 license)
  - Write skeleton README with project description and architecture summary
- **Tests:** Both `training/` and `server/` build/lint cleanly with no code yet (empty pipelines pass)
- **Milestone:** Project scaffolded, dataset decided
- **Git tag:** `v0.1.0-setup`

---

## Phase 1 — Tokenizer & Data Pipeline

- **Duration:** 1 day
- **Hardware:** Local CPU (smoke tests) + Kaggle (full corpus — heavy BPE training and tokenization run there, not locally; see `training/kaggle/`)
- **Tasks:**
  - Load `roneneldan/TinyStories` via the `datasets` library and pull `TinyStoriesV2-GPT4-train.txt` / `TinyStories-valid.txt`
  - Train the BPE tokenizer with HF `tokenizers` (vocab_size = 12,000, byte-level, seeded with all 256 byte tokens so round-trip is lossless for any UTF-8), save `tokenizer.json`
  - Tokenize and chunk the dataset into fixed-length (512-token) packed sequences (`train.bin` / `valid.bin`, uint16, `<|endoftext|>` separators)
  - Build the PyTorch `Dataset`/`DataLoader` (random-offset windows over packed bins)
- **Tests:**
  - Tokenizer round-trip test (encode → decode → identical text, incl. unicode/emoji)
  - DataLoader batch shape/dtype test
- **Milestone:** Tokenizer and data pipeline ready
- **Git tag:** `v0.2.0-data`

---

## Phase 2 — Model Implementation (PyTorch)

- **Duration:** 1–1.5 days
- **Hardware:** Local CPU (no GPU needed yet)
- **Tasks:**
  - Implement `ModelConfig`, RMSNorm, RoPE, causal attention, MLP block, full transformer (Architecture §4)
  - Wire up the forward pass and loss (cross-entropy on next-token prediction)
  - Overfit a single tiny batch as a correctness sanity check
- **Tests:** Loss on a single repeated batch drops to near-zero within a few hundred steps (proves the model/loss/optimizer wiring is correct before spending any GPU budget)
- **Milestone:** Model architecture implemented and sanity-checked
- **Git tag:** `v0.3.0-model`

---

## Phase 3 — Training on Kaggle

- **Duration:** 3–4 days (spread across multiple Kaggle sessions)
- **Hardware:** Kaggle GPU (T4×2 or P100, free tier) — all model training runs there; local CPU only for smoke tests (`uv run pytest`, tiny `--steps`)
- **Tasks:**
  - Port the training script into a Kaggle notebook (`training/kaggle/phase3_train.py`, self-contained; mirrors `training/train.py` — keep in sync)
  - Implement checkpoint/resume: save every N steps to `/kaggle/working` (auto-downloaded as notebook output); on session restart, upload the latest checkpoint into the input dataset and rerun — the script auto-resumes model + optimizer + LR schedule state
  - Run the training loop with cosine LR schedule + warmup, mixed precision (bf16 on T4-class, fp16 + GradScaler fallback)
  - Speed knobs: fused SDPA attention and a low-precision cross-entropy (avoids the fp32 logits upcast that dominates a T4 step at vocab=12000); batch 32 x 512 tokens, `TOTAL_STEPS=25_000` ≈ one epoch in ~4-6h
  - Monitor validation loss and periodically sample generations to track qualitative progress (logged to `checkpoints/metrics.jsonl`)
  - Respect the 12-hour session cap and ~30 GPU-hour/week budget — plan sessions accordingly
- **Tests:**
  - Validation loss trending down across checkpoints
  - Sample generations become more coherent over time (qualitative check, not just loss)
- **Milestone:** Training converged (or budget exhausted with acceptable sample quality)
- **Git tag:** `v0.4.0-trained`

---

## Phase 4 — Export & Publish Weights

- **Duration:** 0.5 day
- **Hardware:** Local/Kaggle CPU
- **Tasks:**
  - Convert the final checkpoint to `.safetensors`
  - Write final `config.json` and confirm `tokenizer.json` is finalized
  - Generate a model card (architecture, dataset, training details, known limitations)
  - Push weights + config + tokenizer to Hugging Face Hub (public)
  - Generate and save reference completions (`sample.py`, fixed prompts + seed) for parity testing
- **Tests:** Fresh Python session reloads the safetensors checkpoint and reproduces the saved reference generations exactly
- **Milestone:** Weights published publicly on Hugging Face Hub
- **Git tag:** `v0.5.0-weights`

---

## Phase 5 — Candle Port & Parity Testing

- **Duration:** 1.5–2 days
- **Hardware:** Local CPU (GPU optional, not required for a 31M model)
- **Tasks:**
  - Implement the matching model definition in Candle (`server/src/model/`)
  - Implement safetensors loading with explicit PyTorch→Candle tensor name mapping
  - Implement greedy decoding first (no sampling) for the parity check
- **Tests:** `server/tests/parity.rs` — greedy-decoded Rust output matches the saved Python reference generations for the fixed prompt set
- **Milestone:** Rust inference verified correct against the Python reference
- **Git tag:** `v0.6.0-candle-port`

---

## Phase 6 — OpenAI-Compatible Server

- **Duration:** 1.5 days
- **Hardware:** Local
- **Tasks:**
  - Axum app scaffold, `GET /v1/models`, `POST /v1/completions` with SSE streaming
  - KV-cache generation loop; temperature/top-k/top-p sampling
  - Optional: `POST /v1/chat/completions` with a simple prompt template (documented as base-model behavior, not instruction-following)
  - Request validation, structured error responses
- **Tests:**
  - Integration tests hitting each endpoint
  - Manual streaming test (`curl -N`) confirms tokens arrive incrementally
- **Milestone:** OpenAI-compatible API functional end-to-end
- **Git tag:** `v0.7.0-server`

---

## Phase 7 — Production Hardening

- **Duration:** 1 day
- **Hardware:** Local
- **Tasks:**
  - Rate limiting (`tower-governor`)
  - Structured logging/tracing (`tracing`)
  - API-key auth middleware
  - Request timeouts, graceful shutdown
- **Tests:**
  - Rate-limit test (requests beyond threshold rejected correctly)
  - Auth-rejection test (missing/invalid key rejected correctly)
  - Basic concurrency/load smoke test
- **Milestone:** Server hardened for a public-facing demo
- **Git tag:** `v0.8.0-hardened`

---

## Phase 8 — Containerization & Deployment

- **Duration:** 1 day
- **Hardware:** Local + cloud (Fly.io/Railway/Shuttle)
- **Tasks:**
  - Multi-stage `Dockerfile`
  - GitHub Actions CI: test + build on push
  - Deploy to chosen host; wire up health-check endpoint
- **Tests:** CI pipeline green; deployed health-check responds; a real `/v1/completions` request against the live URL succeeds
- **Milestone:** Live, publicly reachable endpoint
- **Git tag:** `v0.9.0-deployed`

---

## Phase 9 — Documentation, Demo & Portfolio Polish

- **Duration:** 1 day
- **Hardware:** Local
- **Tasks:**
  - Finalize README: architecture diagram, curl examples, demo GIF
  - Record a short demo video
  - Write the portfolio/LinkedIn post
  - Prepare interview talking points: design trade-offs (Architecture §11), why Rust for serving, what you'd do differently at scale
- **Tests:** A fresh clone + README walkthrough works end-to-end with no undocumented steps
- **Milestone:** **v1.0.0 — project ready to share and discuss in interviews**
- **Git tag:** `v1.0.0`

---

## Stretch Phases (post-v1.0.0, optional)

- **Rust-flavored data mix-in:** retrain or fine-tune on a blend including Rust docs/code for on-brand flavor, once the base recipe is proven
- **Dynamic request batching** in the server for real throughput gains
- **LoRA fine-tune** for lightweight instruction-following, to make `/v1/chat/completions` genuinely useful rather than a documented limitation
