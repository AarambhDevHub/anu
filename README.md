# Anu

*A small decoder-only LLM, trained from scratch in PyTorch and served through an OpenAI-compatible API in Rust.*

> **Status:** 🚧 In development — see [ROADMAP.md](./ROADMAP.md) for current phase.
>
> **Naming note:** "Anu" (Sanskrit: atom / smallest particle) is a working name.

---

## Why I Built This

I built Anu as a hands-on demonstration of the exact skill combination I'm targeting: **Backend Rust engineering + LLM systems work**, in one project rather than two separate toy demos.

It deliberately spans the full pipeline instead of stopping at "trained a model" or "called an API":

- **Trained from scratch** — a decoder-only transformer trained end-to-end in PyTorch, with the weights published openly so anyone can verify and run it themselves.
- **Served in production-style Rust** — the trained weights are loaded into a from-scratch Candle implementation and exposed behind a real Axum server with an OpenAI-compatible API, streaming, rate limiting, and auth — the actual engineering concerns of running an LLM in a backend, not just a script.

If you're reviewing this as part of a **Backend Rust / LLM AI Engineer** application: the [ARCHITECTURE.md](./ARCHITECTURE.md) doc lays out every design decision and trade-off, and I'm happy to walk through any part of it in more depth.

---

## Overview

| | |
|---|---|
| **Training** | PyTorch, decoder-only transformer (~31M params), trained on Kaggle GPUs |
| **Dataset** | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (GPT-4-only subset) |
| **Weights** | Published openly on Hugging Face Hub (`.safetensors`) |
| **Serving** | Rust — Candle (inference) + Axum (HTTP/OpenAI-compatible API) |
| **API** | `/v1/completions`, `/v1/chat/completions`, `/v1/models` — streaming via SSE |

Full design details: [ARCHITECTURE.md](./ARCHITECTURE.md)
Build plan and progress: [ROADMAP.md](./ROADMAP.md)

---

## Project Structure

```
anu/
├── training/     # PyTorch: model, tokenizer, training loop, weight export
├── server/       # Rust: Candle inference + Axum OpenAI-compatible API
├── weights/      # config.json, tokenizer.json (safetensors on HF Hub)
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## Getting Started

### Train the model (Python)

```bash
cd training
pip install -r requirements.txt

python data/train_tokenizer.py      # builds tokenizer.json
python data/prepare_dataset.py      # downloads + tokenizes TinyStories
python train.py                     # trains with checkpoint/resume
python export.py                    # exports final weights to safetensors
```

### Run the server (Rust)

```bash
cd server
cargo run --release
```

### Query it

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8,
    "stream": true
  }'
```

---

## Model

Weights, tokenizer, and model card are published on Hugging Face Hub: *(link added once Phase 4 is complete — see [ROADMAP.md](./ROADMAP.md))*

Trained on [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0), described in [the TinyStories paper](https://arxiv.org/abs/2305.07759).

---

## Tech Stack

**Training:** Python, PyTorch, Hugging Face `tokenizers`, Kaggle GPUs
**Serving:** Rust, Candle, Axum, Tokio
**Ops:** Docker, GitHub Actions

---

## Support This Project

If this project is useful to you, you can support continued open-source work here:

- ☕ [Buy Me a Coffee](https://buymeacoffee.com/aarambhdevhub)
- 💖 [GitHub Sponsors](https://github.com/sponsors/aarambh-darshan)
- 🔗 [Razorpay](https://razorpay.me/@aarambhdevhub)

---

## Connect

Built by **Aarambh Dev Hub**. If you're hiring for Backend Rust or LLM AI Engineer roles and want to talk about this project:

- 💼 [LinkedIn](https://linkedin.com/in/darshan-vichhi-rust-developer)

---

## License

*(TBD — Apache-2.0 or MIT recommended, matching other Aarambh Dev Hub repos)*
