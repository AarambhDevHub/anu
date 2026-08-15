# Anu — Serving Half

Rust side of Anu: Candle inference + Axum OpenAI-compatible API.

## Commands

```bash
cargo build
cargo clippy --all-targets -- -D warnings
cargo test
cargo fmt --check
```

## Layout (Architecture §3)

- `src/model/` — config.rs (mirrors `training/model/config.py`), layers.rs, transformer.rs
- `src/generation.rs` — KV-cache, sampling
- `src/api/` — completions.rs, chat.rs (Phase 6), models.rs
- `src/middleware.rs` — auth, rate limiting, logging
- `tests/parity.rs` — correctness gate against Python reference generations