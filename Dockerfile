# Multi-stage Dockerfile — full build lands in Phase 8.
# Build stage: compile the Rust server. Runtime: minimal image
# with binary + config.json + tokenizer.json (weights from HF Hub).

FROM rust:1 AS builder
WORKDIR /app
COPY server/ server/
RUN cargo build --release --manifest-path server/Cargo.toml

FROM debian:bookworm-slim AS runtime
WORKDIR /app
COPY --from=builder /app/server/target/release/anu-server /usr/local/bin/anu-server
COPY weights/config.json weights/tokenizer.json /app/weights/
EXPOSE 8080
CMD ["anu-server"]