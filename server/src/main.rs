//! Anu serving half: Candle inference + Axum OpenAI-compatible API.
//!
//! Phase 0 scaffold — module skeletons per Architecture §3. Real
//! implementations land in Phases 5-7.

mod api;
mod generation;
mod middleware;
mod model;

fn main() {
    println!("anu-server: scaffold ready");
}