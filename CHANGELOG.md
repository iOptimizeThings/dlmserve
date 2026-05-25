# Changelog

## v0.1.0 — 2026-05-24

First public release.

- LLaDA-8B-Instruct and LLaDA-1.5 supported.
- OpenAI-compatible `/v1/chat/completions` with documented deviations (ADR 005).
- Step-level continuous batching — batch=8 default on consumer GPUs.
- LocalLeap acceleration (arXiv:2510.07081, Apache-2.0) — opt-in via `use_local_leap=True`.
- Prometheus metrics at `/metrics`, liveness at `/health`, readiness at `/ready`.
- Graceful SIGTERM shutdown (30s drain window).
- Docker image (Linux x86_64, CUDA 12.4+).
- Five ADRs documenting every architectural decision.
