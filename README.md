# dlmserve

OpenAI-compatible HTTP serving for diffusion language models.
LLaDA-8B-Instruct and LLaDA-1.5 in v0.1. Dream-7B in v0.1.1 ([issue #1](../../issues/1)).

## Why

Diffusion LLMs use bidirectional attention, a fixed-length canvas, and confidence-ranked parallel commit — not the causal attention, growing KV cache, and one-token decode loop that mainstream serving engines are built around. dlmserve is designed around the diffusion contract directly: per-step batching, no KV reuse assumption, and per-row acceleration (LocalLeap) that composes with batching.

## Quick start

```bash
# Install
pipx install dlmserve

# Or if you're already inside a venv / conda env
pip install dlmserve

# Don't have pipx? One-time setup:
#   sudo apt install pipx && pipx ensurepath
#   (then open a new terminal)

# Serve LLaDA-8B-Instruct (downloads ~5.6 GB INT4 weights on first run)
dlmserve

# Or with Docker (no Python install needed)
docker run --gpus all -p 8000:8000 \
  -e DLMSERVE_MODEL=gsai-ml/LLaDA-8B-Instruct \
  ghcr.io/iOptimizeThings/dlmserve:latest

# Use it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gsai-ml/LLaDA-8B-Instruct","messages":[{"role":"user","content":"What is the capital of France?"}],"num_denoising_steps":16}'
```

Once running, interactive API docs are at `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc`. Prometheus metrics at `/metrics`.

## Examples

```bash
# Interactive multi-turn chat (loads model locally, no server, single user)
uv run python examples/chat.py
uv run python examples/chat.py --model gsai-ml/LLaDA-1.5 --local-leap

# Compare dlmserve throughput vs raw HuggingFace generate() (shows batching speedup)
uv run python benchmarks/compare_hf.py
```

`examples/chat.py` runs at batch=1 by design (one user, one prompt at a time). Batching is a *server* feature — it kicks in when multiple clients hit the running `dlmserve` HTTP server concurrently. To see batching numbers, run `compare_hf.py` or hit the server with concurrent requests.

## Performance (RTX 5070 12 GB, INT4)

| Mode | LLaDA-8B-Instruct | LLaDA-1.5 |
|---|---|---|
| Batch=1, baseline | 32.9 tok/s (1.01× HF ref) | 32.7 tok/s (1.00× HF ref) |
| Batch=4, baseline | 81.7 tok/s (2.52× HF ref) | 82.0 tok/s (2.51× HF ref) |
| Batch=8, baseline | 110.6 tok/s | 110.3 tok/s |
| Batch=1, +LocalLeap | 58.1 tok/s (~1.8× baseline) | 56.5 tok/s (~1.7× baseline) |
| Batch=8, +LocalLeap | 146.8 tok/s (~4.5× batch=1 baseline) | 147.2 tok/s |

dlmserve batch=1 matches the HF reference loop (`reference/llada_reference.py`) to within measurement noise — token-identical at `temperature=0` (proven by `tests/test_reference_match.py`). The throughput gain comes from step-level batching and optional LocalLeap, not from changing the math.

Full numbers, settings, and reproduction: [`docs/benchmarks.md`](docs/benchmarks.md) and [`docs/perf_log.md`](docs/perf_log.md).

## Supported models

| Model | Status | INT4 VRAM |
|---|---|---|
| `gsai-ml/LLaDA-8B-Instruct` | ✓ v0.1 | ~5.6 GB |
| `gsai-ml/LLaDA-1.5` | ✓ v0.1 | ~5.6 GB |
| `Dream-org/Dream-v0-Instruct-7B` | v0.1.1 ([#1](../../issues/1)) | ~5.6 GB |
| `diffusionfamily/diffullama` | v0.1.1 ([#3](../../issues/3)) | ~5.6 GB INT4 |
| `LLaDA-2.0 (inclusionAI)` | v0.1.1 ([#2](../../issues/2)) | — |

## Batching

Automatic continuous batching at the denoising-step level. Concurrent requests share a forward pass, capped by `DLMSERVE_MAX_BATCH` (default 8). LocalLeap composes per-row on top. Live batch-size distribution at `/metrics` (`dlmserve_step_batch_size`). Opt out with `force_single_batch: true` for bit-reproducible output.

## API

OpenAI-compatible `/v1/chat/completions` with documented deviations
([ADR 005](docs/adrs/005-openai-api-deviations.md)).

Diffusion-specific parameters (beyond OpenAI spec):

| Param | Default | Description |
|---|---|---|
| `num_denoising_steps` | 16 | More steps = higher quality, lower throughput. Range [1, 64]. |
| `block_length` | = `max_tokens` | Denoising block size. |
| `use_local_leap` | false | LocalLeap anchor-propagation acceleration ([arXiv:2510.07081](https://arxiv.org/abs/2510.07081)). |
| `force_single_batch` | false | Disable batching for reproducible output. |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DLMSERVE_MODEL` | `gsai-ml/LLaDA-8B-Instruct` | Model ID (HuggingFace). |
| `DLMSERVE_DTYPE` | `int4` | Weight dtype: `int4`, `fp16`, `bf16`. |
| `DLMSERVE_DEVICE` | `cuda` | Device. |
| `DLMSERVE_PORT` | `8000` | HTTP port. |
| `DLMSERVE_MAX_BATCH` | `8` | Max concurrent requests per step. |
| `DLMSERVE_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warning`, `error`. |

## Honest limitations

- **Linux x86_64 only** — macOS and Windows are not supported. Windows users: WSL2 with CUDA passthrough may work but is untested and unsupported.
- **Single-GPU only** — multi-GPU TP is deferred to v0.5+.
- **INT4 fits in 12 GB**; FP16 weights (~16 GB) need a 24 GB+ card.
- **Docker image targets SM 8.0–8.9** (A100, H100, RTX 3090/4090/A6000). **Blackwell GPUs (RTX 50-series, SM 12.0) are not supported by the bundled image** — PyTorch 2.5.1 has no SM 12.0 kernels yet. On Blackwell, install from source (`pip install dlmserve`) against a PyTorch nightly that ships SM 12.0. See [`docs/docker.md`](docs/docker.md).
- **Attention backend is PyTorch SDPA.** FlashAttention-2 is optional (`pip install dlmserve[attn]`) and HF will use it automatically when present — **but FA2 also lacks SM 12.0 kernels**, so Blackwell stays on SDPA regardless.
- **Per-step SSE streaming not yet implemented** — v0.1 emits one SSE chunk for the full output ([issue #5](../../issues/5)).
- **`max_tokens` is a canvas size**, not a stop threshold — generation always fills the canvas and truncates at the first EOS. See [ADR 005](docs/adrs/005-openai-api-deviations.md).

## Built on

- [LLaDA: Large Language Diffusion with mAsking](https://arxiv.org/abs/2502.09992) — Nie et al., 2025. Model weights: `gsai-ml/LLaDA-8B-Instruct` (MIT).
- [LocalLeap: Accelerating Diffusion Language Models via Local Determinism Propagation](https://arxiv.org/abs/2510.07081) — Klear Team, 2024. Apache-2.0.

Full attribution: [`CREDITS.md`](CREDITS.md).

## Roadmap

```
v0.1.1  Dream-7B, LLaDA-2.0, DiffuLLaMA INT4, Fast-dLLM KV cache, per-step SSE
v0.2    BD3-LMs block diffusion, AdaBlock-dLLM adaptive block size
v0.5+   Multi-GPU tensor parallelism
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Issues and PRs welcome.

## License

MIT — see [`LICENSE`](LICENSE).
