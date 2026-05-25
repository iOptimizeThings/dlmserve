# Benchmarks

All numbers are from `docs/perf_log.md`. Reproduce with `benchmarks/perf_ci.py`.

## Hardware

RTX 5070 (12 GB, SM 12.0 Blackwell consumer), CUDA 13.0, torch 2.12.0+cu130.
INT4 (bitsandbytes nf4, bf16 compute). Commit: `192066e` (2026-05-24).

## LLaDA-8B-Instruct

| Mode | Tokens/sec | Step p50 | Step p99 | GPU peak |
|---|---|---|---|---|
| Batch=1, baseline | 32.9 tok/s | 192.1 ms | 192.3 ms | 5.64 GB |
| Batch=8, baseline | 110.6 tok/s | — | — | 5.64 GB |
| Batch=1, +LocalLeap | 58.1 tok/s | 192.4 ms | 192.6 ms | 5.64 GB |
| Batch=8, +LocalLeap | 146.8 tok/s | — | — | 5.64 GB |

LocalLeap steps executed: 11 vs 16 baseline.

## LLaDA-1.5

| Mode | Tokens/sec | Step p50 | Step p99 | GPU peak |
|---|---|---|---|---|
| Batch=1, baseline | 32.8 tok/s | 192.1 ms | 192.4 ms | 5.64 GB |
| Batch=8, baseline | 110.3 tok/s | — | — | 5.64 GB |
| Batch=1, +LocalLeap | 56.5 tok/s | 193.9 ms | 198.4 ms | 5.64 GB |
| Batch=8, +LocalLeap | 147.2 tok/s | — | — | 5.64 GB |

LocalLeap steps executed: 12 vs 16 baseline.

## Speedup summary (LLaDA-8B-Instruct, all vs batch=1 baseline)

| Comparison | Speedup |
|---|---|
| Batch=8 vs batch=1 | 3.4× |
| Batch=1 + LocalLeap vs batch=1 | 1.77× |
| Batch=8 + LocalLeap vs batch=1 | 4.5× |

## How to reproduce

```bash
# LLaDA-8B-Instruct
uv run python benchmarks/perf_ci.py --use-local-leap --append-log

# LLaDA-1.5
DLMSERVE_TEST_MODEL=gsai-ml/LLaDA-1.5 uv run python benchmarks/perf_ci.py --use-local-leap --append-log
```

Results append to `docs/perf_log.md` with date and model.
All reported numbers use `seed=0, temperature=0, steps=16, gen_length=64`.

## Comparison baseline

The only valid apples-to-apples comparison is dlmserve vs the upstream HF
reference loop (`reference/llada_reference.py`, vendored from `gsai-ml/LLaDA`).
There is no other OSS framework that serves LLaDA-8B-Instruct in masked-diffusion
mode as of May 2026 (SGLang serves LLaDA-2.0 block-diffusion only; vLLM feature
request #18532 closed unimplemented).
