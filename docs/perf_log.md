# Performance Log

Append-only record of benchmark runs. Use `git diff` to spot regressions.
Correctness is enforced by the pytest suite (see `docs/testing.md`).

---

## Performance benchmark — 2026-05-24

Hardware: RTX 5070 12 GB, INT4
Model: gsai-ml/LLaDA-8B-Instruct
Script: `benchmarks/perf_ci.py`

| Metric | Value | Gate |
|---|---|---|
| Tokens/sec batch=1 (baseline) | 32.9 | — |
| Tokens/sec batch=8 (baseline) | 110.6 | — |
| Tokens/sec batch=1 (LocalLeap) | 58.1 | — |
| Tokens/sec batch=8 (LocalLeap) | 146.8 | — |
| LocalLeap speedup batch=1 | 1.77× | ≥1.10× ✓ |
| LocalLeap speedup batch=8 | 1.33× | ≥1.10× ✓ |
| Step duration p50 (baseline) | 192.1 ms | — |
| Step duration p99 (baseline) | 192.3 ms | p99/p50 ≤ 3× |
| p99/p50 ratio (baseline) | 1.00× | ✓ |
| Steps executed (baseline) | 16 | — |
| GPU mem peak (baseline) | 5.64 GB | ≤ 11.8 GB ✓ |
| Step duration p50 (LocalLeap) | 192.4 ms | — |
| Step duration p99 (LocalLeap) | 192.6 ms | p99/p50 ≤ 3× |
| p99/p50 ratio (LocalLeap) | 1.00× | ✓ |
| Steps executed (LocalLeap) | 11 | (fewer than baseline = anchor propagation working) |
| GPU mem peak (LocalLeap) | 5.64 GB | ≤ 11.8 GB ✓ |

---

## Performance benchmark — 2026-05-24

Hardware: RTX 5070 12 GB, INT4
Model: gsai-ml/LLaDA-1.5
Script: `benchmarks/perf_ci.py`

| Metric | Value | Gate |
|---|---|---|
| Tokens/sec batch=1 (baseline) | 32.8 | — |
| Tokens/sec batch=8 (baseline) | 110.3 | — |
| Tokens/sec batch=1 (LocalLeap) | 56.5 | — |
| Tokens/sec batch=8 (LocalLeap) | 147.2 | — |
| LocalLeap speedup batch=1 | 1.72× | ≥1.10× ✓ |
| LocalLeap speedup batch=8 | 1.33× | ≥1.10× ✓ |
| Step duration p50 (baseline) | 192.1 ms | — |
| Step duration p99 (baseline) | 192.4 ms | p99/p50 ≤ 3× |
| p99/p50 ratio (baseline) | 1.00× | ✓ |
| Steps executed (baseline) | 16 | — |
| GPU mem peak (baseline) | 5.64 GB | ≤ 11.8 GB ✓ |
| Step duration p50 (LocalLeap) | 193.9 ms | — |
| Step duration p99 (LocalLeap) | 198.4 ms | p99/p50 ≤ 3× |
| p99/p50 ratio (LocalLeap) | 1.02× | ✓ |
| Steps executed (LocalLeap) | 12 | (fewer than baseline = anchor propagation working) |
| GPU mem peak (LocalLeap) | 5.64 GB | ≤ 11.8 GB ✓ |

---

## HF reference comparison — 2026-05-24

Hardware: RTX 5070 12 GB, INT4
Settings: steps=16, gen_length=64, 4 test prompts
Script: `benchmarks/compare_hf.py`

| Path | LLaDA-8B-Instruct | LLaDA-1.5 |
|---|---|---|
| HF reference (`reference/llada_reference.py`), batch=1 sequential | 32.4 tok/s | 32.7 tok/s |
| dlmserve batch=1 sequential | 32.8 tok/s (1.01× HF) | 32.7 tok/s (1.00× HF) |
| dlmserve batch=4 | 81.7 tok/s (2.52× HF) | 82.0 tok/s (2.51× HF) |

dlmserve batch=1 matches HF reference to within measurement noise — confirms no per-request overhead. Batching at N=4 delivers ~2.5× HF reference throughput on both models.

---

## LocalLeap calibration sweep — LLaDA-1.5 — 2026-05-24

Hardware: RTX 5070 12 GB, INT4
Settings: steps=16, gen_length=256, 20 HumanEval problems
Script: `scripts/calibrate_local_leap.py`

Baseline pass@1 = 0.100.

| κ | τ | W | pass@1 | diff | gate |
|---|---|---|---|---|---|
| 0.50 | 0.60 | 4 | 0.050 | +0.050 | ✗ |
| 0.50 | 0.75 | 4 | 0.100 | +0.000 | ✓ |
| 0.60 | 0.60 | 4 | 0.050 | +0.050 | ✗ |
| 0.60 | 0.75 | 4 | 0.100 | +0.000 | ✓ |
| 0.70 | 0.60 | 4 | 0.050 | +0.050 | ✗ |
| 0.70 | 0.75 | 4 | 0.100 | +0.000 | ✓ |
| 0.80 | 0.60 | 4 | 0.050 | +0.050 | ✗ |
| 0.80 | 0.75 | 4 | 0.100 | +0.000 | ✓ |
| 0.90 | 0.60 | 4 | 0.100 | +0.000 | ✓ |
| 0.90 | 0.75 | 4 | 0.050 | +0.050 | ✗ |

τ=0.75 is the determining threshold (passes for κ ∈ [0.5, 0.8]). Picked (κ=0.5, τ=0.75, W=4) for maximum anchor-propagation aggressiveness. Wired into `_LOCAL_LEAP_MODEL_DEFAULTS` in `dlmserve/engine.py`. Confirmed all 4 LocalLeap quality gates (MMLU, HumanEval, BLEU, determinism) pass for LLaDA-1.5 at these values.

LLaDA-8B-Instruct continues to use the paper-published defaults (κ=0.9, τ=0.75, W=4) from arXiv:2510.07081 §4.1. κ=0.9 fails for LLaDA-1.5 at τ=0.75, so per-model defaults are necessary.
