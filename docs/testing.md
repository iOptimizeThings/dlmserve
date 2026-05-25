# Testing guide

Test suite overview and per-model run commands.

Tests are split into CPU-only (no GPU required, fast) and GPU (`-m "slow and gpu"`, requires
`CUDA_VISIBLE_DEVICES` set). GPU tests load the model once per session via the session-scoped
`engine` fixture in `tests/conftest.py` — do not load a second model in the same pytest session
or you will OOM on 12 GB VRAM.

---

## CPU-only tests (no model, no GPU)

Fast unit tests for pure-Python logic. Run on any machine without model weights.

```bash
uv run pytest tests/test_denoise.py tests/test_scheduler.py tests/test_kv_cache.py \
    tests/test_local_leap.py tests/test_env_config.py -v
```

---

## GPU tests by model

### LLaDA-8B-Instruct (default)

Full quality suite — all tests apply including batched BLEU.

```bash
# Reference-match gate
uv run pytest tests/test_reference_match.py -m "slow and gpu" -v

# Quality gate (bit-exact single + batched, BLEU ≥ 0.95, determinism)
uv run pytest tests/test_quality.py -m "slow and gpu" -v

# Batch regression
uv run pytest tests/test_batch_regression.py -m "slow and gpu" -v

# MMLU (downloads ~100 MB on first run)
uv run pytest tests/test_mmlu.py -m "slow and gpu" -v

# HumanEval (downloads ~1 MB on first run)
uv run pytest tests/test_humaneval.py -m "slow and gpu" -v

# LocalLeap quality gate (MMLU, HumanEval, BLEU vs baseline, determinism)
uv run pytest tests/test_local_leap_quality.py -m "slow and gpu" -v
```

Full suite in one command:

```bash
uv run pytest tests/ -m "slow and gpu" -v
```

---

## Performance benchmarks

`perf_ci.py` respects `DLMSERVE_TEST_MODEL` and reads `mask_id` from the loaded model — works for any model.

```bash
# LLaDA baseline + LocalLeap
uv run python benchmarks/perf_ci.py --use-local-leap --append-log

# HF reference comparison (dlmserve batch=1 and batch=4 vs raw HF generate())
uv run python benchmarks/compare_hf.py
uv run python benchmarks/compare_hf.py --model gsai-ml/LLaDA-1.5

# Soak test (100 concurrent, 10 min — requires server running separately)
uv run python benchmarks/soak_test.py --url http://localhost:8000 --duration 600 --concurrency 100
```

---

## perf_log.md status

| Model | Baseline | LocalLeap | Notes |
|---|---|---|---|
| LLaDA-8B-Instruct INT4 | ✓ 2026-05-24 | ✓ 2026-05-24 | See `docs/perf_log.md` |
| LLaDA-1.5 INT4 | ✓ 2026-05-24 | ✓ 2026-05-24 | See `docs/perf_log.md` |

---

## Test file index

| File | GPU? | What it covers |
|---|---|---|
| `test_denoise.py` | no | `init_state`, `step_batch` logic, schedule math |
| `test_scheduler.py` | no | bucket scheduler, step bucketing |
| `test_kv_cache.py` | no | KV cache placeholder API |
| `test_local_leap.py` | no | `commit_with_local_leap` unit tests |
| `test_env_config.py` | no | `DLMSERVE_*` env var parsing |
| `test_reference_match.py` | yes | LLaDA token-exact vs HF reference |
| `test_batch_regression.py` | yes | batch=8 throughput gate |
| `test_quality.py` | yes | bit-exact single/batched, BLEU ≥ 0.95, determinism |
| `test_mmlu.py` | yes | MMLU accuracy within 1% |
| `test_humaneval.py` | yes | HumanEval pass@1 within 2pp |
| `test_local_leap_quality.py` | yes | LocalLeap MMLU, HumanEval, BLEU, determinism |
