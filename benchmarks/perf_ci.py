"""Performance benchmark — run after any throughput-affecting change.

Records tokens/sec, step duration, GPU mem peak, and cold-start time.
Results are printed and optionally appended to docs/perf_log.md.

Usage:
    uv run python benchmarks/perf_ci.py
    uv run python benchmarks/perf_ci.py --append-log   # writes to docs/perf_log.md
    uv run python benchmarks/perf_ci.py --use-local-leap --append-log
    DLMSERVE_TEST_MODEL=gsai-ml/LLaDA-1.5-8B-Instruct uv run python benchmarks/perf_ci.py --append-log

Model: defaults to gsai-ml/LLaDA-8B-Instruct; override with DLMSERVE_TEST_MODEL env var.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date
from pathlib import Path

import torch

from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

DEFAULT_MODEL_ID = "gsai-ml/LLaDA-8B-Instruct"

PERF_LOG = Path(__file__).parent.parent / "docs" / "perf_log.md"


def _params(use_local_leap: bool = False) -> SamplingParams:
    return SamplingParams(
        num_denoising_steps=16,
        gen_length=64,
        block_length=64,
        temperature=0.0,
        seed=0,
        use_local_leap=use_local_leap,
    )


PARAMS_512 = _params(use_local_leap=False)
TEST_PROMPTS = [
    "What is the capital of France?",
    "Write a Python function to reverse a string.",
    "Explain photosynthesis in one sentence.",
    "What is 7 times 8?",
    "Name three planets in our solar system.",
    "What is the speed of light?",
    "Write a haiku about autumn.",
    "What is the Pythagorean theorem?",
]


def _warmup(engine: Engine) -> None:
    engine.generate([TEST_PROMPTS[0]], PARAMS_512)


def benchmark_tokens_per_sec(engine: Engine, params: SamplingParams = PARAMS_512) -> dict[str, float]:
    _warmup(engine)

    # batch=1
    t0 = time.perf_counter()
    for p in TEST_PROMPTS[:4]:
        engine.generate([p], params)
    elapsed = time.perf_counter() - t0
    tps_batch1 = (4 * params.gen_length) / elapsed

    # batch=8
    t0 = time.perf_counter()
    engine.generate(TEST_PROMPTS, params)
    elapsed = time.perf_counter() - t0
    tps_batch8 = (8 * params.gen_length) / elapsed

    return {"tps_batch1": tps_batch1, "tps_batch8": tps_batch8}


def benchmark_step_duration(engine: Engine, params: SamplingParams = PARAMS_512) -> dict[str, float]:
    """Measure p50 and p99 step duration via wall-clock over a batch=4 run."""
    from dlmserve.denoise_loop import init_state, step_batch

    mask_id: int = engine._loaded.mask_id  # type: ignore[attr-defined]
    _warmup(engine)

    prompts = TEST_PROMPTS[:4]
    tok = engine.tokenizer
    states = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(rendered, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].to(engine.device)
        states.append(init_state(ids, params, mask_id=mask_id))

    durations: list[float] = []
    for _ in range(params.num_denoising_steps):
        # Stop early if every state has finished (LocalLeap can drain blocks
        # ahead of schedule). Otherwise step_batch handles done states fine
        # but the per-step timing becomes meaningless.
        if all(s.done for s in states):
            break
        t0 = time.perf_counter()
        step_batch(
            model=engine.model,
            states=states,
            mask_id=mask_id,
            pad_token_id=engine._pad_token_id(),
        )
        durations.append((time.perf_counter() - t0) * 1000)

    durations.sort()
    n = len(durations)
    p50 = durations[n // 2]
    p99 = durations[int(n * 0.99)]
    return {
        "step_p50_ms": p50,
        "step_p99_ms": p99,
        "p99_over_p50": p99 / p50,
        "num_steps_executed": n,
    }


def benchmark_gpu_mem(engine: Engine, params: SamplingParams = PARAMS_512) -> dict[str, float]:
    torch.cuda.reset_peak_memory_stats()
    engine.generate(TEST_PROMPTS, params)
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    return {"gpu_peak_gb": peak_gb}


def benchmark_cold_start(model_id: str) -> dict[str, float]:
    """Time from Engine() call to first token (model already on disk)."""
    t0 = time.perf_counter()
    eng = Engine(model_id=model_id, dtype="int4")
    eng.generate([TEST_PROMPTS[0]], PARAMS_512)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"cold_start_ms": elapsed_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument(
        "--use-local-leap",
        action="store_true",
        help="Enable LocalLeap and report speedup vs baseline.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    model_id = os.environ.get("DLMSERVE_TEST_MODEL", DEFAULT_MODEL_ID)
    print(f"Loading engine ({model_id})...")
    engine = Engine(model_id=model_id, dtype="int4", max_batch=8)

    print("Benchmarking tokens/sec (baseline)...")
    tps = benchmark_tokens_per_sec(engine, PARAMS_512)
    print(f"  batch=1: {tps['tps_batch1']:.1f} tok/s")
    print(f"  batch=8: {tps['tps_batch8']:.1f} tok/s")

    if args.use_local_leap:
        print("Benchmarking tokens/sec (LocalLeap)...")
        params_ll = _params(use_local_leap=True)
        tps_ll = benchmark_tokens_per_sec(engine, params_ll)
        print(f"  batch=1: {tps_ll['tps_batch1']:.1f} tok/s")
        print(f"  batch=8: {tps_ll['tps_batch8']:.1f} tok/s")
        speedup1 = tps_ll["tps_batch1"] / tps["tps_batch1"]
        speedup8 = tps_ll["tps_batch8"] / tps["tps_batch8"]
        print(f"  LocalLeap speedup: batch=1 {speedup1:.2f}×  batch=8 {speedup8:.2f}×")
        tps["tps_batch1_local_leap"] = tps_ll["tps_batch1"]
        tps["tps_batch8_local_leap"] = tps_ll["tps_batch8"]
        tps["local_leap_speedup_b1"] = speedup1
        tps["local_leap_speedup_b8"] = speedup8

    print("Benchmarking step duration (baseline)...")
    steps = benchmark_step_duration(engine, PARAMS_512)
    print(f"  p50: {steps['step_p50_ms']:.1f} ms  p99: {steps['step_p99_ms']:.1f} ms  ratio: {steps['p99_over_p50']:.2f}×  ({steps['num_steps_executed']} steps)")

    print("Benchmarking GPU mem peak (baseline)...")
    mem = benchmark_gpu_mem(engine, PARAMS_512)
    print(f"  peak: {mem['gpu_peak_gb']:.2f} GB")

    steps_ll: dict[str, float] | None = None
    mem_ll: dict[str, float] | None = None
    if args.use_local_leap:
        params_ll = _params(use_local_leap=True)
        print("Benchmarking step duration (LocalLeap)...")
        steps_ll = benchmark_step_duration(engine, params_ll)
        print(f"  p50: {steps_ll['step_p50_ms']:.1f} ms  p99: {steps_ll['step_p99_ms']:.1f} ms  ratio: {steps_ll['p99_over_p50']:.2f}×  ({steps_ll['num_steps_executed']} steps)")

        print("Benchmarking GPU mem peak (LocalLeap)...")
        mem_ll = benchmark_gpu_mem(engine, params_ll)
        print(f"  peak: {mem_ll['gpu_peak_gb']:.2f} GB")

    # Thresholds
    print("\n--- CI gate check ---")
    ok = True
    if steps["p99_over_p50"] > 3:
        print(f"FAIL baseline step p99/p50 = {steps['p99_over_p50']:.2f} > 3")
        ok = False
    if mem["gpu_peak_gb"] > 11.8:
        print(f"FAIL baseline GPU peak {mem['gpu_peak_gb']:.2f} GB > 11.8 GB")
        ok = False
    if steps_ll is not None and steps_ll["p99_over_p50"] > 3:
        print(f"FAIL LocalLeap step p99/p50 = {steps_ll['p99_over_p50']:.2f} > 3")
        ok = False
    if mem_ll is not None and mem_ll["gpu_peak_gb"] > 11.8:
        print(f"FAIL LocalLeap GPU peak {mem_ll['gpu_peak_gb']:.2f} GB > 11.8 GB")
        ok = False
    if ok:
        print("All CI gate checks passed.")

    if args.append_log:
        rows = [
            f"| Tokens/sec batch=1 (baseline) | {tps['tps_batch1']:.1f} | — |",
            f"| Tokens/sec batch=8 (baseline) | {tps['tps_batch8']:.1f} | — |",
        ]
        if 'tps_batch1_local_leap' in tps:
            rows += [
                f"| Tokens/sec batch=1 (LocalLeap) | {tps['tps_batch1_local_leap']:.1f} | — |",
                f"| Tokens/sec batch=8 (LocalLeap) | {tps['tps_batch8_local_leap']:.1f} | — |",
                f"| LocalLeap speedup batch=1 | {tps['local_leap_speedup_b1']:.2f}× | ≥1.10× {'✓' if tps['local_leap_speedup_b1'] >= 1.10 else '✗'} |",
                f"| LocalLeap speedup batch=8 | {tps['local_leap_speedup_b8']:.2f}× | ≥1.10× {'✓' if tps['local_leap_speedup_b8'] >= 1.10 else '✗'} |",
            ]
        rows += [
            f"| Step duration p50 (baseline) | {steps['step_p50_ms']:.1f} ms | — |",
            f"| Step duration p99 (baseline) | {steps['step_p99_ms']:.1f} ms | p99/p50 ≤ 3× |",
            f"| p99/p50 ratio (baseline) | {steps['p99_over_p50']:.2f}× | {'✓' if steps['p99_over_p50'] <= 3 else '✗'} |",
            f"| Steps executed (baseline) | {steps['num_steps_executed']:.0f} | — |",
            f"| GPU mem peak (baseline) | {mem['gpu_peak_gb']:.2f} GB | ≤ 11.8 GB {'✓' if mem['gpu_peak_gb'] <= 11.8 else '✗'} |",
        ]
        if steps_ll and mem_ll:
            rows += [
                f"| Step duration p50 (LocalLeap) | {steps_ll['step_p50_ms']:.1f} ms | — |",
                f"| Step duration p99 (LocalLeap) | {steps_ll['step_p99_ms']:.1f} ms | p99/p50 ≤ 3× |",
                f"| p99/p50 ratio (LocalLeap) | {steps_ll['p99_over_p50']:.2f}× | {'✓' if steps_ll['p99_over_p50'] <= 3 else '✗'} |",
                f"| Steps executed (LocalLeap) | {steps_ll['num_steps_executed']:.0f} | (fewer than baseline = anchor propagation working) |",
                f"| GPU mem peak (LocalLeap) | {mem_ll['gpu_peak_gb']:.2f} GB | ≤ 11.8 GB {'✓' if mem_ll['gpu_peak_gb'] <= 11.8 else '✗'} |",
            ]
        table = "\n".join(rows)
        entry = f"\n---\n\n## Performance benchmark — {date.today()}\n\nHardware: RTX 5070 12 GB, INT4\nModel: {model_id}\nScript: `benchmarks/perf_ci.py`\n\n| Metric | Value | Gate |\n|---|---|---|\n{table}\n"
        with open(PERF_LOG, "a") as f:
            f.write(entry)
        print(f"\nAppended to {PERF_LOG}")


if __name__ == "__main__":
    main()
