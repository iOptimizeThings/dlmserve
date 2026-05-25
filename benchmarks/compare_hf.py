"""Compare dlmserve throughput against the HuggingFace reference implementation.

Loads the model once and runs three paths against the same weights:
  - HF reference: reference/llada_reference.py generate(), batch=1 sequential
  - dlmserve batch=1: Engine.generate(), one prompt at a time
  - dlmserve batch=N: Engine.generate(), all prompts at once

Usage:
    uv run python benchmarks/compare_hf.py
    uv run python benchmarks/compare_hf.py --model gsai-ml/LLaDA-1.5 --steps 16
"""

from __future__ import annotations

import argparse
import time

import torch

from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams
from reference.llada_reference import generate as hf_generate

DEFAULT_MODEL_ID = "gsai-ml/LLaDA-8B-Instruct"

TEST_PROMPTS = [
    "What is the capital of France?",
    "Write a Python function to reverse a string.",
    "Explain photosynthesis in one sentence.",
    "What is 7 times 8?",
]

N = len(TEST_PROMPTS)


def _encode(tokenizer: object, prompt: str, device: torch.device) -> torch.Tensor:
    msgs = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)  # type: ignore[union-attr]
    enc = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")  # type: ignore[operator]
    return enc["input_ids"].to(device)  # type: ignore[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--gen-length", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    print(f"Loading {args.model} (int4)...")
    engine = Engine(model_id=args.model, dtype="int4", max_batch=N)
    model = engine._loaded.model  # type: ignore[attr-defined]
    tokenizer = engine.tokenizer
    mask_id: int = engine._loaded.mask_id  # type: ignore[attr-defined]
    device = engine.device

    params = SamplingParams(
        num_denoising_steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.gen_length,
        temperature=0.0,
    )

    input_ids = [_encode(tokenizer, p, device) for p in TEST_PROMPTS]

    print("Warming up...")
    engine.generate([TEST_PROMPTS[0]], params)
    hf_generate(model, input_ids[0], steps=args.steps, gen_length=args.gen_length,
                block_length=args.gen_length, mask_id=mask_id)

    # HF reference: batch=1 sequential
    print(f"\nHF reference  (batch=1 × {N} sequential)...")
    t0 = time.perf_counter()
    for ids in input_ids:
        hf_generate(model, ids, steps=args.steps, gen_length=args.gen_length,
                    block_length=args.gen_length, mask_id=mask_id)
    hf_tps = (N * args.gen_length) / (time.perf_counter() - t0)
    print(f"  {hf_tps:.1f} tok/s")

    # dlmserve batch=1 sequential
    print(f"\ndlmserve      (batch=1 × {N} sequential)...")
    t0 = time.perf_counter()
    for p in TEST_PROMPTS:
        engine.generate([p], params)
    dlm_b1_tps = (N * args.gen_length) / (time.perf_counter() - t0)
    print(f"  {dlm_b1_tps:.1f} tok/s")

    # dlmserve batch=N
    print(f"\ndlmserve      (batch={N})...")
    t0 = time.perf_counter()
    engine.generate(TEST_PROMPTS, params)
    dlm_bN_tps = (N * args.gen_length) / (time.perf_counter() - t0)
    print(f"  {dlm_bN_tps:.1f} tok/s")

    print("\n" + "=" * 54)
    print(f"  {args.model}  steps={args.steps}  gen_length={args.gen_length}")
    print(f"  {'Path':<36} {'tok/s':>7}  {'vs HF':>6}")
    print("  " + "-" * 52)
    print(f"  {'HF reference (batch=1 sequential)':<36} {hf_tps:>7.1f}  {'1.00×':>6}")
    print(f"  {'dlmserve batch=1 sequential':<36} {dlm_b1_tps:>7.1f}  {dlm_b1_tps/hf_tps:>5.2f}×")
    print(f"  {'dlmserve batch=' + str(N):<36} {dlm_bN_tps:>7.1f}  {dlm_bN_tps/hf_tps:>5.2f}×")
    print("=" * 54)


if __name__ == "__main__":
    main()
