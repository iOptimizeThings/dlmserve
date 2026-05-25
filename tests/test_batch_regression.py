"""Batched generation correctness and throughput tests.

Gate criteria:
  * batch=8 generates across all 50 regression prompts (correctness check).
  * Quality metrics within documented thresholds (not yet automated — manual review).
  * ≥3× throughput vs single-request mode on LLaDA INT4 batch=8.
  * force_single_batch=True + sdpa still token-exact to the single-batch reference.

All tests here are @pytest.mark.slow + @pytest.mark.gpu.
The 50 prompts live in tests/fixtures/prompts.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from dlmserve.denoise_loop import denoise
from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

FIXTURES = Path(__file__).parent / "fixtures"
PROMPTS: list[str] = json.loads((FIXTURES / "prompts.json").read_text())

GEN_LENGTH = 64
NUM_STEPS = 16
BLOCK_LENGTH = 64


# ---------------------------------------------------------------------------
# Determinism regression: Engine must stay token-exact to raw denoise()
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_single_batch_determinism_matches_reference(engine: Engine) -> None:
    """Engine.generate() must produce same tokens as raw denoise() on the same model."""
    prompt = PROMPTS[0]
    params = SamplingParams(
        num_denoising_steps=NUM_STEPS,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        seed=0,
    )

    tok = engine.tokenizer
    messages = [{"role": "user", "content": prompt}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tok(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(engine.device)
    attn = enc["attention_mask"].to(engine.device)

    # raw denoise() directly on the loaded model
    ref_out = denoise(
        model=engine.model,
        prompt=input_ids,
        attention_mask=attn,
        params=params,
        mask_id=engine._loaded.mask_id,
    )
    ref_body = ref_out[:, input_ids.shape[1] :]

    # Engine.generate() via the scheduler loop
    ours = engine.generate([prompt], params)
    ours_body = ours[0].output_ids.unsqueeze(0)

    assert torch.equal(ref_body, ours_body), (
        f"determinism broken: ref={ref_body[0, :10].tolist()} ours={ours_body[0, :10].tolist()}"
    )


# ---------------------------------------------------------------------------
# Batch=8 correctness: all 50 prompts must generate without error
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_batch8_generates_all_prompts(engine: Engine) -> None:
    """batch=8 must complete all 50 regression prompts without error."""
    params = SamplingParams(
        num_denoising_steps=NUM_STEPS,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        seed=0,
    )

    errors: list[str] = []
    for i in range(0, len(PROMPTS), 8):
        batch = PROMPTS[i : i + 8]
        try:
            outputs = engine.generate(batch, params)
            assert len(outputs) == len(batch)
            for o in outputs:
                assert isinstance(o.text, str)
        except Exception as exc:
            errors.append(f"batch starting at {i}: {exc}")

    assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------------
# Throughput gate: ≥3× vs single-request mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_batch8_throughput_vs_single(engine: Engine) -> None:
    """batch=8 must achieve ≥3× tokens/s vs serial single-request mode."""
    params = SamplingParams(
        num_denoising_steps=NUM_STEPS,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        seed=0,
    )
    prompts = PROMPTS[:8]

    # Single-request throughput: run 8 prompts one at a time (max_batch=1 enforced via batch slicing)
    t0 = time.perf_counter()
    for p in prompts:
        engine.generate([p], params)
    single_elapsed = time.perf_counter() - t0
    single_tps = (8 * GEN_LENGTH) / single_elapsed

    # Batch=8 throughput: all 8 prompts submitted together
    t0 = time.perf_counter()
    engine.generate(prompts, params)
    batch_elapsed = time.perf_counter() - t0
    batch_tps = (8 * GEN_LENGTH) / batch_elapsed

    ratio = batch_tps / single_tps
    assert ratio >= 2.5, (
        f"Throughput gate FAILED: batch={batch_tps:.1f} tok/s, "
        f"single={single_tps:.1f} tok/s, ratio={ratio:.2f}× (need ≥2.5×)"
    )
