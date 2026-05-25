"""Quality metrics gate.

Tests:
  1. Bit-exact match (single-batch): engine output must match raw denoise() 50/50.
  2. Bit-exact match (batched, batch=8): same comparison in batch mode.
  3. BLEU ≥ 0.95: batched mode output vs single-request denoise() reference text.
  4. Determinism: same prompt run twice must be bit-exact.

No golden fixture files — the reference is always live denoise() on the same
loaded model. Works with any DLMSERVE_TEST_MODEL without pre-generation steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sacrebleu
import torch

from dlmserve.denoise_loop import denoise
from dlmserve.engine import Engine
from dlmserve.sampler import SamplingParams

FIXTURES = Path(__file__).parent / "fixtures"
PROMPTS: list[str] = json.loads((FIXTURES / "prompts.json").read_text())

GEN_LENGTH = 64
NUM_STEPS = 16
BLOCK_LENGTH = 64

PARAMS = SamplingParams(
    num_denoising_steps=NUM_STEPS,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
    temperature=0.0,
    seed=0,
)


def _run_reference(engine: Engine, prompt: str) -> tuple[list[int], str]:
    """Run raw denoise() on one prompt. Returns (token_ids, text)."""
    tok = engine.tokenizer
    loaded = engine._loaded
    messages = [{"role": "user", "content": prompt}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tok(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(engine.device)
    attn = enc["attention_mask"].to(engine.device)
    out = denoise(
        model=loaded.model,
        prompt=input_ids,
        attention_mask=attn,
        params=PARAMS,
        mask_id=loaded.mask_id,
    )
    body_ids = out[0, input_ids.shape[1] :].tolist()
    text = tok.decode(body_ids, skip_special_tokens=True)
    return body_ids, text


@pytest.fixture(scope="module")
def reference_outputs(engine: Engine) -> list[tuple[list[int], str]]:
    """Compute all 50 reference outputs via raw denoise() once per session."""
    results = []
    for prompt in PROMPTS:
        ids, text = _run_reference(engine, prompt)
        results.append((ids, text))
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Test 1: bit-exact single-batch vs live denoise() reference
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_bit_exact_single_batch(
    engine: Engine, reference_outputs: list[tuple[list[int], str]]
) -> None:
    """50/50 prompts: engine.generate() must match raw denoise() token-exact."""
    mismatches: list[str] = []

    for i, prompt in enumerate(PROMPTS):
        ref_ids, _ = reference_outputs[i]
        outputs = engine.generate([prompt], PARAMS)
        got_ids = outputs[0].output_ids.tolist()
        if got_ids != ref_ids:
            mismatches.append(f"[{i:02d}] got={got_ids[:8]} ref={ref_ids[:8]}")

    assert not mismatches, f"{len(mismatches)}/50 prompts failed bit-exact match:\n" + "\n".join(
        mismatches
    )


# ---------------------------------------------------------------------------
# Test 2: bit-exact batched (batch=8) vs live denoise() reference
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.xfail(
    reason="batched mode not guaranteed token-exact — LSB diffs from reduction order expected",
    strict=False,
)
def test_bit_exact_batched(engine: Engine, reference_outputs: list[tuple[list[int], str]]) -> None:
    """50/50 prompts: batch=8 engine output vs raw denoise() reference.

    Note: batched mode is not guaranteed token-exact — LSB-level
    diffs from reduction order are expected. BLEU test is the real gate.
    """
    mismatches: list[str] = []

    for i in range(0, len(PROMPTS), 8):
        batch_prompts = PROMPTS[i : i + 8]
        outputs = engine.generate(batch_prompts, PARAMS)
        for j, output in enumerate(outputs):
            idx = i + j
            ref_ids, _ = reference_outputs[idx]
            got_ids = output.output_ids.tolist()
            if got_ids != ref_ids:
                mismatches.append(f"[{idx:02d}] got={got_ids[:8]} ref={ref_ids[:8]}")

    assert not mismatches, (
        f"{len(mismatches)}/50 prompts failed bit-exact match in batch mode:\n"
        + "\n".join(mismatches)
        + "\nNote: batched mode is not guaranteed token-exact. "
        "Check BLEU test for the real quality signal."
    )


# ---------------------------------------------------------------------------
# Test 3: BLEU ≥ 0.95 batched mode vs live denoise() reference
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_bleu_batched(engine: Engine, reference_outputs: list[tuple[list[int], str]]) -> None:
    """Batched mode output must score BLEU ≥ 0.95 vs single-request denoise() reference."""
    hypotheses: list[str] = []
    references: list[str] = []

    for i in range(0, len(PROMPTS), 8):
        batch_prompts = PROMPTS[i : i + 8]
        outputs = engine.generate(batch_prompts, PARAMS)
        for j, output in enumerate(outputs):
            idx = i + j
            _, ref_text = reference_outputs[idx]
            hypotheses.append(output.text)
            references.append(ref_text)

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    score = result.score / 100.0

    assert score >= 0.95, (
        f"BLEU gate FAILED: {score:.4f} < 0.95\n"
        "Batched mode output has drifted from the single-batch reference. "
        "Investigate attention reduction order or softmax precision differences."
    )


# ---------------------------------------------------------------------------
# Test 4: determinism CI — run twice, bit-exact
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_determinism_run_twice(engine: Engine) -> None:
    """Same prompts run twice must produce bit-exact identical output."""
    mismatches: list[str] = []

    for i, prompt in enumerate(PROMPTS):
        out_a = engine.generate([prompt], PARAMS)
        out_b = engine.generate([prompt], PARAMS)
        ids_a = out_a[0].output_ids.tolist()
        ids_b = out_b[0].output_ids.tolist()
        if ids_a != ids_b:
            mismatches.append(f"[{i:02d}] run1={ids_a[:6]} run2={ids_b[:6]}")

    assert not mismatches, f"{len(mismatches)}/50 prompts are non-deterministic:\n" + "\n".join(
        mismatches
    )
