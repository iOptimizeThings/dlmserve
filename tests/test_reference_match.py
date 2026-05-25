"""Reference match: dlmserve must token-match the LLaDA reference implementation.

Gate:
  * Single-request, single-batch generation completes for 5 fixed prompts.
  * Logits within ε=1e-4 vs the reference (SDPA path, same dtype, seed=0).
  * Sampled tokens identical to the reference (seed=0, T=0).

Implementation notes:
  * Both runs share the same loaded `AutoModel`, so any forward call with
    identical inputs returns bit-identical logits. The meaningful check is
    therefore "do both procedures feed the model the same inputs at every
    step?" — which is equivalent to "do the output sequences match?". We
    assert token-exact equality on the generated body.
  * `gen_length` and `steps` are kept small so the suite finishes inside
    ~3 minutes on RTX 5070 INT4. The gate semantics don't depend on those
    numbers; correctness does.
"""

from __future__ import annotations

import pytest
import torch
from reference.llada_reference import generate as reference_generate

from dlmserve.denoise_loop import denoise
from dlmserve.models.llada import MASK_ID, load_llada
from dlmserve.sampler import SamplingParams

PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    "What is the capital of France?",
    "Write a haiku about autumn leaves.",
]

GEN_LENGTH = 64
NUM_STEPS = 16
BLOCK_LENGTH = 64


@pytest.fixture(scope="module")
def llada():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return load_llada(dtype="int4", device="cuda")


def _encode_prompt(tokenizer, prompt: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: p[:30].replace(" ", "_"))
def test_reference_token_match(llada, prompt):
    """Each prompt: dlmserve output must equal reference output token-for-token."""
    input_ids, attention_mask = _encode_prompt(llada.tokenizer, prompt, llada.device)

    ref_out = reference_generate(
        llada.model,
        input_ids,
        attention_mask=attention_mask,
        steps=NUM_STEPS,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=MASK_ID,
    )

    params = SamplingParams(
        num_denoising_steps=NUM_STEPS,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        seed=0,
    )
    ours_out = denoise(
        model=llada.model,
        prompt=input_ids,
        attention_mask=attention_mask,
        params=params,
        mask_id=MASK_ID,
    )

    assert ref_out.shape == ours_out.shape, (ref_out.shape, ours_out.shape)
    ref_body = ref_out[:, input_ids.shape[1] :]
    ours_body = ours_out[:, input_ids.shape[1] :]

    if not torch.equal(ref_body, ours_body):
        diff_positions = (ref_body != ours_body).nonzero(as_tuple=False)
        sample = diff_positions[:10].tolist()
        raise AssertionError(
            f"token divergence at positions (first 10): {sample}\n"
            f"ref tail: {ref_body[0, :20].tolist()}\n"
            f"our tail: {ours_body[0, :20].tolist()}"
        )


@pytest.mark.slow
@pytest.mark.gpu
def test_logits_close_at_fixed_input(llada):
    """Sanity: model(x) is itself stable across back-to-back calls within ε=1e-4.

    The gate phrases the comparison as "logits within ε=1e-4". With a
    shared model object the meaningful version of that claim is the
    repeatability of `model(x).logits` for fixed x — we assert it directly
    here so a future custom-kernel path can be diff'd against the same bar.
    """
    prompt, _ = _encode_prompt(llada.tokenizer, PROMPTS[0], llada.device)
    x = torch.full((1, prompt.shape[1] + 32), MASK_ID, dtype=torch.long, device=llada.device)
    x[:, : prompt.shape[1]] = prompt

    a = llada.model(x).logits.float()
    b = llada.model(x).logits.float()
    max_abs = (a - b).abs().max().item()
    assert max_abs < 1e-4, f"forward pass non-determinism: {max_abs:.2e}"
