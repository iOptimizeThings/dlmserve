# ADR 001 — LLaDA absorb-and-resample as the v0.1 math contract

**Status:** Accepted, 2026-05-23.
**Phase:** 1.

## Context

Diffusion language models do not share a single inference procedure. The
candidates we considered for v0.1:

| Family | Inference math | Representative model |
|---|---|---|
| Absorb-and-resample (discrete masked diffusion) | Iteratively unmask top-k confidence positions over a fixed-length canvas | LLaDA |
| Block-AR × within-block diffusion | Generate one block autoregressively, within-block diffusion to refine | BD3-LMs |
| Continuous score-matching | Reverse SDE / probability flow ODE | SEDD, Plaid |

Each carries different state, different KV-cache semantics, and different
scheduler constraints. Designing the engine against a union of all three
would force premature abstraction — and v0.1's whole bet is
shipping a credible MVP fast in a niche nobody else has claimed.

## Decision

v0.1 implements *only* the LLaDA absorb-and-resample process:

- Start from `[prompt | <mask> * gen_length]`.
- For `s = 0 .. S-1`, do a bidirectional forward pass, take the argmax
  prediction at every position, score each masked position by the softmax
  probability of its argmax token, commit the top-`k_s` masked positions
  with the highest scores, and never un-commit.
- The per-step `k_s` schedule is **the integer-division formula** used in
  LLaDA's `get_num_transfer_tokens`, not the `round(L_masked · Δt / t)`
  derivation. The two agree when `steps` divides `mask_count`
  and disagree otherwise; the reference is authoritative because the
  reference-match gate is token-exact match against it. See
  `dlmserve/sampler.py::compute_transfer_schedule`.

## Consequences

- **In-scope for v0.1**: low-confidence remasking, optional semi-AR
  `block_length`, mask token id 126336, temperature ≥ 0 (Gumbel-max at
  float64 per LLaDA paper).
- **Out of scope for v0.1** (deferred):
  re-noising committed tokens (BD3-LMs); classifier-free guidance;
  continuous diffusion; self-conditioning / x0 estimation; vision-language
  variants.
- **Engine implication**: the `committed` mask is monotone, the KV cache
  for committed tokens is stable across denoising steps, and the
  scheduler can group requests by `(step_idx, block_length)`.
- Adding a second math contract is a major-version break, not a minor
  one. v0.2 will revisit when BD3-LMs is ready.

## Gate result

`tests/test_reference_match.py` — 5 prompts, token-exact match
against `reference/llada_reference.py` (vendored verbatim from
ML-GSAI/LLaDA), seed=0, temperature=0, INT4 path, RTX 5070. Run
2026-05-23, 6/6 passed.

## Model zoo extension (2026-05-24)

LLaDA-1.5 was added as a second supported model. It shares the same
absorb-and-resample math contract; no changes to `denoise_loop.py` or
`sampler.py` were required. Quality gates pass (token-exact, MMLU, HumanEval).
