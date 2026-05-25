# ADR 005 — OpenAI API deviations for diffusion serving

**Status:** Accepted (spec), 2026-05-23.
**Phase:** 1 spec; 3 implementation.

## Context

The serving niche we are claiming is "OSS production engine for diffusion
LLMs." Adoption requires the HTTP interface to feel like vLLM/Mercury —
i.e., OpenAI Chat Completions–compatible — even though several OpenAI
fields are autoregressive-only and don't translate.

This ADR is the spec the server implementation follows.

## Decision — what we keep, what we change

### Keep (OpenAI-compatible)

- `POST /v1/chat/completions`
- `model` field (we ignore it past the initial load; one engine = one
  model in v0.1)
- `messages` with `role` ∈ {`system`, `user`, `assistant`}
- `temperature`
- `seed`
- `stream` (with deviation, see below)
- `X-Request-ID` header: honored if sent; UUIDv7 generated otherwise and
  echoed in the response.

### Deviate (must be documented at `/docs` and in README)

- **`stream` emits whole blocks, not tokens.** Each SSE event contains
  the tokens committed in the most recent denoising step. Clients
  written for OpenAI token streaming will see jumps in chunk content.
  This is *fundamental* to diffusion, not fixable.
- **`max_tokens` ↦ `gen_length`** with the caveat that diffusion output
  is fixed-length (gen_length controls the canvas, not a stopping
  threshold). Document that LLaDA tends to emit EOS / padding tail when
  the answer is shorter than gen_length.
- **Diffusion-specific extension params are top-level fields in
  `ChatCompletionRequest`** (not `extra_body`). They are documented at
  `/docs` and rejected with HTTP 400 if unknown — no silent ignore.
  Current extension fields:
  - `num_denoising_steps` (default 128, ge=1, le=512; must divide
    `gen_length / block_length`)
  - `block_length` (default = `gen_length`, i.e., fully bidirectional)
  - `block_schedule` (`"uniform"` | `"front_loaded"` | `"cosine"`,
    default `"uniform"`)
  - `confidence_metric` (`"top1_prob"` | `"entropy_inverse"`, default
    `"top1_prob"`)
  - `target_length` (optional override for canvas size)
  - `attention_backend` (`"sdpa"` is the reference / deterministic
    path; `"fa2"` deferred until SM 12.0 FA2 build is in CI)
  - `force_single_batch` (true → no batching; use for reproducible output)
  - `use_local_leap` (bool, default false; enables LocalLeap
    anchor-propagation acceleration, arXiv:2510.07081)
  - `local_leap_anchor_threshold` (float 0–1, default 0.9; κ in paper)
  - `local_leap_neighbor_threshold` (float 0–1, default 0.75; τ in paper)
  - `local_leap_radius` (int 1–16, default 4; W in paper)

### Reject (autoregressive-only, no diffusion equivalent in v0.1)

- `top_p`, `top_k > 1` (sampling beyond temperature is out of scope for
  v0.1; the gate is `top_k = 1`).
- `presence_penalty`, `frequency_penalty` (per-token AR penalties; the
  diffusion analog would be position-aware penalties, deferred).
- `n > 1` (return multiple completions; trivially serializable but
  deferred).
- `logit_bias` (could be added later but no current demand).
- `tool_choice`, `functions`, `tools` (function calling explicitly out
  of scope for v0.1).
- `logprobs` returns a flat softmax of final-step logits at committed
  positions, **not** a per-step trace. Deferred.

## Consequences

- README ships a small "diffusion vs OpenAI semantics" table. Adoption
  cost for vLLM-style clients is documented up front.
- Failure mode: a client that hard-codes `top_p` or `n=4` will get a
  400 with an explanatory error referencing this ADR. We don't silently
  ignore unsupported fields — silent ignores would let bugs land in
  production.
- The deviations table is the **single most-read document for new
  users**. The README pins it, not the wiki.
