# ADR 002 — KV cache: per-layer committed/pending split, staged rollout

**Status:** Accepted, 2026-05-23.

## Context

Diffusion denoising re-evaluates the entire sequence at every step. Two
naïve approaches:

1. **No cache** — run a full bidirectional forward each step. Simple,
   matches the reference, but throws away the fact that committed tokens
   have stable K/V across steps.
2. **vLLM-style append-only KV cache** — wrong abstraction. Diffusion
   commits tokens at arbitrary positions, not strictly left-to-right, so
   the append-only assumption breaks.

The right structure is a *committed/pending split*: committed tokens have
K/V that does not change across steps; pending (masked) tokens are
re-evaluated every step and their K/V must be recomputed.

The open questions:

1. Per-layer storage or per-request?
2. Paged allocation, contiguous, or both?
3. Cross-step warm-start for pending K/V (LocalLeap)?

## Decision

**Per-layer storage.** Each transformer layer owns one slot per active
request. The model already iterates per-layer; per-request storage would
force gather/scatter across layers on every commit.

**Contiguous allocation in v0.1.** Single-request mode has no
fragmentation pressure. PagedAttention is deferred until profiled need.

**Committed K/V is monotone.** Tokens never un-commit (ADR 001), so the
cache can append committed K/V and reuse it indefinitely.

**No cross-step pending warm-start in v0.1.** LocalLeap-style pending
reuse is deferred, gated on a license audit.

**Staged rollout:**

- **v0.1 (current):** `DiffusionKVCache` is built and carried
  through the engine, but `enabled = False`. `get_committed_kv()` returns
  `None` and the denoise loop runs a full forward each step. This is what
  matches the reference token-exactly.
- **v0.1.1:** Fast-dLLM committed-token KV splice (researched, deferred).
  Monkey-patching HF LLaDA + bitsandbytes 4-bit linear was judged high-risk
  for marginal additional gain on top of LocalLeap (1.32–1.48× already
  landed). Will be reopened ONLY if LocalLeap underperforms in production.
- **v0.2:** block-level KV reuse as part of BD3-LMs math contract work.
  Scope-gated behind the BD3-LMs absorb-and-resample extension.

## Consequences

- v0.1 throughput is bounded by full-forward cost (the reference's
  cost). That's acceptable: the gate is correctness, not throughput.
- The cache API is stable and ready for future activation (`admit_request`,
  `release_request`, `record_commit`, `committed_mask`,
  `get_committed_kv`). Enabling it requires only flipping `enabled = True`;
  the engine does not need to change.
- The `enabled` flag is a one-bit dial; we won't ship multiple cache
  backends behind a `Backend` base class (anti-pattern #11).
