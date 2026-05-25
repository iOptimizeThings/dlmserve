# ADR 003 — Scheduler: step/block bucketing with hard preemption ban

**Status:** Accepted (shipped), 2026-05-23.

## Context

Continuous batching for diffusion is **stricter** than for autoregressive
serving. Requests can only batch when:

1. They are at the **same denoising step** (mid-step batching destroys the
   per-step semantics — confidence ranking is per-batch).
2. They share a **compatible block size** (`block_length`). Different
   `block_length` means different per-step token-commit count.
3. They share a **target length** bucket (close enough that the
   bidirectional attention's `O(L²)` cost doesn't blow up).

This rules out vLLM-style mixed prefill+decode batching outright.

The open questions:

1. Admit new requests vs flush a small running batch?
2. Block-size bucketing granularity?
3. Fairness / starvation prevention?
4. Preemption policy?

## Decision

### Bucket scheduler (shipped in v0.1)

Group ready requests by `(step_idx, block_length_bucket)`. Each tick
pick the largest group up to `MAX_BATCH`, run one denoising step over
that group, then reschedule. Achieved 2.74× throughput at batch=8.

Specific answers to the open questions:

1. **Admit new vs flush small:** flush a running batch when
   `batch_size ≥ MIN_BATCH = 4` **or** when `oldest_wait_ms > 50`. This
   trades a bit of batching efficiency for tail latency.
2. **Bucket granularity:** powers-of-two block sizes
   `[32, 64, 128, 256, 512, 1024]`. Per-model tuning is overkill.
3. **Fairness:** age every waiting request. Increase its effective
   bucket-match priority by +1 per 100 ms waited. A request stuck at a
   rare `(step, block)` combo will eventually run alone.
4. **Preemption:** never. Committed denoising state is expensive (entire
   step's worth of work) and discarding it is strictly bad. New
   requests wait.

### LocalLeap extension (shipped in v0.1)

`group_key` extended from `(step_idx, block_length_bucket)` to
`(step_idx, block_length_bucket, use_local_leap)`. A LocalLeap request
batched with a non-LocalLeap request would propagate anchor commits to
the latter, violating the absorb-and-resample contract for that request.
Adding `use_local_leap` to the key ensures the two populations never
share a batch.

## Consequences

- Quality verified at `batch_size = 8`: MMLU within 1pp, HumanEval within
  2pp, BLEU ≥ 0.95 of single-request baseline.
- Throughput gate was ≥ 2.5× single-request at batch 8 (revised from
  initial 3× estimate); actual result: **2.74×**.
- Anti-starvation is mechanical (aging counter), not an SLA promise.
  Real SLA enforcement is deferred to a later release.
