# ADR 004 — Single-process asyncio execution model

**Status:** Accepted, 2026-05-23.
**Phase:** 1 design, 3 implementation.

## Context

vLLM runs scheduler, worker, and API server across separate processes
with shared-memory queues. The split protects the API from worker
crashes and lets the worker run pure CUDA without GIL contention.

dlmserve could mimic that, or stay single-process. The trade-off:

| Aspect | Multi-process | Single-process asyncio |
|---|---|---|
| Debugger ergonomics | Worse (cross-process traceback hops) | Better |
| IPC overhead | Real (shared mem + serialize) | None |
| Failure isolation | Worker crash leaves API up | One crash = full restart |
| Code surface | ~2× (workers + IPC + supervisors) | Minimum |
| GIL contention | None — workers are subprocesses | Real, but the GPU forward is mostly C-extension code that releases the GIL anyway |

## Decision

**Single-process asyncio for v0.1.** One Python process holds the model,
the scheduler, and the FastAPI app. The denoise loop runs
synchronously under a single event loop; long GPU calls are awaited via
`asyncio.to_thread` so the loop doesn't block on `model(...)`.

## Consequences

- **No `torch.cuda.synchronize()` in the FastAPI handler.** Blocks the event loop; GPU calls go through a worker thread.
- **Crash recovery is process-level.** A model OOM or kernel fault
  takes the whole process down. Acceptable for v0.1; supervisord/k8s
  in front of the container gives auto-restart.
- **Memory budget is one process.** No duplicated weights across workers
  — important on 12 GB hardware.
- **Multi-engine pools (one process, multiple models) are deferred** to
  v0.5+. So is multi-GPU tensor parallelism. The single-process model
  is consciously the smallest thing that works.
