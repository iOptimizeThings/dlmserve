"""Continuous-batching scheduler for diffusion denoising (ADR 003).

  * Groups requests by (step_idx, block_length_bucket).
  * Buckets: [32, 64, 128, 256, 512, 1024] token block sizes.
  * Flushes a group when batch_size >= MIN_BATCH=4 or oldest_wait > 50 ms.
  * Anti-starvation: age each waiting request +1 priority per 100 ms.
  * No preemption; committed denoising state is too expensive to drop.

Driven from `Engine._engine_loop()`. Does not touch model weights or GPU state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import torch

from dlmserve.denoise_loop import DenoiseState
from dlmserve.sampler import SamplingParams

log = logging.getLogger(__name__)

BLOCK_BUCKETS: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
MIN_BATCH: int = 4
FLUSH_WAIT_MS: float = 50.0
AGING_PERIOD_MS: float = 100.0


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid integer") from exc


MAX_BATCH: int = _get_int_env("DLMSERVE_MAX_BATCH", 8)
MAX_WAITING: int = _get_int_env("DLMSERVE_MAX_WAITING", 256)


def _block_bucket(block_length: int) -> int:
    """Round block_length up to the nearest bucket size."""
    for b in BLOCK_BUCKETS:
        if block_length <= b:
            return b
    return BLOCK_BUCKETS[-1]


@dataclass
class Request:
    """A single generation request tracked by the scheduler."""

    req_id: int
    state: DenoiseState
    params: SamplingParams
    admitted_ms: float = field(default_factory=lambda: time.monotonic() * 1000)
    generator: torch.Generator | None = field(default=None, repr=False)
    result: Any = field(default=None, repr=False)
    _future: asyncio.Future[Any] | None = field(default=None, repr=False)

    @property
    def step_idx(self) -> int:
        """Global denoising step (0 … num_denoising_steps − 1)."""
        return self.state.block_idx * self.state.steps_per_block + self.state.step_in_block

    @property
    def block_length_bucket(self) -> int:
        return _block_bucket(self.state.block_length)

    @property
    def group_key(self) -> tuple[int, int, bool]:
        # use_local_leap must be in the bucket: a LocalLeap request batched
        # with a non-LocalLeap request would propagate anchor commits to the
        # latter, violating the absorb-and-resample contract for that user.
        return (self.step_idx, self.block_length_bucket, self.state.params.use_local_leap)


class DiffusionScheduler:
    """
    Groups running requests by (step_idx, block_length_bucket).
    Picks the group with the highest score to run next.
    """

    def __init__(
        self,
        max_batch: int = MAX_BATCH,
        max_waiting: int = MAX_WAITING,
    ) -> None:
        self.max_batch = max_batch
        self.max_waiting = max_waiting
        self._pending: deque[Request] = deque()
        self._active: list[Request] = []
        self._next_id: int = 0

    def admit(
        self,
        state: DenoiseState,
        params: SamplingParams,
        generator: torch.Generator | None = None,
        future: asyncio.Future[Any] | None = None,
    ) -> Request | None:
        """Enqueue a new request. Returns None if the queue is full (caller → HTTP 429)."""
        if len(self._pending) >= self.max_waiting:
            return None
        req = Request(
            req_id=self._next_id,
            state=state,
            params=params,
            generator=generator,
            _future=future,
        )
        self._next_id += 1
        self._pending.append(req)
        log.debug(
            "request queued",
            extra={
                "req_id": req.req_id,
                "prompt_len": state.prompt_len,
                "gen_length": state.gen_length,
                "steps": params.num_denoising_steps,
                "queue_depth": len(self._pending),
            },
        )
        return req

    def _promote_pending(self) -> None:
        """Move pending requests into the active pool up to 2×max_batch."""
        room = self.max_batch * 2 - len(self._active)
        while self._pending and room > 0:
            self._active.append(self._pending.popleft())
            room -= 1

    def schedule(self) -> list[Request]:
        """Return the best batch to run right now (up to max_batch requests)."""
        self._promote_pending()
        if not self._active:
            return []

        now_ms = time.monotonic() * 1000
        groups: dict[tuple[int, int, bool], list[Request]] = defaultdict(list)
        for req in self._active:
            groups[req.group_key].append(req)

        best_key: tuple[int, int, bool] | None = None
        best_score: float = -1.0

        for key, reqs in groups.items():
            oldest_wait_ms = max(now_ms - r.admitted_ms for r in reqs)
            age_bonus = oldest_wait_ms / AGING_PERIOD_MS
            score = float(len(reqs)) + age_bonus
            # Force-flush any group whose oldest member has waited too long
            if oldest_wait_ms > FLUSH_WAIT_MS:
                score += 1000.0
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            return []
        chosen = groups[best_key][: self.max_batch]
        log.debug(
            "batch scheduled",
            extra={
                "step_idx": best_key[0],
                "bucket": best_key[1],
                "batch_size": len(chosen),
                "groups": {str(k): len(v) for k, v in groups.items()},
            },
        )
        return chosen

    def complete(self, req: Request) -> None:
        """Remove a finished request and resolve its async future (if any)."""
        with suppress(ValueError):
            self._active.remove(req)
        log.debug("request completed", extra={"req_id": req.req_id})
        if req._future is not None and not req._future.done():
            req._future.set_result(req.result)

    def has_work(self) -> bool:
        return bool(self._pending or self._active)

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return len(self._active)
