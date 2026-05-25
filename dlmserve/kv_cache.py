"""Committed/pending KV cache for diffusion denoising.

Currently inert: the denoise loop runs a full bidirectional forward each step,
matching the LLaDA reference. No K/V buffers are allocated. The class exists
so the engine API does not need to change when caching is enabled later.

Design notes (see also ADR 002):
* Per-layer storage: one slot per transformer layer.
* Contiguous allocation; no paging needed for single-GPU v0.1.
* Committed tokens are monotone (absorb-and-resample), so committed K/V can
  be appended and reused across denoising steps.
* Pending K/V is recomputed each step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype = torch.bfloat16


class DiffusionKVCache:
    """Committed/pending KV cache. Currently a no-op placeholder.

    Every accessor returns None; the denoise loop always falls back to a full
    forward pass. Wire in real storage when KV caching is enabled.
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.enabled = False
        self._committed_positions: dict[int, torch.Tensor] = {}
        log.debug(
            "KV cache init (inert)", extra={"num_layers": config.num_layers, "enabled": False}
        )

    def admit_request(self, req_id: int, seq_len: int) -> None:
        self._committed_positions[req_id] = torch.zeros(seq_len, dtype=torch.bool)

    def release_request(self, req_id: int) -> None:
        self._committed_positions.pop(req_id, None)

    def record_commit(self, req_id: int, positions: torch.Tensor) -> None:
        """Mark the given positions (1-D long tensor) as committed for this req."""
        if req_id not in self._committed_positions:
            return
        cur = self._committed_positions[req_id]
        cur[positions.to(cur.device)] = True

    def committed_mask(self, req_id: int) -> torch.Tensor | None:
        return self._committed_positions.get(req_id)

    def get_committed_kv(self, req_id: int, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Returns None until KV caching is enabled; callers always recompute."""
        return None

    def invalidate_pending(self, req_id: int) -> None:
        return None
