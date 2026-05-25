"""Bidirectional attention via PyTorch SDPA.

LLaDA uses no causal mask: every position attends to every other position.
FlashAttention-2 is not used by default; SM 12.0 (RTX 5070) requires a source
build. SDPA measures at 0.32 ms/iter for our shapes (docs/perf_log.md).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def bidirectional_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with no causal mask.

    Shapes follow PyTorch SDPA conventions:
        q, k, v: (B, H, L, D) with H = num_heads, D = head_dim.
        attn_mask: optional additive mask broadcastable to (B, H, L, L).
                   Use None for full bidirectional (the LLaDA default).
    """
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
    )
