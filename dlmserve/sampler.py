"""Confidence-based top-k token commit.

Derived from the LLaDA reference (`reference/llada_reference.py`, paper §3).
Each denoising step:
  1. Predict argmax tokens at every position (or Gumbel-sampled if T>0).
  2. Confidence at a masked position = softmax-prob of the predicted token
     (always from the raw logits, even when T>0 — this matches the reference).
  3. Mask off committed positions and positions past the active block.
  4. Take top-k by confidence per batch row; commit those positions.

The sampler is stateless: it takes the current sequence + logits and returns
the next sequence. The denoise loop owns iteration and the per-step k schedule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingParams:
    """Engine-facing sampling configuration.

    Defaults match the LLaDA reference at temperature=0, seed=0.
    """

    num_denoising_steps: int = 128
    gen_length: int = 128
    block_length: int | None = None
    temperature: float = 0.0
    seed: int = 0

    use_local_leap: bool = False
    # Paper-spec defaults for LLaDA-8B-Instruct (arXiv:2510.07081 §4.1, repo
    # friedrichor/LocalLeap scripts/llada/run.sh): κ=0.9, τ=0.75, W=4.
    # Paper's ablation explicitly says τ<0.75 "drops substantially."
    local_leap_anchor_threshold: float = 0.9
    local_leap_neighbor_threshold: float = 0.75
    local_leap_radius: int = 4


def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """LLaDA's Gumbel-max sampler at float64; identity at T=0."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    if generator is None:
        noise = torch.rand_like(logits, dtype=torch.float64)
    else:
        noise = torch.rand(
            logits.shape,
            dtype=torch.float64,
            device=logits.device,
            generator=generator,
        )
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def predict_and_score(
    logits: torch.Tensor,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predicted_tokens, confidence) of shape (B, L).

    `predicted_tokens` is argmax of (logits + optional Gumbel noise).
    `confidence` is softmax(logits)[predicted_tokens] — always from raw
    logits, even when T>0, exactly as the reference does.
    """
    if temperature > 0:
        x0 = add_gumbel_noise(logits, temperature, generator=generator).argmax(dim=-1)
    else:
        x0 = logits.argmax(dim=-1)
    p = logits.softmax(dim=-1)
    x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    return x0, x0_p


def commit_top_k_by_confidence(
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    k_per_row: torch.Tensor,
    block_end_abs: torch.Tensor | int | None = None,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one LLaDA-style commit step. Returns the updated sequence.

    Shapes:
        x:           (B, L) int64 — current sequence with mask_id at pending positions.
        logits:      (B, L, V) — model logits at every position.
        mask_index:  (B, L) bool — True at positions still to be committed.
        k_per_row:   (B,) or (B, 1) int — number of positions to commit this step.
        block_end_abs: scalar int, or (B,) int tensor — positions >= this are not
                       committed this step (semi-AR block restriction). Tensor form
                       needed when batching requests with different prompt lengths.
    """
    x0, x0_p = predict_and_score(logits, temperature=temperature, generator=generator)

    if block_end_abs is not None:
        L = x0_p.shape[1]
        if isinstance(block_end_abs, int):
            if block_end_abs < L:
                x0_p[:, block_end_abs:] = float("-inf")
        else:
            # Per-row tensor (B,) — build a (B, L) boolean mask
            end = block_end_abs.reshape(-1, 1)  # (B, 1)
            positions = torch.arange(L, device=x0_p.device).unsqueeze(0)  # (1, L)
            x0_p = x0_p.masked_fill(positions >= end, float("-inf"))

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.full((), float("-inf"), dtype=x0_p.dtype, device=x0_p.device)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    new_x = x.clone()
    k_flat = k_per_row.reshape(-1)
    for j in range(x.shape[0]):
        k = int(k_flat[j].item())
        if k <= 0:
            continue
        _, idx = torch.topk(confidence[j], k=k)
        new_x[j, idx] = x0[j, idx]
    return new_x


def commit_with_local_leap(
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    k_per_row: torch.Tensor,
    block_end_abs: torch.Tensor | int | None = None,
    *,
    anchor_threshold: float = 0.9,
    neighbor_threshold: float = 0.5,
    radius: int = 2,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """LocalLeap anchor-propagation commit.

    Same top-k anchors as `commit_top_k_by_confidence`. Additionally, for
    each anchor with confidence >= anchor_threshold, masked positions in
    [pos - radius, pos + radius] are also committed if their confidence is
    >= neighbor_threshold. This commits >= k tokens per step, ending blocks
    earlier than the baseline schedule.

    LocalLeap (Kong et al., arXiv:2510.07081, Apache-2.0). See CREDITS.md.
    Mathematically a strict superset of absorb-and-resample: positions only
    transition mask -> committed, never the reverse.
    """
    x0, x0_p = predict_and_score(logits, temperature=temperature, generator=generator)

    if block_end_abs is not None:
        L_loc = x0_p.shape[1]
        if isinstance(block_end_abs, int):
            if block_end_abs < L_loc:
                x0_p[:, block_end_abs:] = float("-inf")
        else:
            end = block_end_abs.reshape(-1, 1)
            positions = torch.arange(L_loc, device=x0_p.device).unsqueeze(0)
            x0_p = x0_p.masked_fill(positions >= end, float("-inf"))

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.full((), float("-inf"), dtype=x0_p.dtype, device=x0_p.device)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    new_x = x.clone()
    k_flat = k_per_row.reshape(-1)
    L = x.shape[1]
    offsets = torch.arange(-radius, radius + 1, device=x.device)

    for j in range(x.shape[0]):
        k = int(k_flat[j].item())
        if k <= 0:
            continue
        topk_vals, topk_idx = torch.topk(confidence[j], k=k)
        new_x[j, topk_idx] = x0[j, topk_idx]

        strong = topk_idx[topk_vals >= anchor_threshold]
        if strong.numel() == 0:
            continue

        neighbor_pos = (strong.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, L - 1).flatten()
        neighbor_pos = torch.unique(neighbor_pos)
        keep = confidence[j, neighbor_pos] >= neighbor_threshold
        commit_pos = neighbor_pos[keep]
        if commit_pos.numel() > 0:
            new_x[j, commit_pos] = x0[j, commit_pos]
    return new_x


def compute_transfer_schedule(mask_count: torch.Tensor, steps: int) -> torch.Tensor:
    """Per-step token-commit counts, replicating LLaDA's `get_num_transfer_tokens`.

    Args:
        mask_count: (B,) or (B, 1) int — number of masked positions in the active block.
        steps:      number of denoising steps to spread the commits over.
    Returns:
        (B, steps) int64 — number of commits at each step. Sums to mask_count per row.
    """
    mask_count = mask_count.reshape(-1, 1).to(torch.int64)
    base = mask_count // steps
    remainder = mask_count % steps
    sched = (
        torch.zeros(mask_count.size(0), steps, device=mask_count.device, dtype=torch.int64) + base
    )
    for i in range(mask_count.size(0)):
        sched[i, : int(remainder[i].item())] += 1
    return sched
