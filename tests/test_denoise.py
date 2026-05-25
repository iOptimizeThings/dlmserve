"""CPU-only tests for denoise-loop helpers.

The end-to-end gate lives in `test_reference_match.py` and requires a
GPU + model weights. This file targets the math primitives that don't.
"""

from __future__ import annotations

import torch
from reference.llada_reference import get_num_transfer_tokens

from dlmserve.sampler import (
    SamplingParams,
    commit_top_k_by_confidence,
    compute_transfer_schedule,
)


def test_compute_transfer_schedule_matches_reference():
    """LLaDA's get_num_transfer_tokens has subtle integer-division rules
    (base + 1 for the first `remainder` rows). Verify a few shapes."""
    for mask_count, steps in [(128, 16), (100, 16), (1, 8), (64, 4), (33, 5)]:
        mask = torch.zeros(2, mask_count + 5, dtype=torch.bool)
        mask[:, :mask_count] = True
        ref = get_num_transfer_tokens(mask, steps)
        ours = compute_transfer_schedule(mask.sum(dim=1), steps)
        assert torch.equal(ref, ours), (mask_count, steps, ref, ours)
        # Each row's per-step counts must sum back to that row's mask count.
        assert torch.equal(ours.sum(dim=1), mask.sum(dim=1).to(torch.int64))


def test_commit_top_k_commits_only_masked_positions():
    """A committed position must stay put regardless of its predicted token."""
    mask_id = 999
    x = torch.tensor([[1, mask_id, 2, mask_id, mask_id]])
    # Logits force argmax = 7 at every position.
    logits = torch.full((1, 5, 10), -10.0)
    logits[..., 7] = 10.0
    mask_index = x == mask_id
    k = torch.tensor([2])
    new_x = commit_top_k_by_confidence(x, logits, mask_index, k)
    # Position 0 and 2 must keep their original tokens.
    assert new_x[0, 0].item() == 1
    assert new_x[0, 2].item() == 2
    # Two masked positions filled, one still masked.
    assert (new_x[0] == 7).sum().item() == 2
    assert (new_x[0] == mask_id).sum().item() == 1


def test_commit_respects_block_end_clip():
    """Positions past block_end must not be committed even if masked."""
    mask_id = 999
    x = torch.full((1, 6), mask_id, dtype=torch.long)
    logits = torch.full((1, 6, 10), -10.0)
    logits[..., 5] = 10.0
    mask_index = x == mask_id
    new_x = commit_top_k_by_confidence(x, logits, mask_index, torch.tensor([3]), block_end_abs=3)
    # Only positions [0, 1, 2] can be committed; [3, 4, 5] stay masked.
    assert (new_x[0, :3] == 5).all()
    assert (new_x[0, 3:] == mask_id).all()


def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.temperature == 0.0
    assert p.seed == 0
    assert p.num_denoising_steps == 128
    assert p.gen_length == 128
