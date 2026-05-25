"""LocalLeap anchor-propagation tests (CPU-only).

End-to-end speedup measurement requires the LLaDA model and lives in
`benchmarks/perf_ci.py --use-local-leap`. These tests verify the algorithm
itself: anchors are committed, neighbors propagate, masked-only invariant
holds, and the `use_local_leap=False` path produces identical results to
`commit_top_k_by_confidence`.
"""

from __future__ import annotations

import torch

from dlmserve.sampler import (
    SamplingParams,
    commit_top_k_by_confidence,
    commit_with_local_leap,
)

MASK_ID = 999


def _logits_favoring(target_token: int, vocab: int = 10, confidence: float = 10.0) -> float:
    return confidence


def test_local_leap_off_is_identity_with_top_k():
    """With anchor_threshold > max confidence, no propagation occurs, so the
    result must equal `commit_top_k_by_confidence`."""
    torch.manual_seed(0)
    B, L, V = 2, 16, 32
    x = torch.full((B, L), MASK_ID, dtype=torch.long)
    x[:, 0] = 1  # one prompt token, rest masked
    logits = torch.randn(B, L, V)
    mask_index = x == MASK_ID
    k = torch.tensor([3, 4])

    baseline = commit_top_k_by_confidence(x, logits.clone(), mask_index, k)
    leap = commit_with_local_leap(
        x,
        logits.clone(),
        mask_index,
        k,
        anchor_threshold=1.1,  # > softmax max of 1.0, so no propagation
        neighbor_threshold=0.0,
        radius=3,
    )
    assert torch.equal(baseline, leap)


def test_local_leap_commits_neighbors_when_anchor_confident():
    """A highly confident anchor with confident neighbors should expand the
    committed set beyond the top-k count."""
    B, L, V = 1, 10, 20
    x = torch.full((B, L), MASK_ID, dtype=torch.long)
    logits = torch.full((B, L, V), -10.0)

    # Make position 5 a sharp anchor for token 7, with confidence ~1.0
    logits[0, 5, 7] = 50.0
    # Neighbors at 4 and 6: moderate confidence (~0.8 after softmax) for token 3
    logits[0, 4, 3] = 5.0
    logits[0, 6, 3] = 5.0
    # Positions 0..3 and 7..9: low confidence, near uniform
    # (already -10 everywhere — softmax is ~uniform)

    mask_index = x == MASK_ID
    k = torch.tensor([1])  # only 1 anchor via top-k

    out = commit_with_local_leap(
        x,
        logits,
        mask_index,
        k,
        anchor_threshold=0.5,
        neighbor_threshold=0.5,
        radius=2,
    )
    # Anchor at 5 committed to token 7
    assert out[0, 5].item() == 7
    # Neighbors 4 and 6 also committed to token 3 (their argmax)
    assert out[0, 4].item() == 3
    assert out[0, 6].item() == 3
    # Total committed > k=1
    assert (out[0] != MASK_ID).sum().item() >= 3


def test_local_leap_never_uncommits():
    """The absorb-and-resample invariant: committed positions stay committed."""
    B, L, V = 1, 8, 16
    x = torch.tensor([[1, 2, MASK_ID, MASK_ID, MASK_ID, MASK_ID, 3, 4]])
    # Force argmax = 5 at every masked position with high confidence
    logits = torch.full((B, L, V), -10.0)
    logits[..., 5] = 50.0
    mask_index = x == MASK_ID
    k = torch.tensor([1])

    out = commit_with_local_leap(
        x,
        logits,
        mask_index,
        k,
        anchor_threshold=0.5,
        neighbor_threshold=0.5,
        radius=3,
    )
    # Originally-committed positions must keep their tokens
    assert out[0, 0].item() == 1
    assert out[0, 1].item() == 2
    assert out[0, 6].item() == 3
    assert out[0, 7].item() == 4


def test_local_leap_respects_block_end():
    """Propagation must not commit past block_end_abs even from an anchor at
    the boundary."""
    B, L, V = 1, 10, 16
    x = torch.full((B, L), MASK_ID, dtype=torch.long)
    logits = torch.full((B, L, V), -10.0)
    # Strong anchor at position 4 with all neighbors also confident
    logits[0, 4, 7] = 50.0
    logits[0, 3, 7] = 50.0
    logits[0, 5, 7] = 50.0  # past block_end=5

    mask_index = x == MASK_ID
    k = torch.tensor([1])

    out = commit_with_local_leap(
        x,
        logits,
        mask_index,
        k,
        block_end_abs=5,  # positions [5..9] are out of block
        anchor_threshold=0.5,
        neighbor_threshold=0.5,
        radius=2,
    )
    assert out[0, 4].item() == 7
    assert out[0, 3].item() == 7
    # Position 5 is past block_end — must NOT be committed
    assert out[0, 5].item() == MASK_ID
    assert (out[0, 5:] == MASK_ID).all()


def test_local_leap_handles_boundary_anchors():
    """Anchors at position 0 or L-1 must not crash from negative or
    out-of-range neighbor indices."""
    B, L, V = 1, 5, 10
    x = torch.full((B, L), MASK_ID, dtype=torch.long)
    logits = torch.full((B, L, V), -10.0)
    logits[0, 0, 3] = 50.0   # anchor at left boundary
    logits[0, L - 1, 3] = 50.0  # anchor at right boundary

    mask_index = x == MASK_ID
    k = torch.tensor([2])
    out = commit_with_local_leap(
        x,
        logits,
        mask_index,
        k,
        anchor_threshold=0.5,
        neighbor_threshold=0.5,
        radius=3,
    )
    assert out[0, 0].item() == 3
    assert out[0, L - 1].item() == 3


def test_sampling_params_local_leap_defaults_off():
    """Defaults match paper-spec for LLaDA-8B-Instruct (arXiv:2510.07081 §4.1)."""
    p = SamplingParams()
    assert p.use_local_leap is False
    assert p.local_leap_anchor_threshold == 0.9
    assert p.local_leap_neighbor_threshold == 0.75
    assert p.local_leap_radius == 4
