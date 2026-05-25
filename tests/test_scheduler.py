"""Tests for the bucketing scheduler.

The scheduler is pure-Python — no model calls here. GPU integration tests
live in test_batch_regression.py.
"""

from __future__ import annotations

import torch

from dlmserve.denoise_loop import DenoiseState, init_state
from dlmserve.sampler import SamplingParams
from dlmserve.scheduler import (
    DiffusionScheduler,
    Request,
    _block_bucket,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    prompt_len: int = 10,
    gen_length: int = 128,
    block_length: int = 128,
    steps: int = 16,
    device: str = "cpu",
) -> DenoiseState:
    params = SamplingParams(
        num_denoising_steps=steps,
        gen_length=gen_length,
        block_length=block_length,
    )
    prompt = torch.zeros(1, prompt_len, dtype=torch.long)
    mask_id = 999
    return init_state(prompt, params, mask_id=mask_id)


def _make_req(
    sched: DiffusionScheduler,
    prompt_len: int = 10,
    block_length: int = 128,
) -> Request | None:
    state = _make_state(prompt_len=prompt_len, block_length=block_length)
    params = SamplingParams(block_length=block_length)
    return sched.admit(state, params)


# ---------------------------------------------------------------------------
# Block bucketing
# ---------------------------------------------------------------------------


def test_block_bucket_rounds_up():
    assert _block_bucket(1) == 32
    assert _block_bucket(32) == 32
    assert _block_bucket(33) == 64
    assert _block_bucket(128) == 128
    assert _block_bucket(1024) == 1024
    assert _block_bucket(2000) == 1024  # clamped to max


def test_all_standard_lengths_have_a_bucket():
    for bl in (32, 64, 128, 256, 512, 1024):
        assert _block_bucket(bl) == bl


# ---------------------------------------------------------------------------
# Admit + queue depth
# ---------------------------------------------------------------------------


def test_admit_returns_request_with_incrementing_ids():
    sched = DiffusionScheduler()
    a = _make_req(sched)
    b = _make_req(sched)
    c = _make_req(sched)
    assert a is not None and b is not None and c is not None
    assert (a.req_id, b.req_id, c.req_id) == (0, 1, 2)


def test_admit_respects_max_waiting():
    sched = DiffusionScheduler(max_waiting=2)
    assert _make_req(sched) is not None
    assert _make_req(sched) is not None
    assert _make_req(sched) is None  # queue full


def test_queue_depth_tracks_pending():
    sched = DiffusionScheduler()
    assert sched.queue_depth == 0
    _make_req(sched)
    assert sched.queue_depth == 1
    _make_req(sched)
    assert sched.queue_depth == 2


# ---------------------------------------------------------------------------
# Scheduling: grouping by (step_idx, block_length_bucket)
# ---------------------------------------------------------------------------


def test_schedule_returns_empty_when_no_requests():
    sched = DiffusionScheduler()
    assert sched.schedule() == []


def test_schedule_promotes_pending_to_active():
    sched = DiffusionScheduler()
    _make_req(sched, block_length=128)
    _make_req(sched, block_length=128)
    batch = sched.schedule()
    assert len(batch) >= 1
    assert sched.active_count >= 1


def test_schedule_groups_by_bucket():
    """Two requests with different block_lengths should not be batched together."""
    sched = DiffusionScheduler(max_batch=8)
    _make_req(sched, block_length=64)
    _make_req(sched, block_length=128)
    batch = sched.schedule()
    # Both may be promoted to active, but the chosen batch must be from one bucket
    buckets = {r.block_length_bucket for r in batch}
    assert len(buckets) == 1


def test_schedule_picks_largest_group():
    sched = DiffusionScheduler(max_batch=8)
    # 3 requests with block=128, 1 with block=64
    for _ in range(3):
        _make_req(sched, block_length=128)
    _make_req(sched, block_length=64)
    batch = sched.schedule()
    # Largest group is block=128
    assert all(r.block_length_bucket == 128 for r in batch)
    assert len(batch) == 3


def test_schedule_respects_max_batch():
    sched = DiffusionScheduler(max_batch=4)
    for _ in range(10):
        _make_req(sched, block_length=128)
    batch = sched.schedule()
    assert len(batch) <= 4


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def test_complete_removes_from_active():
    sched = DiffusionScheduler()
    _make_req(sched, block_length=128)
    batch = sched.schedule()
    assert sched.active_count == 1
    req = batch[0]
    req.result = "done"
    sched.complete(req)
    assert sched.active_count == 0


def test_has_work_clears_after_all_done():
    sched = DiffusionScheduler()
    _make_req(sched)
    assert sched.has_work()
    batch = sched.schedule()
    for req in batch:
        sched.complete(req)
    assert not sched.has_work()


# ---------------------------------------------------------------------------
# Step index
# ---------------------------------------------------------------------------


def test_step_idx_reflects_state():
    state = _make_state(gen_length=128, block_length=64, steps=16)
    # 2 blocks, 8 steps each → steps_per_block=8
    params = SamplingParams(num_denoising_steps=16, gen_length=128, block_length=64)
    sched = DiffusionScheduler()
    req = sched.admit(state, params)
    assert req is not None
    assert req.step_idx == 0  # block_idx=0, step_in_block=0

    state.step_in_block = 3
    assert req.step_idx == 3

    state.block_idx = 1
    state.step_in_block = 0
    assert req.step_idx == 8  # 1*8 + 0
