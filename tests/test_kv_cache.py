"""Tests for the inert KV cache.

The cache is built but disabled (enabled=False). These tests pin its
current behavior so enabling it later doesn't silently break the engine.
"""

from __future__ import annotations

import torch

from dlmserve.kv_cache import CacheConfig, DiffusionKVCache


def _make_cache() -> DiffusionKVCache:
    return DiffusionKVCache(
        CacheConfig(num_layers=4, num_heads=8, head_dim=64, max_seq_len=128)
    )


def test_cache_is_disabled_by_default():
    cache = _make_cache()
    assert cache.enabled is False
    assert cache.get_committed_kv(req_id=0, layer=0) is None


def test_admit_record_release_lifecycle():
    cache = _make_cache()
    cache.admit_request(req_id=7, seq_len=10)
    assert cache.committed_mask(7) is not None
    assert cache.committed_mask(7).shape == (10,)
    cache.record_commit(7, torch.tensor([1, 3, 5]))
    mask = cache.committed_mask(7)
    assert mask[1].item() and mask[3].item() and mask[5].item()
    assert not mask[0].item()
    cache.release_request(7)
    assert cache.committed_mask(7) is None


def test_unknown_request_record_is_noop():
    cache = _make_cache()
    cache.record_commit(123, torch.tensor([0]))
    assert cache.committed_mask(123) is None
