"""Shared pytest configuration and constants.

Set DLMSERVE_TEST_MODEL to override the model used in all slow/gpu tests.
Defaults to gsai-ml/LLaDA-8B-Instruct (the production model, already cached as INT4).

The session-scoped engine fixture loads the model once per pytest session.
LLaDA INT4 is ~5.9 GB VRAM; loading twice per session wastes 2-3 min.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Generator

import pytest
import torch

from dlmserve.engine import Engine

TEST_MODEL_ID = os.environ.get("DLMSERVE_TEST_MODEL", "gsai-ml/LLaDA-8B-Instruct")


@pytest.fixture(scope="session")
def engine() -> Generator[Engine, None, None]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    e = Engine(model_id=TEST_MODEL_ID, dtype="int4", max_batch=8)
    yield e
    del e
    gc.collect()
    torch.cuda.empty_cache()
