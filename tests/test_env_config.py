"""Unit tests for DLMSERVE_* env var configuration.

Tests cover the helper functions directly since module-level constants are
resolved at import time.
"""

from __future__ import annotations

import pytest


def test_int_env_default(monkeypatch):
    monkeypatch.delenv("DLMSERVE_MAX_BATCH", raising=False)
    from dlmserve.scheduler import _get_int_env

    assert _get_int_env("DLMSERVE_MAX_BATCH", 8) == 8


def test_int_env_override(monkeypatch):
    monkeypatch.setenv("DLMSERVE_MAX_BATCH", "16")
    from dlmserve.scheduler import _get_int_env

    assert _get_int_env("DLMSERVE_MAX_BATCH", 8) == 16


def test_int_env_malformed_raises(monkeypatch):
    monkeypatch.setenv("DLMSERVE_MAX_BATCH", "not_a_number")
    from dlmserve.scheduler import _get_int_env

    with pytest.raises(ValueError, match="DLMSERVE_MAX_BATCH"):
        _get_int_env("DLMSERVE_MAX_BATCH", 8)


def test_int_env_waiting_default(monkeypatch):
    monkeypatch.delenv("DLMSERVE_MAX_WAITING", raising=False)
    from dlmserve.scheduler import _get_int_env

    assert _get_int_env("DLMSERVE_MAX_WAITING", 256) == 256


def test_int_env_waiting_override(monkeypatch):
    monkeypatch.setenv("DLMSERVE_MAX_WAITING", "512")
    from dlmserve.scheduler import _get_int_env

    assert _get_int_env("DLMSERVE_MAX_WAITING", 256) == 512


def test_float_env_poll_default(monkeypatch):
    monkeypatch.delenv("DLMSERVE_LOOP_POLL_MS", raising=False)
    from dlmserve.engine import _get_float_env

    assert _get_float_env("DLMSERVE_LOOP_POLL_MS", 1.0) == pytest.approx(1.0)


def test_float_env_poll_override(monkeypatch):
    monkeypatch.setenv("DLMSERVE_LOOP_POLL_MS", "5.0")
    from dlmserve.engine import _get_float_env

    assert _get_float_env("DLMSERVE_LOOP_POLL_MS", 1.0) == pytest.approx(5.0)


def test_float_env_poll_malformed_raises(monkeypatch):
    monkeypatch.setenv("DLMSERVE_LOOP_POLL_MS", "bad")
    from dlmserve.engine import _get_float_env

    with pytest.raises(ValueError, match="DLMSERVE_LOOP_POLL_MS"):
        _get_float_env("DLMSERVE_LOOP_POLL_MS", 1.0)
