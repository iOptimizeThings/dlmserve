"""Prometheus metrics exporter.

Six required gauges/histograms:
  dlmserve:requests_in_flight
  dlmserve:denoising_step_duration_seconds  (histogram)
  dlmserve:step_batch_size                  (histogram)
  dlmserve:committed_tokens_per_request     (histogram)
  dlmserve:throughput_tokens_per_second
  dlmserve:queue_depth
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

from prometheus_client import Gauge, Histogram, generate_latest

requests_in_flight = Gauge(
    "dlmserve_requests_in_flight",
    "Number of requests currently being denoised",
)

denoising_step_duration_seconds = Histogram(
    "dlmserve_denoising_step_duration_seconds",
    "Wall-clock time per denoising step across the full batch",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

step_batch_size = Histogram(
    "dlmserve_step_batch_size",
    "Number of requests processed in a single denoising step",
    buckets=(1, 2, 4, 8, 16, 32),
)

committed_tokens_per_request = Histogram(
    "dlmserve_committed_tokens_per_request",
    "Total tokens committed by the time a request reaches DONE",
    buckets=(32, 64, 128, 256, 512, 1024, 2048),
)

throughput_tokens_per_second = Gauge(
    "dlmserve_throughput_tokens_per_second",
    "Rolling tokens/s committed across all active batches (updated per step)",
)

queue_depth = Gauge(
    "dlmserve_queue_depth",
    "Number of requests waiting in the scheduler queue",
)

_tokens_committed_window: list[tuple[float, int]] = []  # (timestamp, count)
_WINDOW_SECONDS = 10.0


def record_step(
    batch_size: int,
    step_duration_s: float,
    tokens_committed: int,
) -> None:
    """Call once per denoising step with the batch's aggregate stats."""
    denoising_step_duration_seconds.observe(step_duration_s)
    step_batch_size.observe(batch_size)

    now = time.monotonic()
    _tokens_committed_window.append((now, tokens_committed))
    # Prune old entries outside the rolling window
    cutoff = now - _WINDOW_SECONDS
    while _tokens_committed_window and _tokens_committed_window[0][0] < cutoff:
        _tokens_committed_window.pop(0)

    total = sum(c for _, c in _tokens_committed_window)
    elapsed = now - _tokens_committed_window[0][0] if _tokens_committed_window else 1.0
    throughput_tokens_per_second.set(total / max(elapsed, 1e-6))


def record_request_done(gen_length: int) -> None:
    committed_tokens_per_request.observe(gen_length)
    requests_in_flight.dec()


def record_request_admitted() -> None:
    requests_in_flight.inc()


def metrics_response() -> bytes:
    """Return the full Prometheus text exposition."""
    return generate_latest()
