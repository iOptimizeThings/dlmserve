"""Structured logging setup for dlmserve.

Structured JSON logs, one event per line.
Schema fields: ts, level, logger, event, [req_id, step, ...]

Usage:
    DLMSERVE_LOG_LEVEL=debug dlmserve          # verbose
    DLMSERVE_LOG_LEVEL=warning dlmserve        # quiet

Levels: debug | info | warning | error | critical  (default: info)
"""

from __future__ import annotations

import json
import logging
import os
import time


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        # Any extra fields passed as kwargs to log calls come through as record attributes
        for key in ("req_id", "step", "batch_size", "elapsed_ms", "model_id", "tokens"):
            val = getattr(record, key, None)
            if val is not None:
                base[key] = val
        return json.dumps(base)


def setup_logging(level: str | None = None) -> None:
    """Configure root logger and silence noisy third-party loggers."""
    raw = level or os.environ.get("DLMSERVE_LOG_LEVEL", "info")
    numeric = getattr(logging, raw.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(numeric)
    # Replace any existing handlers (uvicorn installs its own)
    root.handlers = [handler]

    # Quiet down noisy libraries at INFO — still visible at DEBUG
    for noisy in ("transformers", "bitsandbytes", "accelerate", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))
