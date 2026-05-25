"""Model loaders. bd3lm.py is v0.2 scope, do not create in v0.1."""

from __future__ import annotations

from dlmserve.models.llada import LoadedLLaDA

LoadedModel = LoadedLLaDA

__all__ = ["LoadedLLaDA", "LoadedModel"]
