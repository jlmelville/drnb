"""Nearest-neighbor plugin integration."""

from .registry import NNPluginSpec, get_registry

__all__ = ["get_registry", "NNPluginSpec"]
