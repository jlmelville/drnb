"""Plugin-facing protocol API for drnb.

This module intentionally exposes only the minimal dataclasses and loaders a plugin
needs to speak the IPC protocol. Python-specific conveniences (neighbor IO, CLI runner,
result writers, etc.) live under ``drnb_plugin_sdk.helpers`` so they can evolve
independently or be skipped entirely by non-Python implementations.
"""

from .protocol import (
    PROTOCOL_VERSION,
    JSONValue,
    PluginContext,
    PluginInputPaths,
    PluginNeighbors,
    PluginOptions,
    PluginOutputPaths,
    PluginRequest,
    PluginSourcePaths,
    context_from_payload,
    context_to_payload,
    env_flag,
    request_to_dict,
    sanitize_params,
)

__all__ = [
    "PROTOCOL_VERSION",
    "JSONValue",
    "PluginContext",
    "PluginInputPaths",
    "PluginNeighbors",
    "PluginOptions",
    "PluginSourcePaths",
    "PluginOutputPaths",
    "PluginRequest",
    "context_from_payload",
    "context_to_payload",
    "env_flag",
    "request_to_dict",
    "sanitize_params",
]
