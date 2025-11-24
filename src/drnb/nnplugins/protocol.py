from __future__ import annotations

from typing import Any

from drnb_nn_plugin_sdk import (
    JSONValue,
    NNPluginContext,
    context_to_payload as _ctx_to_payload,
)


def context_to_payload(ctx: NNPluginContext | None) -> dict[str, JSONValue] | None:
    """Serialize NNPluginContext into the SDK-friendly payload."""
    return _ctx_to_payload(ctx)


def context_from_payload(data: dict[str, Any] | None) -> NNPluginContext | None:
    """Convert the SDK context payload back into NNPluginContext."""
    from drnb_nn_plugin_sdk import context_from_payload as _ctx_from_payload

    return _ctx_from_payload(data)
