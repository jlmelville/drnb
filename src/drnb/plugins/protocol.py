from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from drnb_plugin_sdk import PluginRequest
from drnb_plugin_sdk.protocol import (
    context_from_payload as sdk_context_from_payload,
    request_to_dict as sdk_request_to_dict,
)

from drnb.embed.context import EmbedContext

JSONValue = bool | int | float | str | None | list["JSONValue"] | dict[str, "JSONValue"]

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def context_to_payload(ctx: EmbedContext | None) -> dict[str, JSONValue] | None:
    """Serialize EmbedContext into the SDK-friendly payload."""
    if ctx is None:
        return None
    payload: dict[str, JSONValue] = {}
    for field in (
        "dataset_name",
        "embed_method_name",
        "embed_method_variant",
        "drnb_home",
        "data_sub_dir",
        "nn_sub_dir",
        "triplet_sub_dir",
        "experiment_name",
    ):
        value = getattr(ctx, field, None)
        if isinstance(value, Path):
            payload[field] = str(value)
        else:
            payload[field] = value
    return payload


def context_from_payload(data: dict[str, Any] | None) -> EmbedContext | None:
    """Convert the SDK context payload back into EmbedContext."""
    plugin_ctx = sdk_context_from_payload(data)
    if plugin_ctx is None:
        return None
    return EmbedContext(
        dataset_name=plugin_ctx.dataset_name,
        embed_method_name=plugin_ctx.embed_method_name,
        embed_method_variant=plugin_ctx.embed_method_variant or "",
        drnb_home=plugin_ctx.drnb_home,
        data_sub_dir=plugin_ctx.data_sub_dir or "data",
        nn_sub_dir=plugin_ctx.nn_sub_dir or "nn",
        triplet_sub_dir=plugin_ctx.triplet_sub_dir or "triplets",
        experiment_name=plugin_ctx.experiment_name,
    )


def env_flag(var_name: str, default: bool = False) -> bool:
    """Interpret env vars consistently with plugin expectations."""
    from os import environ

    raw = environ.get(var_name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm in _TRUTHY:
        return True
    if norm in _FALSY:
        return False
    return default


def sanitize_params(params: dict[str, Any] | None) -> dict[str, JSONValue]:
    """Serialize params into JSON-safe values before constructing the request."""
    if params is None:
        return {}
    return {
        str(key): _sanitize_value(value, path=str(key)) for key, value in params.items()
    }


def _sanitize_value(value: Any, path: str) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(val, path=f"{path}.{key}")
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _sanitize_value(v, path=f"{path}[{idx}]") for idx, v in enumerate(value)
        ]
    raise TypeError(f"Unsupported parameter type at {path}: {type(value).__name__}")


def request_to_dict(req: PluginRequest) -> dict[str, Any]:
    """Convert a PluginRequest dataclass into the JSON dict written to disk."""
    return sdk_request_to_dict(req)
