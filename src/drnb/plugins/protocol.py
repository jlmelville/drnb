from __future__ import annotations

from pathlib import Path
from typing import Any

import drnb_plugin_sdk as sdk
from drnb_plugin_sdk import (
    JSONValue,
)

from drnb.embed.context import EmbedContext


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
    plugin_ctx = sdk.context_from_payload(data)
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
