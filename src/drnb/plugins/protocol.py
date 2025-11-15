from __future__ import annotations

from typing import Any

from drnb_plugin_sdk import protocol as sdk_protocol

from drnb.embed.context import EmbedContext


def context_from_payload(data: dict[str, Any] | None) -> EmbedContext | None:
    """Convert the SDK context payload back into EmbedContext."""
    plugin_ctx = sdk_protocol.context_from_payload(data)
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
