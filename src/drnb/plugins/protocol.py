from __future__ import annotations

from typing import Any

from drnb.embed.context import EmbedContext
from drnb_plugin_sdk import protocol as sdk_protocol

PROTOCOL_VERSION = sdk_protocol.PROTOCOL_VERSION
PluginNeighbors = sdk_protocol.PluginNeighbors
PluginInputPaths = sdk_protocol.PluginInputPaths
PluginOptions = sdk_protocol.PluginOptions
PluginOutputPaths = sdk_protocol.PluginOutputPaths
PluginRequest = sdk_protocol.PluginRequest
env_flag = sdk_protocol.env_flag
request_to_dict = sdk_protocol.request_to_dict
sanitize_params = sdk_protocol.sanitize_params
load_request = sdk_protocol.load_request


def context_to_payload(ctx: EmbedContext | None) -> dict[str, Any] | None:
    return sdk_protocol.context_to_payload(ctx)


def context_from_payload(data: dict[str, Any] | None) -> EmbedContext | None:
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
