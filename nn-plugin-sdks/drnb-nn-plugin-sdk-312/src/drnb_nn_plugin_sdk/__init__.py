"""Protocol API for drnb nearest-neighbor plugins (Python 3.12).

This package exposes only the IPC dataclasses and JSON helpers required by NN
plugins. Convenience utilities live under ``drnb_nn_plugin_sdk.helpers`` and
depend solely on NumPy and the standard library.
"""

from .protocol import (
    JSONValue,
    NN_PLUGIN_PROTOCOL_VERSION,
    DrnbNNPluginProtocolError,
    NNPluginContext,
    NNPluginInputPaths,
    NNPluginOptions,
    NNPluginOutputPaths,
    NNPluginRequest,
    context_from_payload,
    context_to_payload,
    env_flag,
    load_request,
    request_to_dict,
    sanitize_params,
)

__all__ = [
    "NN_PLUGIN_PROTOCOL_VERSION",
    "DrnbNNPluginProtocolError",
    "JSONValue",
    "NNPluginContext",
    "NNPluginInputPaths",
    "NNPluginOptions",
    "NNPluginOutputPaths",
    "NNPluginRequest",
    "context_from_payload",
    "context_to_payload",
    "env_flag",
    "load_request",
    "request_to_dict",
    "sanitize_params",
]
