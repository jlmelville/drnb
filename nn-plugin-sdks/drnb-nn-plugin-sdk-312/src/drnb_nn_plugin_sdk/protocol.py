from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

JSONScalar = bool | int | float | str | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

NN_PLUGIN_PROTOCOL_VERSION = 1
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


class DrnbNNPluginProtocolError(RuntimeError):
    """Raised for protocol-level issues: version mismatches, malformed requests, or missing required fields."""


@dataclass
class NNPluginContext:
    dataset_name: str
    drnb_home: Path | None = None
    data_sub_dir: str | None = None
    nn_sub_dir: str | None = None
    experiment_name: str | None = None


@dataclass
class NNPluginInputPaths:
    x_path: str


@dataclass
class NNPluginOptions:
    keep_temps: bool = False
    log_path: str | None = None
    use_sandbox_copies: bool | None = False


@dataclass
class NNPluginOutputPaths:
    result_path: str
    response_path: str | None = None


@dataclass
class NNPluginRequest:
    protocol_version: int
    method: str
    metric: str
    n_neighbors: int
    params: dict[str, JSONValue] = field(default_factory=dict)
    return_distance: bool = True
    context: dict[str, JSONValue] | None = None
    input: NNPluginInputPaths | None = None
    options: NNPluginOptions = field(default_factory=NNPluginOptions)
    output: NNPluginOutputPaths | None = None


_CONTEXT_FIELDS = (
    "dataset_name",
    "drnb_home",
    "data_sub_dir",
    "nn_sub_dir",
    "experiment_name",
)


def context_from_payload(data: dict[str, Any] | None) -> NNPluginContext | None:
    """Deserialize a raw payload into a lightweight NNPluginContext."""
    if not data:
        return None
    kwargs: dict[str, Any] = {}
    for field in _CONTEXT_FIELDS:
        value = data.get(field)
        if field == "drnb_home" and value:
            kwargs[field] = Path(value)
        else:
            kwargs[field] = value
    if not kwargs.get("dataset_name"):
        raise ValueError("Serialized context missing dataset_name")
    return NNPluginContext(**kwargs)


def load_request(path: str | Path) -> NNPluginRequest:
    """Load and validate an NNPluginRequest from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    proto = raw.get("protocol") or raw.get("protocol_version")
    if proto != NN_PLUGIN_PROTOCOL_VERSION:
        raise DrnbNNPluginProtocolError(
            f"protocol mismatch: expected {NN_PLUGIN_PROTOCOL_VERSION}, got {proto}"
        )
    raw["protocol_version"] = proto
    raw.pop("protocol", None)
    return _decode_request(raw)


def _decode_request(raw: dict[str, Any]) -> NNPluginRequest:
    input_payload = raw.get("input")
    output_payload = raw.get("output")
    if not input_payload:
        raise DrnbNNPluginProtocolError("Request missing input block")
    if not output_payload:
        raise DrnbNNPluginProtocolError("Request missing output block")
    if "x_path" not in input_payload:
        raise DrnbNNPluginProtocolError("Request missing input.x_path")
    return NNPluginRequest(
        protocol_version=raw["protocol_version"],
        method=raw["method"],
        metric=raw["metric"],
        n_neighbors=int(raw["n_neighbors"]),
        params=raw.get("params") or {},
        return_distance=bool(raw.get("return_distance", True)),
        context=raw.get("context"),
        input=NNPluginInputPaths(**input_payload),
        options=NNPluginOptions(**(raw.get("options") or {})),
        output=NNPluginOutputPaths(**output_payload),
    )


def context_to_payload(ctx: NNPluginContext | None) -> dict[str, JSONValue] | None:
    """Serialize NNPluginContext into JSON-friendly payload."""
    if ctx is None:
        return None
    payload: dict[str, JSONValue] = {}
    for field in _CONTEXT_FIELDS:
        value = getattr(ctx, field, None)
        if isinstance(value, Path):
            payload[field] = str(value)
        else:
            payload[field] = value
    return payload


def env_flag(var_name: str, default: bool = False) -> bool:
    """Interpret env vars consistently between host and plugins."""
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
    return _convert_json_value(params, path="params")


def request_to_dict(req: NNPluginRequest) -> dict[str, Any]:
    """Convert an NNPluginRequest dataclass into the JSON dict written to disk."""
    payload = asdict(req)
    payload["params"] = _convert_json_value(req.params, path="params")
    if payload.get("context") is not None:
        payload["context"] = _convert_json_value(req.context, path="context")
    payload["protocol"] = payload.pop("protocol_version")
    return payload


def _convert_json_value(value: Any, path: str | None = None) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return _convert_json_value(value.item(), path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result: dict[str, JSONValue] = {}
        for key, val in value.items():
            key_str = str(key)
            child_path = f"{path}.{key_str}" if path else key_str
            result[key_str] = _convert_json_value(val, path=child_path)
        return result
    if isinstance(value, (list, tuple, set)):
        return [
            _convert_json_value(v, path=f"{path}[{idx}]" if path else f"[{idx}]")
            for idx, v in enumerate(value)
        ]
    location = path or "<root>"
    raise TypeError(f"Unsupported parameter type at {location}: {type(value).__name__}")
