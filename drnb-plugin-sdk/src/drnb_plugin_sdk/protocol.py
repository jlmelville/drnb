from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

JSONScalar = bool | int | float | str | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

PROTOCOL_VERSION = 1
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


@dataclass
class PluginContext:
    dataset_name: str
    embed_method_name: str
    embed_method_variant: str | None = None
    drnb_home: Path | None = None
    data_sub_dir: str | None = None
    nn_sub_dir: str | None = None
    triplet_sub_dir: str | None = None
    experiment_name: str | None = None


@dataclass
class PluginNeighbors:
    idx_path: str | None = None
    dist_path: str | None = None


@dataclass
class PluginInputPaths:
    x_path: str
    init_path: str | None = None
    neighbors: PluginNeighbors = field(default_factory=PluginNeighbors)


@dataclass
class PluginOptions:
    keep_temps: bool = False
    log_path: str | None = None
    use_precomputed_knn: bool | None = None


@dataclass
class PluginOutputPaths:
    result_path: str


@dataclass
class PluginRequest:
    protocol_version: int
    method: str
    params: dict[str, JSONValue]
    context: dict[str, JSONValue] | None
    input: PluginInputPaths
    options: PluginOptions
    output: PluginOutputPaths


def env_flag(var_name: str, default: bool = False) -> bool:
    """Interpret an environment flag the same way host and plugin expect."""
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
    """Convert params into a JSON-safe structure, rejecting unsupported types."""
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


_CONTEXT_FIELDS = (
    "dataset_name",
    "embed_method_name",
    "embed_method_variant",
    "drnb_home",
    "data_sub_dir",
    "nn_sub_dir",
    "triplet_sub_dir",
    "experiment_name",
)


def context_to_payload(ctx: Any | None) -> dict[str, JSONValue] | None:
    """Serialize an arbitrary context object (duck-typed) into JSON."""
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


def context_from_payload(data: dict[str, Any] | None) -> PluginContext | None:
    """Deserialize a raw payload into a lightweight PluginContext."""
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
    if not kwargs.get("embed_method_name"):
        raise ValueError("Serialized context missing embed_method_name")
    return PluginContext(**kwargs)


def request_to_dict(req: PluginRequest) -> dict[str, Any]:
    payload = asdict(req)
    payload["protocol"] = payload["protocol_version"]
    return payload


def load_request(path: str | Path) -> PluginRequest:
    """Load and validate a PluginRequest from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    proto = raw.get("protocol") or raw.get("protocol_version")
    if proto != PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {PROTOCOL_VERSION}, got {proto}"
        )
    raw["protocol_version"] = proto
    raw.pop("protocol", None)
    return _decode_request(raw)


def _decode_request(raw: dict[str, Any]) -> PluginRequest:
    input_payload = raw.get("input") or {}
    options_payload = raw.get("options") or {}
    output_payload = raw.get("output") or {}
    request = PluginRequest(
        protocol_version=raw["protocol_version"],
        method=raw["method"],
        params=raw.get("params") or {},
        context=raw.get("context"),
        input=PluginInputPaths(
            x_path=input_payload["x_path"],
            init_path=input_payload.get("init_path"),
            neighbors=PluginNeighbors(**(input_payload.get("neighbors") or {})),
        ),
        options=PluginOptions(**options_payload),
        output=PluginOutputPaths(**output_payload),
    )
    return request
