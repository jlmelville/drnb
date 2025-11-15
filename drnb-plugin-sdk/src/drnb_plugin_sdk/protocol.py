from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    response_path: str | None = None


@dataclass
class PluginRequest:
    protocol_version: int
    method: str
    params: dict[str, JSONValue]
    context: dict[str, JSONValue] | None
    input: PluginInputPaths
    options: PluginOptions
    output: PluginOutputPaths


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
        output=PluginOutputPaths(
            result_path=output_payload["result_path"],
            response_path=output_payload.get("response_path"),
        ),
    )
    return request
