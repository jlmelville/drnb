from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROTOCOL_VERSION = 1


@dataclass
class PluginContext:
    dataset_name: str
    embed_method_name: str
    embed_method_variant: Optional[str] = None
    drnb_home: Optional[Path] = None
    data_sub_dir: Optional[str] = None
    nn_sub_dir: Optional[str] = None
    triplet_sub_dir: Optional[str] = None
    experiment_name: Optional[str] = None


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


def context_from_payload(data: Dict[str, Any] | None) -> Optional[PluginContext]:
    if not data:
        return None
    kwargs: Dict[str, Any] = {}
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
