from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from drnb.eval.factory import create_evaluators

EXPERIMENT_FORMAT_VERSION = 2
RESULT_FORMAT_VERSION = 1
RESULT_JSON = "result.json"
MANIFEST_JSON = "manifest.json"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_MISSING = "missing"
RUN_STATUS_PARTIAL_EVALS = "evals_partial"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_col(colname: str, sep: str = "-") -> str:
    """Return everything up to (but not including) the second `sep` in the string."""
    index = colname.find(sep)
    if index == -1:
        return colname
    index2 = colname.find(sep, index + 1)
    if index2 == -1:
        return colname
    return colname[:index2]


def safe_component(name: str) -> str:
    return name.replace(os.sep, "_")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def eval_to_dict(ev) -> dict:
    return {
        "eval_type": ev.eval_type,
        "label": ev.label,
        "value": ev.value,
        "info": json_safe(ev.info),
    }


def eval_from_dict(data: dict):
    from drnb.eval.base import EvalResult

    return EvalResult(
        eval_type=data.get("eval_type", ""),
        label=data.get("label", ""),
        value=data.get("value", 0.0),
        info=data.get("info", {}),
    )


def context_to_dict(ctx: Any) -> dict | None:
    if ctx is None:
        return None
    return {
        "dataset_name": getattr(ctx, "dataset_name", None),
        "embed_method_name": getattr(ctx, "embed_method_name", None),
        "embed_method_variant": getattr(ctx, "embed_method_variant", ""),
        "drnb_home": str(getattr(ctx, "drnb_home", ""))
        if getattr(ctx, "drnb_home", None) is not None
        else None,
        "data_sub_dir": getattr(ctx, "data_sub_dir", "data"),
        "nn_sub_dir": getattr(ctx, "nn_sub_dir", "nn"),
        "triplet_sub_dir": getattr(ctx, "triplet_sub_dir", "triplets"),
        "experiment_name": getattr(ctx, "experiment_name", None),
    }


def context_from_dict(data: dict | None):
    if not data:
        return None
    from drnb.embed.context import EmbedContext

    home = data.get("drnb_home")
    if home is not None:
        data = {**data, "drnb_home": Path(home)}
    return EmbedContext(**data)


def method_to_manifest(method: Any) -> Any:
    if isinstance(method, list):
        return {"kind": "list", "value": [method_to_manifest(m) for m in method]}
    if isinstance(method, tuple):
        return {
            "kind": "tuple",
            "value": [json_safe(method[0]), json_safe(method[1])],
        }
    return {"kind": "json", "value": json_safe(method)}


def method_from_manifest(entry: Any) -> Any:
    kind = entry.get("kind")
    value = entry.get("value")
    if kind == "list":
        return [method_from_manifest(v) for v in value]
    if kind == "tuple":
        return (value[0], value[1])
    return value


def param_signature(method: Any, evaluations: list[Any]) -> str:
    payload = {
        "method": method_to_manifest(method),
        "evaluations": json_safe(evaluations),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def method_signature(method: Any) -> str:
    """Stable signature for an embedding method independent of evaluations."""
    payload = method_to_manifest(method)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def labels_for_evaluator(evaluator: Any) -> list[str]:
    to_str = getattr(evaluator, "to_str", None)
    neighbors = getattr(evaluator, "n_neighbors", None)
    if callable(to_str) and neighbors is not None:
        if not isinstance(neighbors, (list, tuple)):
            neighbors = [neighbors]
        try:
            return [short_col(to_str(n)) for n in neighbors]
        except TypeError:
            pass
    return [short_col(str(evaluator))]


def expected_eval_labels(evaluations: list[Any]) -> list[str]:
    evaluators = create_evaluators(evaluations)
    return expected_eval_labels_from_evaluators(evaluators)


def expected_eval_labels_from_evaluators(evaluators: list[Any]) -> list[str]:
    labels: list[str] = []
    for evaluator in evaluators:
        labels.extend(labels_for_evaluator(evaluator))
    return labels


def merge_eval_results_by_label(
    existing: list, new: list, expected_labels: list[str]
) -> list:
    label_map: dict[str, Any] = {}
    for ev in existing or []:
        label_map[short_col(ev.label)] = ev
    for ev in new or []:
        label_map[short_col(ev.label)] = ev
    merged: list[Any] = []
    expected_set = [short_col(lbl) for lbl in expected_labels]
    for lbl in expected_set:
        if lbl in label_map:
            merged.append(label_map.pop(lbl))
    merged.extend(label_map.values())
    return merged


def merge_evaluations(eval1: list[Any], eval2: list[Any]) -> list[Any]:
    """Combine evaluation lists, preserving order and handling unhashable entries."""
    merged: list[Any] = []
    for ev in list(eval1) + list(eval2):
        if not any(existing == ev for existing in merged):
            merged.append(ev)
    return merged


def normalize_evaluations(evaluations: Any | list[Any]) -> list[Any]:
    """Normalize evaluation input to a list without breaking tuple-based eval specs."""
    if evaluations is None:
        return []
    if isinstance(evaluations, tuple) and len(evaluations) == 2 and isinstance(
        evaluations[0], str
    ):
        return [evaluations]
    if isinstance(evaluations, (list, tuple)):
        return list(evaluations)
    return [evaluations]


def eval_label_set(res: dict | None) -> set[str]:
    if not res or "evaluations" not in res:
        return set()
    return {short_col(ev.label) for ev in res["evaluations"]}
