from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from drnb.experiment.common import (
    MANIFEST_JSON,
    RESULT_FORMAT_VERSION,
    RESULT_JSON,
    context_from_dict,
    context_to_dict,
    eval_from_dict,
    eval_to_dict,
    json_safe,
    safe_component,
)
from drnb.io import get_path


def experiment_dir(
    name: str, drnb_home: Path | str | None, *, create: bool = True
) -> Path:
    base = get_path(drnb_home, sub_dir="experiments", create_sub_dir=create)
    exp_dir = base / name
    if create:
        exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def manifest_path(
    name: str, drnb_home: Path | str | None, *, create: bool = True
) -> Path:
    return experiment_dir(name, drnb_home, create=create) / MANIFEST_JSON


def experiment_storage_state(
    name: str, drnb_home: Path | str | None
) -> tuple[Path | None, Path | None]:
    """Return existing experiment dir and manifest if present without creating them."""
    try:
        base = get_path(drnb_home, sub_dir="experiments", create_sub_dir=False)
    except (FileNotFoundError, ValueError):
        return None, None
    exp_dir = base / name
    if not exp_dir.exists():
        return None, None
    manifest = exp_dir / MANIFEST_JSON
    return exp_dir, manifest if manifest.exists() else None


def shard_dir(
    exp_name: str,
    drnb_home: Path | str | None,
    method_name: str,
    dataset: str,
    run_record: dict | None,
    *,
    create: bool = True,
) -> Path:
    exp_dir = experiment_dir(exp_name, drnb_home, create=create)
    shard_rel = run_record.get("shard") if run_record else None
    if shard_rel:
        return exp_dir / shard_rel
    return exp_dir / "results" / safe_component(method_name) / safe_component(dataset)


class LazyResult:
    """Wrapper that loads a shard on first access."""

    def __init__(self, shard_dir: Path, loader: Callable[[Path], dict]):
        self.shard_dir = shard_dir
        self._loader = loader
        self._cache: dict | None = None

    def _load(self) -> dict:
        if self._cache is None:
            self._cache = self._loader(self.shard_dir)
        return self._cache

    def __getitem__(self, key: str):
        return self._load()[key]

    def __iter__(self) -> Iterator:
        return iter(self._load())

    def __contains__(self, key: str) -> bool:
        return key in self._load()

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def materialize(self) -> dict:
        return self._load()


def write_result_shard(
    exp_name: str,
    drnb_home: Path | str | None,
    method_name: str,
    dataset: str,
    embed_result: dict,
    run_record: dict | None,
) -> Path:
    shard_dir_path = shard_dir(
        exp_name, drnb_home, method_name, dataset, run_record, create=True
    )
    shard_dir_path.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    for key, value in embed_result.items():
        if key == "context":
            entries[key] = {"type": "context", "value": context_to_dict(value)}
            continue
        if isinstance(value, np.ndarray):
            filename = f"{key}.npz"
            np.savez_compressed(shard_dir_path / filename, data=value)
            entries[key] = {"type": "npz", "file": filename}
            continue
        if is_dataclass(value):
            value = asdict(value)
        if hasattr(value, "eval_type") and hasattr(value, "label"):
            entries[key] = {"type": "eval_result", "value": eval_to_dict(value)}
            continue
        if (
            isinstance(value, list)
            and value
            and all(hasattr(v, "eval_type") for v in value)
        ):
            entries[key] = {
                "type": "eval_results",
                "value": [eval_to_dict(v) for v in value],
            }
            continue
        entries[key] = {"type": "json", "value": json_safe(value)}

    meta = {"version": RESULT_FORMAT_VERSION, "entries": entries}
    with open(shard_dir_path / RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    exp_dir = experiment_dir(exp_name, drnb_home, create=True)
    return shard_dir_path.relative_to(exp_dir)


def load_result_shard(shard_dir_path: Path) -> dict:
    meta_path = shard_dir_path / RESULT_JSON
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    version = meta.get("version")
    if version not in (None, RESULT_FORMAT_VERSION):
        raise ValueError(f"Unsupported result format version {version} in {meta_path}")
    entries = meta.get("entries", {})
    result = {}
    for key, entry in entries.items():
        entry_type = entry.get("type")
        if entry_type == "npz":
            data = np.load(shard_dir_path / entry["file"])
            arr = data[data.files[0]]
            result[key] = arr
        elif entry_type == "context":
            result[key] = context_from_dict(entry.get("value"))
        elif entry_type == "eval_result":
            result[key] = eval_from_dict(entry["value"])
        elif entry_type == "eval_results":
            result[key] = [eval_from_dict(v) for v in entry.get("value", [])]
        else:
            result[key] = entry.get("value")
    return result
