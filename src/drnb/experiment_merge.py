from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from drnb.experiment_common import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_MISSING,
    RUN_STATUS_PARTIAL_EVALS,
    expected_eval_labels,
    merge_evaluations,
    method_signature,
    now_iso,
    param_signature,
)
from drnb.experiment_persistence import (
    LazyResult,
    experiment_dir,
    experiment_storage_state,
)
from drnb.experiment_runner import result_progress

if TYPE_CHECKING:
    from drnb.experiment import Experiment


def merge_experiments(
    exp1: "Experiment",
    exp2: "Experiment",
    name: str | None = None,
    *,
    overwrite: bool = False,
) -> "Experiment":
    """Merge two experiments, allowing holes when datasets are missing in one side."""
    merged_name = name or f"merged-{exp1.name}-{exp2.name}"
    merged_home = exp1.drnb_home or exp2.drnb_home
    dest_existing, _ = experiment_storage_state(merged_name, merged_home)
    if dest_existing is not None and not overwrite:
        raise ValueError(
            f"Destination experiment directory already exists: {dest_existing}. "
            "Choose a new name or pass overwrite=True."
        )
    from drnb.experiment import Experiment

    merged = Experiment(name=merged_name, warn_on_existing=False)
    merged.drnb_home = merged_home

    import drnb.experiment as exp_mod

    merged_datasets: list[str] = []
    seen = set()
    for dataset in exp1.datasets + exp2.datasets:
        if dataset not in seen:
            merged_datasets.append(dataset)
            seen.add(dataset)
    for dataset in merged_datasets:
        merged.add_dataset(dataset)

    merged.evaluations = merge_evaluations(exp1.evaluations, exp2.evaluations)
    merged_expected_labels = expected_eval_labels(merged.evaluations)

    dest_exp_dir = experiment_dir(merged_name, merged.drnb_home, create=True)

    def copy_from(exp_src: Experiment):
        src_exp_dir = experiment_dir(exp_src.name, exp_src.drnb_home, create=False)
        for method, method_name in exp_src.methods:
            if not merged._has_method_name(method_name):
                merged.add_method(method, name=method_name)
            else:
                merged_method = next(m for m, n in merged.methods if n == method_name)
                if method_signature(method) != method_signature(merged_method):
                    raise ValueError(
                        f"Method '{method_name}' has conflicting configurations "
                        "across experiments; rename the method or align configs before merging."
                    )
            if method_name not in merged.results:
                merged.results[method_name] = {}
            merged_method = next(m for m, n in merged.methods if n == method_name)
            signature = param_signature(merged_method, merged.evaluations)
            for dataset, res in exp_src.results.get(method_name, {}).items():
                run_entry = exp_src.run_info.get(method_name, {}).get(dataset)
                shard_rel = None
                dest_shard_dir = None
                if run_entry and run_entry.get("shard"):
                    shard_rel = Path(run_entry["shard"])
                    src_shard_dir = src_exp_dir / shard_rel
                    dest_shard_dir = dest_exp_dir / shard_rel
                    if src_shard_dir.exists():
                        if dest_shard_dir.exists():
                            shutil.rmtree(dest_shard_dir)
                        dest_shard_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(src_shard_dir, dest_shard_dir)
                if shard_rel and dest_shard_dir and dest_shard_dir.exists():
                    merged.results[method_name][dataset] = LazyResult(
                        dest_shard_dir, merged._load_result_shard
                    )
                    materialized = merged.results[method_name][dataset].materialize()
                    status, completed, expected_count, _ = result_progress(
                        materialized, merged_expected_labels
                    )
                else:
                    merged.results[method_name][dataset] = res
                    materialized = res
                    status, completed, expected_count, _ = result_progress(
                        materialized, merged_expected_labels
                    )
                merged.run_info.setdefault(method_name, {})[dataset] = {
                    "status": status,
                    "signature": signature,
                    "updated_at": now_iso(),
                    "shard": str(shard_rel) if shard_rel else "",
                    "evals_completed": completed,
                    "evals_expected": expected_count,
                }

    copy_from(exp1)
    copy_from(exp2)

    merged.name = merged_name
    return merged
