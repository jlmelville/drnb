from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drnb.embed import check_embed_method, get_embedder_name
from drnb.experiment.common import (
    EXPERIMENT_FORMAT_VERSION,
    MANIFEST_JSON,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_PARTIAL_EVALS,
    expected_eval_labels,
    json_safe,
    merge_evaluations,
    method_from_manifest,
    method_to_manifest,
    normalize_evaluations,
    now_iso,
    param_signature,
    short_col,
)
from drnb.experiment.merge import merge_experiments
from drnb.experiment.persistence import (
    LazyResult,
    experiment_dir,
    experiment_storage_state,
    load_result_shard,
    manifest_path,
    shard_dir,
    write_result_shard,
)
from drnb.experiment.report import (
    plot as plot_report,
    status as status_report,
    to_df as to_df_report,
)
from drnb.experiment.runner import result_progress, run_missing_evaluations
from drnb.log import log
from drnb.util import dts_to_str

__all__ = [
    "Experiment",
    "read_experiment",
    "merge_experiments",
]

UNKNOWN_VERSION_INFO = {"package": "unknown", "version": "unknown"}


def ensure_experiment_name(experiment_name: str | None) -> str:
    """Ensure that the experiment name is not empty and return a default name if it is."""
    if experiment_name is None or not experiment_name:
        experiment_name = f"experiment-{dts_to_str()}"
        log.info("Using experiment name: %s", experiment_name)
    return experiment_name


@dataclass
class Experiment:
    """Run and checkpoint embedding experiments."""

    name: str = ""
    datasets: list[str] = field(default_factory=list)
    methods: list = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    evaluations: list = field(default_factory=list)
    verbose: bool = False
    drnb_home: Path | str | None = None
    warn_on_existing: bool = True
    run_info: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.warn_on_existing or not self.name:
            return
        exp_dir, manifest = experiment_storage_state(self.name, self.drnb_home)
        if exp_dir is None or self.run_info:
            return
        log.warning(
            "Experiment directory already exists: %s. Existing shards may be reused; "
            "use a new name or clear_storage() if you want a fresh run.",
            exp_dir,
        )

    def add_method(self, method, *, params=None, name: str = ""):
        """Add an embedding method to the experiment."""
        method = check_embed_method(method, params)
        if not name:
            name = get_embedder_name(method)
        if self._has_method_name(name):
            raise ValueError(f"Experiment already has a embedding method '{name}'")
        self.methods.append((method, name))

    def add_datasets(self, datasets: list[str]):
        """Add a list of datasets to the experiment."""
        for dataset in datasets:
            self.add_dataset(dataset)

    def add_dataset(self, dataset: str):
        """Add a dataset to the experiment."""
        if dataset in self.datasets:
            raise ValueError(f"Experiment already has a dataset '{dataset}'")
        self.datasets.append(dataset)

    def add_evaluations(self, evaluations: Any | list[Any]):
        """Add one or more evaluation metrics (deduped by equality)."""
        normalized = normalize_evaluations(evaluations)
        self.evaluations = merge_evaluations(self.evaluations, normalized)

    def _has_method_name(self, name: str) -> bool:
        return any(method_name == name for _, method_name in self.methods)

    def _ensure_unique(self):
        if len(self.datasets) != len(set(self.datasets)):
            raise ValueError("Duplicate datasets configured in experiment")
        method_names = [name for _, name in self.methods]
        if len(method_names) != len(set(method_names)):
            raise ValueError("Duplicate method names configured in experiment")

    def _lazy_result(self, method_name: str, dataset: str) -> LazyResult:
        run_record = self.run_info.get(method_name, {}).get(dataset)
        shard_path = shard_dir(
            self.name, self.drnb_home, method_name, dataset, run_record, create=False
        )
        return LazyResult(shard_path, load_result_shard)

    def _write_result_shard(
        self, method_name: str, dataset: str, embed_result: dict
    ) -> Path:
        run_record = self.run_info.get(method_name, {}).get(dataset)
        return write_result_shard(
            self.name,
            self.drnb_home,
            method_name,
            dataset,
            embed_result,
            run_record,
        )

    def _load_result_shard(self, shard_dir_path: Path) -> dict:
        return load_result_shard(shard_dir_path)

    def run(self):
        """Run the experiment, checkpointing each dataset/method pair as soon as it completes."""
        import drnb.embed.pipeline as pl

        self.name = ensure_experiment_name(self.name)
        self._ensure_unique()
        exp_existing, manifest_existing = experiment_storage_state(
            self.name, self.drnb_home
        )
        exp_dir = experiment_dir(self.name, self.drnb_home, create=True)
        if (
            self.warn_on_existing
            and (exp_existing is not None or manifest_existing is not None)
            and not self.run_info
        ):
            log.warning(
                "Experiment directory already exists: %s. Existing shards may be reused; "
                "use a new name or clear_storage() if you want a fresh run.",
                exp_dir,
            )
        log.info("Experiment %s writing to %s", self.name, exp_dir)
        expected_labels = expected_eval_labels(self.evaluations)
        run_info_updates: dict[str, dict[str, dict[str, Any]]] = {}
        for method, method_name in self.methods:
            pipeline = pl.create_pipeline(
                method=method,
                eval_metrics=self.evaluations,
                plot=False,
                verbose=self.verbose,
            )
            method_results = self.results.get(method_name)
            if method_results is None:
                method_results = {}
                self.results[method_name] = method_results
            signature = param_signature(method, self.evaluations)
            for dataset in self.datasets:
                run_record = self.run_info.get(method_name, {}).get(dataset)
                existing = method_results.get(dataset)
                if isinstance(existing, LazyResult):
                    try:
                        existing = existing.materialize()
                    except FileNotFoundError:
                        existing = None
                status, completed, expected_count, missing = result_progress(
                    existing, expected_labels
                )
                prior_sig = run_record.get("signature") if run_record else None

                if status == RUN_STATUS_COMPLETED and prior_sig == signature:
                    log.info(
                        "Skipping %s on %s (already completed)",
                        method_name,
                        dataset,
                    )
                    if dataset not in method_results:
                        method_results[dataset] = self._lazy_result(
                            method_name, dataset
                        )
                    continue

                if prior_sig is not None and prior_sig != signature:
                    log.info(
                        "Rerunning %s on %s due to signature mismatch (prior=%s current=%s)",
                        method_name,
                        dataset,
                        prior_sig,
                        signature,
                    )

                if (
                    status == RUN_STATUS_PARTIAL_EVALS
                    and prior_sig == signature
                    and existing
                    and missing
                ):
                    log.info(
                        "Reusing coords for %s on %s; computing %d missing evaluations",
                        method_name,
                        dataset,
                        len(missing),
                    )
                    embed_result = run_missing_evaluations(
                        pipeline, dataset, existing, missing, expected_labels
                    )
                    method_results[dataset] = embed_result
                else:
                    log.info("Running %s on %s", method_name, dataset)
                    embed_result = pipeline.run(dataset)
                    method_results[dataset] = embed_result

                shard_path = write_result_shard(
                    self.name,
                    self.drnb_home,
                    method_name,
                    dataset,
                    method_results[dataset],
                    run_record,
                )
                status_update, completed_update, expected_update, _ = result_progress(
                    method_results[dataset], expected_labels
                )
                self._update_run_info(
                    method_name,
                    dataset,
                    signature,
                    shard_path,
                    run_info_updates,
                    status=status_update,
                    evals_completed=completed_update,
                    evals_expected=expected_update,
                    version_info=method_results[dataset].get("version_info"),
                )

                self._write_manifest(run_info_updates)
                run_info_updates = {}

        self._write_manifest(run_info_updates)

    def _update_run_info(
        self,
        method_name: str,
        dataset: str,
        signature: str,
        shard_path: Path,
        run_info_updates: dict,
        status: str = RUN_STATUS_COMPLETED,
        evals_completed: int | None = None,
        evals_expected: int | None = None,
        version_info: Any | None = None,
    ):
        if method_name not in run_info_updates:
            run_info_updates[method_name] = {}
        entry = {
            "status": status,
            "signature": signature,
            "updated_at": now_iso(),
            "shard": str(shard_path),
        }
        if evals_completed is not None:
            entry["evals_completed"] = evals_completed
        if evals_expected is not None:
            entry["evals_expected"] = evals_expected
        if version_info is not None:
            entry["version_info"] = json_safe(version_info)
        run_info_updates[method_name][dataset] = entry

    def _write_manifest(self, run_info_updates: dict):
        if run_info_updates:
            for method_name, datasets in run_info_updates.items():
                if method_name not in self.run_info:
                    self.run_info[method_name] = {}
                self.run_info[method_name].update(datasets)
        exp_dir = experiment_dir(self.name, self.drnb_home, create=True)
        manifest = {
            "format_version": EXPERIMENT_FORMAT_VERSION,
            "name": self.name,
            "datasets": self.datasets,
            "methods": [
                {"name": name, "method": method_to_manifest(method)}
                for method, name in self.methods
            ],
            "evaluations": json_safe(self.evaluations),
            "run_info": self.run_info,
        }
        with open(exp_dir / MANIFEST_JSON, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def clear_task(self, method_name: str, dataset: str):
        """Drop a single shard and run record so the pair will rerun on the next `run`."""
        run_record = self.run_info.get(method_name, {}).get(dataset)
        shard_dir_path = shard_dir(
            self.name, self.drnb_home, method_name, dataset, run_record, create=False
        )
        if shard_dir_path.exists():
            shutil.rmtree(shard_dir_path)
        if method_name in self.run_info:
            self.run_info[method_name].pop(dataset, None)
        if method_name in self.results:
            self.results[method_name].pop(dataset, None)
            if not self.results[method_name]:
                self.results.pop(method_name, None)
        self._write_manifest({})

    def clear_method(self, method_name: str):
        """Drop all shards and run records for a method so every dataset reruns."""
        datasets = list(self.run_info.get(method_name, {}).keys())
        for dataset in datasets:
            self.clear_task(method_name, dataset)

    def reset(self):
        """Delete all shards and metadata for this experiment (destructive)."""
        try:
            exp_dir = experiment_dir(self.name, self.drnb_home, create=False)
        except FileNotFoundError:
            exp_dir = None
        if exp_dir and exp_dir.exists():
            shutil.rmtree(exp_dir)
        self.results.clear()
        self.run_info.clear()
        self.datasets = []
        self.methods = []
        self.evaluations = []

    def clear_storage(self):
        """Delete on-disk data for this experiment but keep the in-memory setup."""
        try:
            exp_dir = experiment_dir(self.name, self.drnb_home, create=False)
        except FileNotFoundError:
            exp_dir = None
        if exp_dir and exp_dir.exists():
            log.warning("Deleting experiment storage at %s", exp_dir)
            shutil.rmtree(exp_dir)
        self.results.clear()
        self.run_info.clear()

    def save(self, name: str | None = None):
        """Save the experiment to the repository. If `name` is provided, the experiment will be renamed."""
        if name is not None:
            self.name = name
            log.info("Renaming experiment to %s", self.name)
        self.name = ensure_experiment_name(self.name)
        self._ensure_unique()
        exp_dir = experiment_dir(self.name, self.drnb_home, create=True)
        run_info_updates: dict[str, dict[str, dict[str, Any]]] = {}
        for method, method_name in self.methods:
            method_results = self.results.get(method_name, {})
            signature = param_signature(method, self.evaluations)
            for dataset, result in method_results.items():
                run_record = self.run_info.get(method_name, {}).get(dataset)
                if isinstance(result, LazyResult):
                    if run_record is not None:
                        continue
                    result = result.materialize()
                shard_path = write_result_shard(
                    self.name,
                    self.drnb_home,
                    method_name,
                    dataset,
                    result,
                    run_record,
                )
                status_update, completed_update, expected_update, _ = result_progress(
                    result, expected_eval_labels(self.evaluations)
                )
                self._update_run_info(
                    method_name,
                    dataset,
                    signature,
                    shard_path,
                    run_info_updates,
                    status=status_update,
                    evals_completed=completed_update,
                    evals_expected=expected_update,
                    version_info=(
                        result.get("version_info") if isinstance(result, dict) else None
                    ),
                )
        self._write_manifest(run_info_updates)

    def status(self):
        return status_report(self)

    def to_df(
        self,
        datasets: list[str] | None = None,
        methods: list[str] | str | None = None,
        metrics: list[str] | None = None,
    ):
        return to_df_report(self, datasets=datasets, methods=methods, metrics=metrics)

    def _version_for(self, method_name: str, dataset: str) -> Any:
        run_entry = self.run_info.get(method_name, {}).get(dataset)
        if run_entry and "version_info" in run_entry:
            return run_entry["version_info"]
        res = self.results.get(method_name, {}).get(dataset)
        if isinstance(res, LazyResult):
            try:
                res = res.materialize()
            except FileNotFoundError:
                res = None
        if isinstance(res, dict) and "version_info" in res:
            return res["version_info"]
        return dict(UNKNOWN_VERSION_INFO)

    def versions(self, *, as_df: bool = False):
        """Return embedder version metadata per method/dataset.

        By default returns a nested dict of the form:
        {method_name: {dataset: version_info or [version_info,...]}}.
        If ``as_df`` is True, returns a DataFrame with one row per version entry
        (chained embedders create multiple rows, one per component).
        """

        def _version_row(
            method: str, dataset: str, info: Any, *, component_idx: int | None
        ) -> dict[str, Any]:
            if not isinstance(info, dict):
                info = {
                    "package": UNKNOWN_VERSION_INFO["package"],
                    "version": str(info),
                }
            return {
                "method": method,
                "dataset": dataset,
                "package": info.get("package", UNKNOWN_VERSION_INFO["package"]),
                "version": info.get("version", UNKNOWN_VERSION_INFO["version"]),
                "component": component_idx,
            }

        version_map: dict[str, dict[str, Any]] = {}
        for _, method_name in self.methods:
            datasets = set(self.run_info.get(method_name, {}).keys()) | set(
                self.results.get(method_name, {}).keys()
            )
            if not datasets:
                continue
            ordered_datasets = [ds for ds in self.datasets if ds in datasets]
            for ds in datasets:
                if ds not in ordered_datasets:
                    ordered_datasets.append(ds)
            method_versions: dict[str, Any] = {}
            for dataset in ordered_datasets:
                method_versions[dataset] = self._version_for(method_name, dataset)
            if method_versions:
                version_map[method_name] = method_versions

        if not as_df:
            return version_map

        rows: list[dict[str, Any]] = []
        for method, ds_map in version_map.items():
            for dataset, info in ds_map.items():
                if isinstance(info, list):
                    for idx, component in enumerate(info):
                        rows.append(
                            _version_row(method, dataset, component, component_idx=idx)
                        )
                else:
                    rows.append(_version_row(method, dataset, info, component_idx=None))

        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "pandas is required to return versions as a DataFrame"
            ) from exc

        columns = [
            "method",
            "dataset",
            "package",
            "version",
            "component",
        ]
        return pd.DataFrame(rows, columns=columns)

    def plot(
        self,
        datasets: list[str] | None = None,
        methods: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        align: bool = True,
        grid_color: str = "#dddddd",
        **kwargs,
    ):
        return plot_report(
            self,
            datasets=datasets,
            methods=methods,
            figsize=figsize,
            align=align,
            grid_color=grid_color,
            **kwargs,
        )

    @classmethod
    def _from_manifest(cls, manifest_path: Path) -> "Experiment":
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        exp = cls(warn_on_existing=False)
        version = manifest.get("format_version")
        if version != EXPERIMENT_FORMAT_VERSION:
            raise ValueError(f"Unsupported experiment format version {version}")
        exp.name = manifest.get("name") or manifest_path.parent.name
        exp.drnb_home = manifest_path.parent.parent.parent
        exp.datasets = manifest.get("datasets", [])
        exp.evaluations = manifest.get("evaluations", [])
        exp.run_info = manifest.get("run_info", {})
        exp.methods = []
        for method_entry in manifest.get("methods", []):
            method_name = method_entry.get("name", "")
            method_config = method_from_manifest(method_entry.get("method", {}))
            exp.methods.append((method_config, method_name))

        exp.results = {}
        for _, method_name in exp.methods:
            method_results = {}
            run_entries = exp.run_info.get(method_name, {})
            for dataset in run_entries:
                method_results[dataset] = exp._lazy_result(method_name, dataset)
            if method_results:
                exp.results[method_name] = method_results
        return exp


def read_experiment(
    experiment_name: str,
) -> Experiment:
    """Read an experiment from the repository using the v2 manifest/shard format."""
    manifest = manifest_path(experiment_name, None, create=False)
    if not manifest.exists():
        raise FileNotFoundError(f"No manifest found for experiment '{experiment_name}'")
    log.info("Reading experiment %s from %s", experiment_name, manifest.parent)
    return Experiment._from_manifest(manifest)
