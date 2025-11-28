from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from drnb.io import get_path
from drnb.log import log
from drnb.util import dts_to_str

if TYPE_CHECKING:
    import pandas as pd


EXPERIMENT_FORMAT_VERSION = 2
RESULT_FORMAT_VERSION = 1
RUN_STATUS_COMPLETED = "completed"
RESULT_JSON = "result.json"
MANIFEST_JSON = "manifest.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(name: str) -> str:
    return name.replace(os.sep, "_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _eval_to_dict(ev) -> dict:
    return {
        "eval_type": ev.eval_type,
        "label": ev.label,
        "value": ev.value,
        "info": _json_safe(ev.info),
    }


def _eval_from_dict(data: dict):
    from drnb.eval.base import EvalResult

    return EvalResult(
        eval_type=data.get("eval_type", ""),
        label=data.get("label", ""),
        value=data.get("value", 0.0),
        info=data.get("info", {}),
    )


def _method_to_manifest(method: Any) -> Any:
    if isinstance(method, list):
        return {"kind": "list", "value": [_method_to_manifest(m) for m in method]}
    if isinstance(method, tuple):
        return {
            "kind": "tuple",
            "value": [_json_safe(method[0]), _json_safe(method[1])],
        }
    return {"kind": "json", "value": _json_safe(method)}


def _method_from_manifest(entry: Any) -> Any:
    kind = entry.get("kind")
    value = entry.get("value")
    if kind == "list":
        return [_method_from_manifest(v) for v in value]
    if kind == "tuple":
        return (value[0], value[1])
    return value


def _context_to_dict(ctx: Any) -> dict | None:
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


def _context_from_dict(data: dict | None):
    if not data:
        return None
    from drnb.embed.context import EmbedContext

    home = data.get("drnb_home")
    if home is not None:
        data = {**data, "drnb_home": Path(home)}
    return EmbedContext(**data)


def _param_signature(method: Any, evaluations: list[Any]) -> str:
    payload = {
        "method": _method_to_manifest(method),
        "evaluations": _json_safe(evaluations),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _experiment_dir(
    name: str, drnb_home: Path | str | None, *, create: bool = True
) -> Path:
    base = get_path(drnb_home, sub_dir="experiments", create_sub_dir=create)
    exp_dir = base / name
    if create:
        exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _manifest_path(
    name: str, drnb_home: Path | str | None, *, create: bool = True
) -> Path:
    return _experiment_dir(name, drnb_home, create=create) / MANIFEST_JSON


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


@dataclass
class Experiment:
    """Run and checkpoint embedding experiments.

    Each (method, dataset) run is persisted immediately as a shard on disk. Subsequent
    runs skip shards whose parameter signatures match and rerun shards whose signatures
    differ, so outputs stay consistent with the code and params in use.
    """

    name: str = ""
    datasets: list[str] = field(default_factory=list)
    uniq_datasets: set[str] = field(default_factory=set)
    methods: list = field(default_factory=list)
    uniq_method_names: set[str] = field(default_factory=set)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    evaluations: list = field(default_factory=list)
    verbose: bool = False
    drnb_home: Path | str | None = None
    run_info: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _announced_output: bool = field(default=False, init=False, repr=False)

    def add_method(self, method, *, params=None, name: str = ""):
        """Add an embedding method to the experiment."""
        from drnb.embed import check_embed_method, get_embedder_name

        method = check_embed_method(method, params)
        if not name:
            name = get_embedder_name(method)
        if name in self.uniq_method_names:
            raise ValueError(f"Experiment already has a embedding method '{name}'")
        self.methods.append((method, name))
        self.uniq_method_names.add(name)

    def add_datasets(self, datasets: list[str]):
        """Add a list of datasets to the experiment."""
        for dataset in datasets:
            self.add_dataset(dataset)

    def add_dataset(self, dataset: str):
        """Add a dataset to the experiment."""
        if dataset in self.uniq_datasets:
            raise ValueError(f"Experiment already has a dataset '{dataset}'")
        self.uniq_datasets.add(dataset)
        self.datasets.append(dataset)

    def run(self):
        """Run the experiment, checkpointing each dataset/method pair as soon as it
        completes.

        If a prior shard exists and the signature matches, the work is skipped. If the
        signature differs, rerun and overwrite to keep results aligned with current
        parameters.
        """
        import drnb.embed.pipeline as pl

        self.name = ensure_experiment_name(self.name)
        self._ensure_sets()
        exp_dir = _experiment_dir(self.name, self.drnb_home, create=True)
        log.info("Experiment %s writing to %s", self.name, exp_dir)
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
            signature = _param_signature(method, self.evaluations)
            for dataset in self.datasets:
                run_record = self.run_info.get(method_name, {}).get(dataset)
                if run_record is not None:
                    prior_sig = run_record.get("signature")
                    if (
                        run_record.get("status") == RUN_STATUS_COMPLETED
                        and prior_sig == signature
                    ):
                        log.info(
                            "Skipping %s on %s (already completed, matching signature)",
                            method_name,
                            dataset,
                        )
                        if dataset not in method_results:
                            method_results[dataset] = self._lazy_result(
                                method_name, dataset
                            )
                        continue
                    if prior_sig != signature:
                        log.info(
                            "Rerunning %s on %s due to signature mismatch (prior=%s current=%s)",
                            method_name,
                            dataset,
                            prior_sig,
                            signature,
                        )
                log.info("Running %s on %s", method_name, dataset)
                embed_result = pipeline.run(dataset)
                method_results[dataset] = embed_result
                shard_path = self._write_result_shard(
                    method_name, dataset, embed_result
                )
                self._update_run_info(method_name, dataset, signature, shard_path)
                self._write_manifest()

    def _ensure_sets(self):
        if not self.uniq_method_names and self.methods:
            self.uniq_method_names = {name for _, name in self.methods}
        if not self.uniq_datasets and self.datasets:
            self.uniq_datasets = set(self.datasets)

    def _shard_dir(
        self, method_name: str, dataset: str, *, create: bool = True
    ) -> Path:
        exp_dir = _experiment_dir(self.name, self.drnb_home, create=create)
        run_record = self.run_info.get(method_name, {}).get(dataset)
        if run_record and "shard" in run_record:
            return exp_dir / run_record["shard"]
        return (
            exp_dir
            / "results"
            / _safe_component(method_name)
            / _safe_component(dataset)
        )

    def _lazy_result(self, method_name: str, dataset: str) -> LazyResult:
        return LazyResult(
            self._shard_dir(method_name, dataset, create=False), self._load_result_shard
        )

    def _write_result_shard(
        self, method_name: str, dataset: str, embed_result: dict
    ) -> Path:
        shard_dir = self._shard_dir(method_name, dataset)
        shard_dir.mkdir(parents=True, exist_ok=True)

        entries: dict[str, dict] = {}
        for key, value in embed_result.items():
            if key == "context":
                entries[key] = {"type": "context", "value": _context_to_dict(value)}
                continue
            if isinstance(value, np.ndarray):
                filename = f"{key}.npz"
                np.savez_compressed(shard_dir / filename, data=value)
                entries[key] = {"type": "npz", "file": filename}
                continue
            if is_dataclass(value):
                value = asdict(value)
            if hasattr(value, "eval_type") and hasattr(value, "label"):
                entries[key] = {"type": "eval_result", "value": _eval_to_dict(value)}
                continue
            if (
                isinstance(value, list)
                and value
                and all(hasattr(v, "eval_type") for v in value)
            ):
                entries[key] = {
                    "type": "eval_results",
                    "value": [_eval_to_dict(v) for v in value],
                }
                continue
            try:
                entries[key] = {"type": "json", "value": _json_safe(value)}
            except TypeError:
                entries[key] = {"type": "text", "value": str(value)}

        meta = {"version": RESULT_FORMAT_VERSION, "entries": entries}
        with open(shard_dir / RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        exp_dir = _experiment_dir(self.name, self.drnb_home, create=True)
        return shard_dir.relative_to(exp_dir)

    def _load_result_shard(self, shard_dir: Path) -> dict:
        meta_path = shard_dir / RESULT_JSON
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        entries = meta.get("entries", {})
        result = {}
        for key, entry in entries.items():
            entry_type = entry.get("type")
            if entry_type == "npz":
                data = np.load(shard_dir / entry["file"])
                arr = data[data.files[0]]
                result[key] = arr
            elif entry_type == "context":
                result[key] = _context_from_dict(entry.get("value"))
            elif entry_type == "eval_result":
                result[key] = _eval_from_dict(entry["value"])
            elif entry_type == "eval_results":
                result[key] = [_eval_from_dict(v) for v in entry.get("value", [])]
            else:
                result[key] = entry.get("value")
        return result

    def _update_run_info(
        self, method_name: str, dataset: str, signature: str, shard_path: Path
    ):
        if method_name not in self.run_info:
            self.run_info[method_name] = {}
        self.run_info[method_name][dataset] = {
            "status": RUN_STATUS_COMPLETED,
            "signature": signature,
            "updated_at": _now_iso(),
            "shard": str(shard_path),
        }

    def _write_manifest(self):
        exp_dir = _experiment_dir(self.name, self.drnb_home, create=True)
        manifest = {
            "format_version": EXPERIMENT_FORMAT_VERSION,
            "name": self.name,
            "datasets": self.datasets,
            "methods": [
                {"name": name, "method": _method_to_manifest(method)}
                for method, name in self.methods
            ],
            "evaluations": _json_safe(self.evaluations),
            "run_info": self.run_info,
        }
        with open(exp_dir / MANIFEST_JSON, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def clear_task(self, method_name: str, dataset: str):
        """Drop a single shard and run record so the pair will rerun on the next `run`."""
        shard_dir = self._shard_dir(method_name, dataset, create=False)
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        if method_name in self.run_info:
            self.run_info[method_name].pop(dataset, None)
        if method_name in self.results:
            self.results[method_name].pop(dataset, None)
            if not self.results[method_name]:
                self.results.pop(method_name, None)
        self._write_manifest()

    def clear_method(self, method_name: str):
        """Drop all shards and run records for a method so every dataset reruns."""
        datasets = list(self.run_info.get(method_name, {}).keys())
        for dataset in datasets:
            self.clear_task(method_name, dataset)

    def reset(self):
        """Delete all shards and metadata for this experiment (destructive)."""
        try:
            exp_dir = _experiment_dir(self.name, self.drnb_home, create=False)
        except FileNotFoundError:
            exp_dir = None
        if exp_dir and exp_dir.exists():
            shutil.rmtree(exp_dir)
        self.results.clear()
        self.run_info.clear()
        self.datasets = []
        self.uniq_datasets = set()
        self.methods = []
        self.uniq_method_names = set()
        self.evaluations = []

    @classmethod
    def _from_manifest(cls, manifest_path: Path) -> Experiment:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        exp = cls()
        version = manifest.get("format_version")
        if version != EXPERIMENT_FORMAT_VERSION:
            raise ValueError(f"Unsupported experiment format version {version}")
        exp.name = manifest.get("name") or manifest_path.parent.name
        exp.drnb_home = manifest_path.parent.parent.parent
        exp.datasets = manifest.get("datasets", [])
        exp.uniq_datasets = set(exp.datasets)
        exp.evaluations = manifest.get("evaluations", [])
        exp.run_info = manifest.get("run_info", {})
        exp.methods = []
        exp.uniq_method_names = set()
        for method_entry in manifest.get("methods", []):
            method_name = method_entry.get("name", "")
            method_config = _method_from_manifest(method_entry.get("method", {}))
            exp.methods.append((method_config, method_name))
            if method_name:
                exp.uniq_method_names.add(method_name)

        exp.results = {}
        for _, method_name in exp.methods:
            method_results = {}
            run_entries = exp.run_info.get(method_name, {})
            for dataset in run_entries:
                method_results[dataset] = exp._lazy_result(method_name, dataset)
            if method_results:
                exp.results[method_name] = method_results
        return exp

    def to_df(
        self,
        datasets: list[str] | None = None,
        methods: list[str] | str | None = None,
        metrics: list[str] | None = None,
    ):
        """Convert the results of the experiment to a DataFrame."""
        import pandas as pd

        if methods is None:
            methods = [name for _, name in self.methods]
        if not isinstance(methods, list):
            methods = [methods]
        if not methods:
            return pd.DataFrame()
        if metrics is None:
            if not self.results:
                return pd.DataFrame()
            first_method_results = list(self.results.values())[0]
            first_dataset_result = next(iter(first_method_results.values()), None)
            if not first_dataset_result or "evaluations" not in first_dataset_result:
                log.info("No evaluations found; returning empty DataFrame")
                return pd.DataFrame()
            metrics = get_metric_names(first_method_results)
        if datasets is None:
            datasets = self.datasets
        dfs = []
        for method in methods:
            df = results_to_df(self.results[method], datasets=datasets)
            if df.empty:
                log.info(
                    "No evaluations found for method %s; returning empty DataFrame",
                    method,
                )
                return pd.DataFrame()
            dfs.append(df[metrics])
        df = pd.concat(dfs, axis=1, keys=methods)

        index = pd.MultiIndex.from_product(
            [methods, dfs[0].columns], names=["method", "metric"]
        )
        return df.reindex(index, axis=1)

    def plot(
        self,
        datasets: list[str] | None = None,
        methods: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        align: bool = True,
        grid_color: str = "#dddddd",  # light gray color for grid
        **kwargs,
    ):
        """Plot the results of the experiment, datasets on the rows and methods on the
        columns.

        Parameters
        ----------
        datasets : List[str] | None
            List of datasets to plot. If None, uses all datasets.
        methods : List[str] | None
            List of methods to plot. If None, uses all methods.
        figsize : Tuple[float, float] | None
            Figure size (width, height). If None, calculated based on number of plots.
        align : bool
            If True, align embeddings using Kabsch alignment to the first dataset.
        grid_color : str
            Color for the vertical grid lines between plots. Set to "" to disable.
        **kwargs
            Additional arguments passed to result_plot.
        """
        import matplotlib.pyplot as plt

        from drnb.embed import get_coords
        from drnb.plot.common import result_plot

        if not self.results:
            raise ValueError("No results to plot")

        if methods is None:
            methods = [name for _, name in self.methods]
        if datasets is None:
            datasets = self.datasets

        if figsize is None:
            figsize = (len(methods) * 6, len(datasets) * 4)

        fig, axes = plt.subplots(
            nrows=len(datasets), ncols=len(methods), figsize=figsize, squeeze=False
        )

        fixed = None
        for i, dataset in enumerate(datasets):
            for j, method in enumerate(methods):
                if j == 0:
                    # We need to explicitly reset this for each row. It's possible
                    # that the first result doesn't exist and so this part of the loop
                    # will skip, in which case we can accidentally inherit the fixed
                    # coordinates from the previous row.
                    fixed = None
                result = self.results.get(method, {}).get(dataset)
                if result is None:
                    axes[i, j].axis("off")
                    axes[i, j].text(
                        0.5,
                        0.5,
                        "No result",
                        ha="center",
                        va="center",
                        transform=axes[i, j].transAxes,
                    )
                    continue
                if isinstance(result, LazyResult):
                    result = result.materialize()
                if (
                    not isinstance(result, dict)
                    or "coords" not in result
                    or result.get("context") is None
                ):
                    axes[i, j].axis("off")
                    axes[i, j].text(
                        0.5,
                        0.5,
                        "No result",
                        ha="center",
                        va="center",
                        transform=axes[i, j].transAxes,
                    )
                    continue
                if align and j == 0:
                    fixed = get_coords(result)

                result_plot(
                    result,
                    ax=axes[i, j],
                    title=f"{method} on {dataset}",
                    fixed=fixed,
                    **kwargs,
                )
        plt.tight_layout()

        # Add subtle vertical lines between subplots if grid_color is specified
        if grid_color:
            for i in range(len(datasets)):
                for j in range(1, len(methods)):
                    # Get the position of the current subplot
                    pos = axes[i, j].get_position()
                    # Draw a vertical line at the left edge of the subplot
                    fig.add_artist(
                        plt.Line2D(
                            [pos.x0, pos.x0],  # x coordinates
                            [pos.y0, pos.y1],  # y coordinates
                            color=grid_color,
                            linewidth=1,
                            transform=fig.transFigure,
                            clip_on=False,
                        )
                    )

        plt.show()

    def save(
        self,
        name: str | None = None,
    ):
        """Save the experiment to the repository. If `name` is provided, the experiment
        will be renamed.

        Parameters
        ----------

        name : str | None
            Optional new name for the experiment.
        """
        if name is not None:
            self.name = name
            log.info("Renaming experiment to %s", self.name)
        self.name = ensure_experiment_name(self.name)
        self._ensure_sets()
        exp_dir = _experiment_dir(self.name, self.drnb_home, create=True)
        for method, method_name in self.methods:
            method_results = self.results.get(method_name, {})
            signature = _param_signature(method, self.evaluations)
            for dataset, result in method_results.items():
                run_record = self.run_info.get(method_name, {}).get(dataset)
                if isinstance(result, LazyResult):
                    if run_record is not None:
                        continue
                    result = result.materialize()
                shard_path = self._write_result_shard(method_name, dataset, result)
                self._update_run_info(method_name, dataset, signature, shard_path)
        self._write_manifest()


def read_experiment(
    experiment_name: str,
) -> Experiment:
    """Read an experiment from the repository using the v2 manifest/shard format."""
    manifest = _manifest_path(experiment_name, None, create=False)
    if not manifest.exists():
        raise FileNotFoundError(f"No manifest found for experiment '{experiment_name}'")
    log.info("Reading experiment %s from %s", experiment_name, manifest.parent)
    return Experiment._from_manifest(manifest)


def ensure_experiment_name(experiment_name: str | None) -> str:
    """Ensure that the experiment name is not empty and return a default name if it
    is."""
    if experiment_name is None or not experiment_name:
        experiment_name = f"experiment-{dts_to_str()}"
        log.info("Using experiment name: %s", experiment_name)
    return experiment_name


def short_col(colname: str, sep: str = "-") -> str:
    """Return everything up to (but not including) the second `sep` in the string
    `colname` or return `colname` in its entirety if `sep` doesn't occur twice.

    Shortens longer evaluation labels, e.g. `nnp-15-noself-euclidean` becomes `nnp-15`.
    """
    index = colname.find(sep)
    if index == -1:
        return colname
    index2 = colname.find(sep, index + 1)
    if index2 == -1:
        return colname
    return colname[:index2]


def get_metric_names(results: dict[str, Any]) -> list[str]:
    """Get the metric names from the first entry in the results dictionary."""
    first = next(iter(results.values()), None)
    if not first or "evaluations" not in first:
        return []
    return [short_col(ev.label) for ev in first["evaluations"]]


def results_to_df(
    results: dict[str, Any], datasets: list[str] | None = None
) -> pd.DataFrame:
    """Convert the results of an experiment to a DataFrame."""
    import pandas as pd

    col_names = get_metric_names(results)
    if not col_names:
        return pd.DataFrame()
    if datasets is None:
        datasets = list(results.keys())
    df = pd.DataFrame(index=datasets, columns=col_names)
    for name in datasets:
        res = results.get(name)
        if not res or "evaluations" not in res:
            continue
        eval_map = {short_col(ev.label): ev.value for ev in res["evaluations"]}
        df.loc[name] = [eval_map.get(col) for col in col_names]
    return df


def merge_experiments(
    exp1: Experiment, exp2: Experiment, name: str | None = None
) -> Experiment:
    """Merge two experiments, allowing holes when datasets are missing in one side.

    Parameters
    ----------
    exp1 : Experiment
        First experiment
    exp2 : Experiment
        Second experiment
    name : str | None, optional
        Name for the merged experiment. If None, will generate a name by combining
        the original experiment names.

    Returns
    -------
    Experiment
        A new experiment containing the merged results

    Notes
    -----
    The merged experiment will:
    - Include all datasets present in either experiment (order preserves exp1.datasets,
    then new items from exp2.datasets)
    - Combine methods and results from both experiments; missing method/dataset pairs
    are left empty
    - Combine evaluations from both experiments (deduplicated, preserving order)
    - Use provided name or generate one by combining the original experiment names
    """
    # Union of datasets preserving order
    merged_datasets: list[str] = []
    seen = set()
    for dataset in exp1.datasets + exp2.datasets:
        if dataset not in seen:
            merged_datasets.append(dataset)
            seen.add(dataset)

    # Create new experiment
    merged = Experiment()

    for dataset in merged_datasets:
        merged.add_dataset(dataset)

    # Add methods and results from exp1
    for method, name in exp1.methods:
        merged.add_method(method, name=name)
        merged.results[name] = {}
        for dataset, res in exp1.results.get(name, {}).items():
            merged.results[name][dataset] = res

    # Add methods and results from exp2
    for method, name in exp2.methods:
        if name not in merged.uniq_method_names:
            merged.add_method(method, name=name)
            merged.results[name] = {}
        for dataset, res in exp2.results.get(name, {}).items():
            merged.results[name][dataset] = res

    # Combine evaluations
    merged.evaluations = _merge_evaluations(exp1.evaluations, exp2.evaluations)

    # Set name
    merged.name = name if name is not None else f"merged-{exp1.name}-{exp2.name}"

    return merged


def _merge_evaluations(eval1: list[Any], eval2: list[Any]) -> list[Any]:
    """Combine evaluation lists, preserving order and handling unhashable entries."""
    merged: list[Any] = []
    for ev in list(eval1) + list(eval2):
        if not any(existing == ev for existing in merged):
            merged.append(ev)
    return merged
