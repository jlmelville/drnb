from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import matplotlib.pyplot as plt
import pandas as pd

import drnb.embed.pipeline as pl
from drnb.embed import check_embed_method, get_embedder_name
from drnb.io import read_pickle, write_pickle
from drnb.log import log
from drnb.plot import result_plot
from drnb.util import dts_to_str


@dataclass
class Experiment:
    name: str = ""
    datasets: List[str] = field(default_factory=list)
    uniq_datasets: Set[str] = field(default_factory=set)
    methods: List = field(default_factory=list)
    uniq_method_names: Set[str] = field(default_factory=set)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluations: List = field(default_factory=list)

    def add_method(self, method, *, params=None, name: str = ""):
        method = check_embed_method(method, params)
        if not name:
            name = get_embedder_name(method)
        if name in self.uniq_method_names:
            raise ValueError(f"Experiment already has a embedding method '{name}'")
        self.methods.append((method, name))
        self.uniq_method_names.add(name)

    def add_datasets(self, datasets):
        for dataset in datasets:
            self.add_dataset(dataset)

    def add_dataset(self, dataset):
        if dataset in self.uniq_datasets:
            raise ValueError(f"Experiment already has a dataset '{dataset}'")
        self.uniq_datasets.add(dataset)
        self.datasets.append(dataset)

    def run(self):
        self.name = check_experiment(self.name)
        for method, method_name in self.methods:
            pipeline = pl.create_pipeline(
                method=method,
                eval_metrics=self.evaluations,
                plot=False,
            )
            method_results = self.results.get(method_name)
            if method_results is None:
                method_results = {}
                self.results[method_name] = method_results
            for dataset in self.datasets:
                if dataset in method_results:
                    continue
                log.info("Running %s on %s", method_name, dataset)
                embed_result = pipeline.run(dataset)
                method_results[dataset] = embed_result

    def to_df(self, datasets=None, methods=None, metrics=None):
        if methods is None:
            methods = [name for _, name in self.methods]
        if not isinstance(methods, list):
            methods = [methods]
        if not methods:
            return pd.DataFrame()
        if metrics is None:
            metrics = get_metric_names(list(self.results.values())[0])
        if datasets is None:
            datasets = self.datasets
        dfs = []
        for method in methods:
            dfs.append(
                results_to_df(self.results[method], datasets=datasets)[metrics],
            )
        df = pd.concat(dfs, axis=1, keys=methods)

        index = pd.MultiIndex.from_product(
            [methods, dfs[0].columns], names=["method", "metric"]
        )
        return df.reindex(index, axis=1)

    def plot(self, datasets=None, methods=None, figsize=None, **kwargs):
        """Plot the results of the experiment, datasets on the rows and methods on the
        columns."""
        if methods is None:
            methods = [name for _, name in self.methods]
        if datasets is None:
            datasets = self.datasets

        if figsize is None:
            figsize = (len(methods) * 6, len(datasets) * 4)
        _, axes = plt.subplots(nrows=len(datasets), ncols=len(methods), figsize=figsize)
        for i, dataset in enumerate(datasets):
            for j, method in enumerate(methods):
                if len(axes.shape) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                result_plot(
                    self.results[method][dataset],
                    ax=ax,
                    title=f"{method} on {dataset}",
                    **kwargs,
                )
        plt.show()
        plt.tight_layout()

    def save(self, compression="gzip", overwrite=False, name=None):
        """If `name` is provided, the experiment will be renamed."""
        if name is not None:
            self.name = name
            log.info("Renaming experiment to %s", self.name)
        self.name = check_experiment(self.name)

        write_pickle(
            x=self,
            name=self.name,
            suffix="",
            sub_dir="experiments",
            create_sub_dir=True,
            compression=compression,
            overwrite=overwrite,
        )


def read_experiment(name, compression="any"):
    return read_pickle(
        name,
        sub_dir="experiments",
        suffix="",
        compression=compression,
    )


def check_experiment(experiment):
    if experiment is None or not experiment:
        experiment = f"experiment-{dts_to_str()}"
        log.info("Using experiment name: %s", experiment)
    return experiment


def short_col(colname, sep="-") -> str:
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


def get_metric_names(results) -> List[str]:
    return [short_col(ev.label) for ev in list(results.values())[0]["evaluations"]]


def results_to_df(results, datasets=None):
    col_names = get_metric_names(results)
    if datasets is None:
        datasets = results.keys()
    df = pd.DataFrame(index=results.keys(), columns=col_names)
    for name, res in results.items():
        if name not in datasets:
            continue
        df.loc[name] = [ev.value for ev in res["evaluations"]]
    return df
