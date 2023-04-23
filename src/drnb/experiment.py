from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

import drnb.embed.pipeline as pl
from drnb.embed import check_embed_method, get_embedder_name
from drnb.log import log
from drnb.plot import result_plot
from drnb.util import default_dict, default_list, default_set, dts_to_str


@dataclass
class Experiment:
    name: str = ""
    datasets: list = default_list()
    uniq_datasets: set = default_set()
    methods: list = default_list()
    uniq_method_names: set = default_set()
    results: dict = default_dict()
    evaluations: list = default_list()

    def add_method(self, method, *, params=None, name=""):
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

    def plot(self, datasets=None, methods=None, figsize=None):
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
                result_plot(
                    self.results[method][dataset],
                    title=f"{method} on {dataset}",
                    ax=axes[i, j],
                )
        plt.tight_layout()


def check_experiment(experiment):
    if experiment is None or not experiment:
        experiment = f"experiment-{dts_to_str()}"
        log.info("Using experiment name: %s", experiment)
    return experiment


def short_col(colname, sep="-"):
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


def get_metric_names(results):
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
