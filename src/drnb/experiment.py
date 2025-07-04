from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import drnb.embed.pipeline as pl
from drnb.embed import check_embed_method, get_coords, get_embedder_name
from drnb.io import read_pickle, write_pickle
from drnb.log import log
from drnb.plot.common import result_plot
from drnb.util import dts_to_str


@dataclass
class Experiment:
    """Class to run and store the results of an experiment."""

    name: str = ""
    datasets: List[str] = field(default_factory=list)
    uniq_datasets: Set[str] = field(default_factory=set)
    methods: List = field(default_factory=list)
    uniq_method_names: Set[str] = field(default_factory=set)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluations: List = field(default_factory=list)
    verbose: bool = False

    def add_method(self, method, *, params=None, name: str = ""):
        """Add an embedding method to the experiment."""
        method = check_embed_method(method, params)
        if not name:
            name = get_embedder_name(method)
        if name in self.uniq_method_names:
            raise ValueError(f"Experiment already has a embedding method '{name}'")
        self.methods.append((method, name))
        self.uniq_method_names.add(name)

    def add_datasets(self, datasets: List[str]):
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
        """Run the experiment."""
        self.name = ensure_experiment_name(self.name)
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
            for dataset in self.datasets:
                if dataset in method_results:
                    continue
                log.info("Running %s on %s", method_name, dataset)
                embed_result = pipeline.run(dataset)
                method_results[dataset] = embed_result

    def to_df(
        self,
        datasets: List[str] | None = None,
        methods: List[str] | str | None = None,
        metrics: List[str] | None = None,
    ):
        """Convert the results of the experiment to a DataFrame."""
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

    def plot(
        self,
        datasets: List[str] | None = None,
        methods: List[str] | None = None,
        figsize: Tuple[float, float] | None = None,
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
                if align and j == 0:
                    fixed = get_coords(self.results[method][dataset])

                result_plot(
                    self.results[method][dataset],
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
        compression: Literal["gzip", "bz2", ""] | None = "gzip",
        overwrite: bool = False,
        name: str = None,
    ):
        """Save the experiment to the repository. If `name` is provided, the experiment
        will be renamed."""
        if name is not None:
            self.name = name
            log.info("Renaming experiment to %s", self.name)
        self.name = ensure_experiment_name(self.name)

        write_pickle(
            x=self,
            name=self.name,
            suffix="",
            sub_dir="experiments",
            create_sub_dir=True,
            compression=compression,
            overwrite=overwrite,
        )


def read_experiment(
    experiment_name: str,
    compression: List[str] | Literal["gzip", "bz2", "any", ""] = "any",
) -> Experiment:
    """Read an experiment from the repository."""
    return read_pickle(
        experiment_name,
        sub_dir="experiments",
        suffix="",
        compression=compression,
    )


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


def get_metric_names(results: Dict[str, Any]) -> List[str]:
    """Get the metric names from the first entry in the results dictionary."""
    return [short_col(ev.label) for ev in list(results.values())[0]["evaluations"]]


def results_to_df(
    results: Dict[str, Any], datasets: List[str] | None = None
) -> pd.DataFrame:
    """Convert the results of an experiment to a DataFrame."""
    col_names = get_metric_names(results)
    if datasets is None:
        datasets = results.keys()
    df = pd.DataFrame(index=results.keys(), columns=col_names)
    for name, res in results.items():
        if name not in datasets:
            continue
        df.loc[name] = [ev.value for ev in res["evaluations"]]
    return df


def merge_experiments(
    exp1: Experiment, exp2: Experiment, name: str | None = None
) -> Experiment:
    """Merge two experiments, keeping only datasets that exist in both.

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
    - Include only datasets present in both experiments
    - Combine methods and results from both experiments
    - Use evaluations from both experiments
    - Use provided name or generate one by combining the original experiment names
    """
    # Find common datasets
    common_datasets = exp1.uniq_datasets.intersection(exp2.uniq_datasets)
    if not common_datasets:
        raise ValueError("No datasets in common between experiments")

    # Create new experiment
    merged = Experiment()

    # Add common datasets in the order they appear in exp1
    for dataset in exp1.datasets:
        if dataset in common_datasets:
            merged.add_dataset(dataset)

    # Add methods and results from exp1
    for method, name in exp1.methods:
        merged.add_method(method, name=name)
        merged.results[name] = {
            dataset: exp1.results[name][dataset] for dataset in common_datasets
        }

    # Add methods and results from exp2
    for method, name in exp2.methods:
        if name not in merged.uniq_method_names:
            merged.add_method(method, name=name)
            merged.results[name] = {
                dataset: exp2.results[name][dataset] for dataset in common_datasets
            }

    # Combine evaluations
    merged.evaluations = list(set(exp1.evaluations + exp2.evaluations))

    # Set name
    merged.name = name if name is not None else f"merged-{exp1.name}-{exp2.name}"

    return merged
