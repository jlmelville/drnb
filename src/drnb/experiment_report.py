from __future__ import annotations

from typing import Any

import numpy as np

from drnb.experiment_common import (
    RUN_STATUS_PARTIAL_EVALS,
    expected_eval_labels,
    param_signature,
    short_col,
)
from drnb.experiment_persistence import LazyResult


def status(experiment: "Experiment"):
    """Return a status summary DataFrame (rows=datasets, columns=methods)."""
    import pandas as pd

    from drnb.experiment_runner import result_progress

    methods = [name for _, name in experiment.methods]
    expected_labels = expected_eval_labels(experiment.evaluations)
    df = pd.DataFrame(index=experiment.datasets, columns=methods)
    for method in methods:
        method_config = next((m for m, n in experiment.methods if n == method), None)
        current_sig = (
            param_signature(method_config, experiment.evaluations)
            if method_config is not None
            else None
        )
        for dataset in experiment.datasets:
            run_entry = experiment.run_info.get(method, {}).get(dataset)
            res = experiment.results.get(method, {}).get(dataset)
            status_value: str
            completed = run_entry.get("evals_completed") if run_entry else None
            expected = run_entry.get("evals_expected") if run_entry else None
            sig_mismatch = (
                current_sig is not None
                and run_entry is not None
                and current_sig != run_entry.get("signature")
            )
            if sig_mismatch or run_entry is None:
                status_value, completed, expected, _ = result_progress(
                    res, expected_labels
                )
            else:
                status_value = run_entry.get("status", "missing")
                if completed is None or expected is None:
                    status_value, completed, expected, _ = result_progress(
                        res, expected_labels
                    )
            if status_value == RUN_STATUS_PARTIAL_EVALS and expected is not None:
                df.loc[dataset, method] = (
                    f"{RUN_STATUS_PARTIAL_EVALS}({completed}/{expected})"
                )
            else:
                df.loc[dataset, method] = status_value
    return df


def to_df(
    experiment: "Experiment",
    datasets: list[str] | None = None,
    methods: list[str] | str | None = None,
    metrics: list[str] | None = None,
):
    """Convert the results of the experiment to a DataFrame."""
    import pandas as pd

    if methods is None:
        methods = [name for _, name in experiment.methods]
    if not isinstance(methods, list):
        methods = [methods]
    if not methods:
        return pd.DataFrame()
    if metrics is None:
        metrics = []
        for method in methods:
            method_results = experiment.results.get(method, {})
            for metric_name in get_metric_names(method_results):
                if metric_name not in metrics:
                    metrics.append(metric_name)
        if not metrics:
            return pd.DataFrame()
    if datasets is None:
        datasets = experiment.datasets
    dfs = []
    for method in methods:
        method_results = experiment.results.get(method, {})
        df = pd.DataFrame(index=datasets, columns=metrics)
        for dataset in datasets:
            res = method_results.get(dataset)
            if not res or "evaluations" not in res:
                continue
            if isinstance(res, LazyResult):
                res = res.materialize()
            eval_map = {short_col(ev.label): ev.value for ev in res["evaluations"]}
            df.loc[dataset] = [eval_map.get(metric) for metric in metrics]
        dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=methods)

    index = pd.MultiIndex.from_product(
        [methods, dfs[0].columns], names=["method", "metric"]
    )
    return df.reindex(index, axis=1)


def plot(
    experiment: "Experiment",
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    align: bool = True,
    grid_color: str = "#dddddd",
    **kwargs,
):
    """Plot the results of the experiment, datasets on the rows and methods on the columns."""
    import matplotlib.pyplot as plt

    from drnb.embed import get_coords
    from drnb.plot.common import result_plot

    if not experiment.results:
        raise ValueError("No results to plot")

    if methods is None:
        methods = [name for _, name in experiment.methods]
    if datasets is None:
        datasets = experiment.datasets

    if figsize is None:
        figsize = (len(methods) * 6, len(datasets) * 4)

    fig, axes = plt.subplots(
        nrows=len(datasets), ncols=len(methods), figsize=figsize, squeeze=False
    )

    fixed = None
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            if j == 0:
                fixed = None
            result = experiment.results.get(method, {}).get(dataset)
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

    if grid_color:
        for i in range(len(datasets)):
            for j in range(1, len(methods)):
                pos = axes[i, j].get_position()
                fig.add_artist(
                    plt.Line2D(
                        [pos.x0, pos.x0],
                        [pos.y0, pos.y1],
                        color=grid_color,
                        linewidth=1,
                        transform=fig.transFigure,
                        clip_on=False,
                    )
                )

    plt.show()


def get_metric_names(results: dict[str, Any]) -> list[str]:
    """Get the metric names from the first entry in the results dictionary."""
    first = next(iter(results.values()), None)
    if not first or "evaluations" not in first:
        return []
    if isinstance(first, LazyResult):
        first = first.materialize()
    return [short_col(ev.label) for ev in first["evaluations"]]


def results_to_df(results: dict[str, Any], datasets: list[str] | None = None):
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
        if isinstance(res, LazyResult):
            res = res.materialize()
        if not res or "evaluations" not in res:
            continue
        eval_map = {short_col(ev.label): ev.value for ev in res["evaluations"]}
        df.loc[name] = [eval_map.get(col) for col in col_names]
    return df
