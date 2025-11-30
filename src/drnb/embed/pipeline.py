from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

import drnb.io as nbio
import drnb.io.dataset as dataio
from drnb.embed import (
    check_embed_method,
    embedder,
    get_embedder_name,
)
from drnb.embed.base import Embedder
from drnb.embed.context import EmbedContext
from drnb.embed.factory import create_embedder
from drnb.embed.version import get_embedder_version_info
from drnb.eval.base import EmbeddingEval, EvalResult, evaluate_embedding
from drnb.eval.factory import create_evaluators
from drnb.io.embed import create_embed_exporter
from drnb.log import log, log_verbosity
from drnb.types import ActionConfig, EmbedConfig
from drnb.util import dts_to_str

if TYPE_CHECKING:
    from drnb.plot.protocol import PlotterProtocol
    from drnb.plot.scale.ko import ColorByKo
    from drnb.plot.scale.lid import ColorByLid
    from drnb.plot.scale.nbrpres import ColorByNbrPres
    from drnb.plot.scale.rte import ColorByRte
    from drnb.plot.scale.so import ColorBySo


@dataclass
class EmbedPipelineExporter:
    """Exporter for embedded data and data useful for evaluation (e.g neighbors)."""

    out_types: list[str] = field(default_factory=list)
    export_dict: dict = field(default_factory=dict)

    def cache_data(
        self, requirers: list[dict], embedding_result: dict, ctx: EmbedContext
    ):
        """Cache derived data (neighbors, triplets) if needed."""
        embed_coords = embedding_result["coords"]

        for requirer in requirers:
            requires = requirer.requires()
            if requires is None or not requires:
                continue
            if isinstance(requires, dict):
                requires = [requires]
            for require in requires:
                require_name = require.get("name", "unknown")
                if require_name == "triplets":
                    self.cache_triplets(require, embed_coords, ctx)
                elif require_name == "neighbors":
                    self.cache_neighbors(require, embed_coords, ctx)
                else:
                    log.info("Don't know how to cache %s, skipping", require_name)

    def cache_triplets(
        self, require: dict, embed_coords: np.ndarray, ctx: EmbedContext
    ):
        """Cache the triplets if needed."""
        from drnb.triplets import create_triplets_request, find_triplet_files

        triplet_info = find_triplet_files(
            name=ctx.embed_triplets_name,
            n_triplets_per_point=require["n_triplets_per_point"],
            drnb_home=ctx.drnb_home,
            sub_dir=ctx.experiment_name,
            metric=require["metric"],
            seed=require["random_state"],
        )
        if triplet_info:
            return

        triplet_info = find_triplet_files(
            name=ctx.dataset_name,
            n_triplets_per_point=require["n_triplets_per_point"],
            drnb_home=ctx.drnb_home,
            sub_dir=ctx.triplet_sub_dir,
            metric=require["metric"],
            seed=require["random_state"],
        )
        if not triplet_info:
            return

        triplet_info = triplet_info[0]
        triplets_request = create_triplets_request(
            {
                "n_triplets_per_point": triplet_info.n_triplets_per_point,
                "seed": triplet_info.seed,
                "metric": require["metric"],
            }
        )
        if triplets_request is None:
            return
        triplet_output_paths = triplets_request.create_triplets(
            embed_coords,
            dataset_name=ctx.embed_triplets_name,
            triplet_dir=ctx.experiment_name,
        )

        if "triplets" not in self.export_dict:
            self.export_dict["triplets"] = []
        self.export_dict["triplets"].append(
            {
                "request": triplets_request,
                "paths": nbio.stringify_paths(triplet_output_paths),
            }
        )

    def cache_neighbors(
        self, require: dict, embed_coords: np.ndarray, ctx: EmbedContext
    ):
        """Cache the nearest neighbors if needed."""
        from drnb.neighbors.compute import create_neighbors_request
        from drnb.neighbors.store import find_candidate_neighbors_info

        neighbors_info = find_candidate_neighbors_info(
            name=ctx.embed_nn_name,
            n_neighbors=require["n_neighbors"],
            metric=require["metric"],
            exact=True,
            return_distance=True,
            drnb_home=ctx.drnb_home,
            sub_dir=ctx.experiment_name,
        )

        if neighbors_info:
            return

        neighbors_info = find_candidate_neighbors_info(
            name=ctx.dataset_name,
            n_neighbors=require["n_neighbors"],
            metric=require["metric"],
            exact=True,
            return_distance=False,
            drnb_home=ctx.drnb_home,
            sub_dir=ctx.nn_sub_dir,
        )
        if not neighbors_info:
            return

        neighbors_request = create_neighbors_request(
            {"n_neighbors": require["n_neighbors"], "metric": require["metric"]}
        )
        if neighbors_request is None:
            return
        neighbors_output_paths = neighbors_request.create_neighbors(
            embed_coords,
            dataset_name=ctx.embed_nn_name,
            nbr_dir=ctx.experiment_name,
        )
        if not neighbors_output_paths:
            return

        if "neighbors" not in self.export_dict:
            self.export_dict["neighbors"] = []
        self.export_dict["neighbors"].append(
            {
                "request": neighbors_request,
                "paths": nbio.stringify_paths(neighbors_output_paths),
            }
        )

    def export(self, embedding_result: dict, ctx: EmbedContext):
        """Export the embedded data."""
        if self.out_types is not None:
            exporters = create_embed_exporter(
                embed_method_label=ctx.embed_method_label,
                out_type=self.out_types,
                sub_dir=ctx.experiment_name,
                suffix=None,
                create_sub_dir=True,
                drnb_home=ctx.drnb_home,
                verbose=True,
            )
            for exporter in exporters:
                exporter.export(name=ctx.dataset_name, embedded=embedding_result)

        if "evaluations" in embedding_result:
            self.export_dict["evaluations"] = embedding_result["evaluations"]

        if self.export_dict:
            nbio.write_json(
                self.export_dict,
                name=ctx.dataset_name,
                suffix=ctx.embed_method_label,
                drnb_home=ctx.drnb_home,
                sub_dir=ctx.experiment_name,
                create_sub_dir=True,
                verbose=True,
            )
            self.export_dict.clear()


# pylint: disable=too-many-instance-attributes
@dataclass
class EmbedderPipeline:
    """Pipeline for embedding data and evaluating the embedding. The pipeline can
    include multiple evaluators and plotters."""

    embed_method_name: str
    # a way to refer to a specific parameterization of an embedding method,
    # e.g. the method might be umap, but you may want to refer to it as densvis
    embed_method_variant: str = ""
    reader: dataio.DatasetReader = field(default_factory=dataio.DatasetReader)
    embedder: Embedder | list[Embedder] = field(default_factory=list)
    evaluators: list[EmbeddingEval] = field(default_factory=list)
    plotters: list[PlotterProtocol] = field(default_factory=list)
    exporter: EmbedPipelineExporter | None = None
    verbose: bool = False
    pipeline_name: str | None = None

    def run_many(self, dataset_names: list[str], verbose: bool | None = None) -> dict:
        """Run the pipeline on multiple datasets. Returns a dictionary mapping from the
        dataset name to the embedding result."""
        if verbose is None:
            verbose = self.verbose
        with log_verbosity(verbose):
            results = {}
            for dataset_name in dataset_names:
                log.info("Running %s", dataset_name)
                results[dataset_name] = self._run(dataset_name)
        return results

    def run(self, dataset_name: str, verbose: bool | None = None) -> dict:
        """Run the pipeline. Returns the embedding result."""
        if verbose is None:
            verbose = self.verbose
        with log_verbosity(verbose):
            return self._run(dataset_name)

    def _run(self, dataset_name) -> dict:
        if self.pipeline_name is None:
            self.pipeline_name = f"pipeline-{dts_to_str()}"

        ctx = EmbedContext(
            embed_method_name=self.embed_method_name,
            embed_method_variant=self.embed_method_variant,
            dataset_name=dataset_name,
            drnb_home=self.reader.drnb_home,
            data_sub_dir=self.reader.sub_dir,
            experiment_name=self.pipeline_name,
        )
        log.info("Getting dataset %s", ctx.dataset_name)
        x, y = self.reader.read_data(ctx.dataset_name)

        log.info("Embedding")
        if isinstance(self.embedder, list):
            embedding_result = None
            for _embedder in self.embedder:
                if embedding_result is not None:
                    if isinstance(embedding_result, dict):
                        # pylint: disable=unsubscriptable-object
                        embedding_result = embedding_result["coords"]
                _embedder.precomputed_init = embedding_result
                embedding_result = _embedder.embed(x, ctx=ctx)
        else:
            embedding_result = self.embedder.embed(x, ctx=ctx)
        if not isinstance(embedding_result, dict):
            embedding_result = {"coords": embedding_result}
        embedding_result = cast(dict[str, Any], embedding_result)
        if "version_info" not in embedding_result:
            embedding_result["version_info"] = get_embedder_version_info(
                self.embedder, self.embed_method_name
            )

        if self.exporter is not None:
            log.info("Caching data")
            self.exporter.cache_data(
                self.evaluators + self.plotters, embedding_result, ctx
            )

        log.info("Evaluating")
        evaluations = evaluate_embedding(self.evaluators, x, embedding_result, ctx=ctx)
        if evaluations:
            embedding_result["evaluations"] = evaluations

        if self.exporter is not None:
            log.info("Exporting")
            self.exporter.export(embedding_result=embedding_result, ctx=ctx)

        if self.plotters:
            log.info("Plotting")
            for plotter in self.plotters:
                plotter.plot(embedding_result, data=x, y=y, ctx=ctx)

        embedding_result["context"] = ctx
        return embedding_result


def create_exporter(
    export: str | list[str] | bool | None,
) -> EmbedPipelineExporter | None:
    """Create an exporter for the pipeline, using the file types specified in `export`.
    If `export` is None, no exporter is created. If `export` is True, the default file
    type is used (pkl)."""
    if export is not None:
        if not isinstance(export, (list, tuple)):
            if isinstance(export, bool):
                if export:
                    export = ["pkl"]
                else:
                    export = None
            else:
                export = [export]
    if export is not None:
        return EmbedPipelineExporter(
            out_types=export,
        )
    return None


def create_pipeline(
    method: EmbedConfig | ActionConfig | list | Callable,
    data_config: dict | None = None,
    plot: bool | dict | str = True,
    eval_metrics: str | list[str] | None = None,
    export: str | list[str] | bool | None = None,
    verbose: bool = False,
    embed_method_variant: str = "",
) -> EmbedderPipeline:
    """Create an embedding pipeline.

    Args:
        method: Embedding method or a list of embedding methods.
        data_config: Configuration for reading the dataset.
        plot: Whether to plot the embedding or a dictionary of plot options.
        eval_metrics: Evaluation metrics to use.
        export: File types to export the embedding to.
        verbose: Whether to log verbosely.
        embed_method_variant: A way to refer to a specific parameterization of an
            embedding method.
    """
    if data_config is None:
        data_config = {}
    reader = dataio.DatasetReader(**data_config)
    # shut up pylint
    _embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotters = []
    if plot:
        from drnb.plot.factory import create_plotters

        plotters = create_plotters(plot)
    exporter = create_exporter(export)

    return EmbedderPipeline(
        embed_method_name=get_embedder_name(method),
        embed_method_variant=embed_method_variant,
        reader=reader,
        embedder=_embedder,
        evaluators=evaluators,
        plotters=plotters,
        exporter=exporter,
        verbose=verbose,
    )


def color_by_ko(
    n_neighbors: int,
    color_scale: dict | None = None,
    normalize: bool = True,
    log1p: bool = False,
) -> ColorByKo:
    """Create a Color by K-Occurrence plotter."""
    from drnb.plot.scale import ColorScale
    from drnb.plot.scale.ko import ColorByKo

    return ColorByKo(
        n_neighbors,
        scale=ColorScale.new(color_scale),
        normalize=normalize,
        log1p=log1p,
    )


def color_by_so(
    n_neighbors: int,
    color_scale: dict | None = None,
    normalize: bool = True,
    log1p: bool = False,
) -> ColorBySo:
    """Create a Color by S-Occurrence plotter."""
    from drnb.plot.scale import ColorScale
    from drnb.plot.scale.so import ColorBySo

    return ColorBySo(
        n_neighbors,
        scale=ColorScale.new(color_scale),
        normalize=normalize,
        log1p=log1p,
    )


def color_by_lid(
    n_neighbors: int,
    metric: str = "euclidean",
    color_scale: dict | None = None,
    remove_self: bool = True,
    eps: float = 1.0e-10,
    knn_params: dict | None = None,
) -> ColorByLid:
    """Create a Color by Local Intrinsic Dimensionality plotter."""
    from drnb.plot.scale import ColorScale
    from drnb.plot.scale.lid import ColorByLid

    return ColorByLid(
        n_neighbors=n_neighbors,
        metric=metric,
        scale=ColorScale.new(color_scale),
        remove_self=remove_self,
        eps=eps,
        knn_params=knn_params,
    )


def color_by_nbr_pres(
    n_neighbors: int,
    normalize: bool = True,
    color_scale: dict | None = None,
    metric: str = "euclidean",
) -> ColorByNbrPres:
    """Create a Color by Neighbor Preservation plotter."""
    from drnb.plot.scale import ColorScale
    from drnb.plot.scale.nbrpres import ColorByNbrPres

    return ColorByNbrPres(
        n_neighbors,
        normalize=normalize,
        scale=ColorScale.new(color_scale),
        metric=metric,
    )


def color_by_rte(
    n_triplets_per_point: int,
    normalize: bool = True,
    color_scale: dict | None = None,
    metric: str = "euclidean",
) -> ColorByRte:
    """Create a Color by Random Triplet Error plotter."""
    from drnb.plot.scale import ColorScale
    from drnb.plot.scale.rte import ColorByRte

    return ColorByRte(
        n_triplets_per_point=n_triplets_per_point,
        normalize=normalize,
        scale=ColorScale.new(color_scale),
        metric=metric,
    )


def diag_plots(metric: str = "euclidean") -> list[PlotterProtocol]:
    """Create some default diagnostic plots:

    - ColorByKo -- color by the k-occurrence. A measure of hubness. Bigger means an item is considered
    a neighbor by a larger number of other items in the dataset. The value of the k-occurrence can go
    between 0 and N.
    - ColorBySo -- color by the s-occurrence. Another measure of hubness. Bigger means an item appears
    as a neighbor of its own neighbors (rather than the whole dataset as measured by k-occurrence).
    - ColorByLid -- color by the local intrinsic dimension estimate, using the nearest neighbor approach
    of [Levina and Bickel](https://papers.nips.cc/paper_files/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html).
    - ColorByNbrPres -- color by neighbor preservation: of the k-nearest neighbors of each item in the
    2D output, how many of the high-dimensional neighbors are preserved? Normalized to a number between
    0 (no neighbors preserved) and 1 (all of them).
    - ColorByRte -- color by random triplet evaluation: The triangle distances between three randomly
    sampled points are evaluated in the low and high dimensions and the RTE is the proportion of those
    ordered distances where the ordering is the same in the high and low dimensions. 5 triplets are
    used for each item in the dataset.
    """
    return [
        color_by_ko(15, color_scale={"palette": "Spectral"}),
        color_by_so(15, color_scale={"palette": "Spectral"}),
        color_by_lid(15, metric=metric, color_scale={"palette": "Spectral"}),
        color_by_nbr_pres(15, color_scale={"palette": "Spectral"}, metric=metric),
        color_by_rte(5, color_scale={"palette": "Spectral"}, metric=metric),
    ]


def extra_plots(metric: str = "euclidean") -> list[ActionConfig]:
    """Create some extra diagnostic plots. These are not scatterplots of the embedded
    coordinates. Plots are:

    - `nnphist` -- a histogram of the nearest neighbor preservation values.
    - `rthist` -- a histogram of the random triplet preservation values.
    - `rpscatter` -- a scatter plot of embedded against ambient distances. Distances are
    sampled randomly with 5 distances sampled per point in the dataset.
    - `lidhist` -- a histogram of the local intrinsic dimensionality (using the
    Levina-Bickel method).
    """
    return [
        ("nnphist", {"metric": metric}),
        ("rthist", {"metric": metric}),
        ("rpscatter", {"metric": metric}),
        ("lidhist", {"metric": metric}),
    ]


def standard_metrics() -> list[ActionConfig]:
    """Create a list of standard evaluation metrics."""
    return [
        "rte",
        "rpc",
        ("nnp", {"n_neighbors": [15, 50, 150]}),
        # ("lp", dict(n_neighbors=[15, 50, 150])),
        # ("unnp", dict(n_neighbors=[15, 50, 150])),
        # ("soccur", dict(n_neighbors=[15, 50, 150])),
    ]


# Automatically adds usual eval and plotting
def standard_pipeline(
    method: EmbedConfig | str | list | tuple,
    *,
    params: dict | None = None,
    verbose: bool = False,
    extra_eval: str | list[str] | None = None,
    extra_plot: bool | dict | str | None = None,
) -> EmbedderPipeline:
    """Create a standard embedding pipeline, with default evaluation metrics and
    plots."""
    method = check_embed_method(method, params)

    if extra_plot is None:
        plot = True
    else:
        plot = extra_plot

    if extra_eval is None:
        extra_eval = []
    return create_pipeline(
        method=method,
        eval_metrics=standard_metrics() + extra_eval,
        plot=plot,
        verbose=verbose,
    )


def standard_eval(
    method: str | list | tuple,
    dataset: str,
    *,
    params: dict | None = None,
    verbose: bool = False,
    extra_eval: str | list[str] | None = None,
    extra_plot: bool | dict | str | None = None,
) -> list[EvalResult]:
    """Run a one-off standard pipeline and return the evaluation results."""
    return standard_pipeline(
        method,
        params=params,
        extra_eval=extra_eval,
        extra_plot=extra_plot,
        verbose=verbose,
    ).run(dataset)["evaluations"]


def embed(data, method, params=None):
    """Simple wrapper to run embedding on raw data"""
    return create_pipeline(method=embedder(method, params=params)).embedder.embed(data)
