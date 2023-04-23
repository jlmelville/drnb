import pathlib
from dataclasses import dataclass, field
from typing import Any

import drnb.io as nbio
import drnb.io.dataset as dataio
import drnb.plot as nbplot
from drnb.embed import check_embed_method, embedder, get_embedder_name
from drnb.embed.factory import create_embedder
from drnb.eval import evaluate_embedding
from drnb.eval.factory import create_evaluators
from drnb.io.embed import create_embed_exporter
from drnb.log import log, log_verbosity
from drnb.neighbors import create_neighbors_request, find_candidate_neighbors_info
from drnb.triplets import create_triplets_request, find_triplet_files
from drnb.util import default_dict, default_list, dts_to_str, islisty


@dataclass
class EmbedderPipeline:
    embed_method_name: str
    # a way to refer to a specific parameterization of an embedding method,
    # e.g. the method might be umap, but you may want to refer to it as densvis
    embed_method_variant: str = ""
    importer: Any = dataio.DatasetImporter()
    embedder: Any = None
    evaluators: list = field(default_factory=list)
    plotters: list = default_list()
    exporter: Any = None
    verbose: bool = False
    pipeline_name: str = f"pipeline-{dts_to_str()}"

    def run_many(self, dataset_names, verbose=None):
        if verbose is None:
            verbose = self.verbose
        with log_verbosity(verbose):
            results = {}
            for dataset_name in dataset_names:
                log.info("Running %s", dataset_name)
                results[dataset_name] = self._run(dataset_name)
        return results

    def run(self, dataset_name, verbose=None):
        if verbose is None:
            verbose = self.verbose
        with log_verbosity(verbose):
            return self._run(dataset_name)

    def _run(self, dataset_name):
        ctx = EmbedContext(
            embed_method_name=self.embed_method_name,
            embed_method_variant=self.embed_method_variant,
            dataset_name=dataset_name,
            drnb_home=self.importer.drnb_home,
            data_sub_dir=self.importer.sub_dir,
            experiment_name=self.pipeline_name,
        )
        log.info("Getting dataset %s", ctx.dataset_name)
        x, y = self.importer.import_data(ctx.dataset_name)

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
            embedding_result = dict(coords=embedding_result)

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


@dataclass
class EmbedPipelineExporter:
    out_types: default_list()
    export_dict: dict = default_dict()

    def cache_data(self, requirers, embedding_result, ctx):
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

    def cache_triplets(self, require, embed_coords, ctx):
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
            dict(
                n_triplets_per_point=triplet_info.n_triplets_per_point,
                seed=triplet_info.seed,
                metric=require["metric"],
            )
        )
        triplet_output_paths = triplets_request.create_triplets(
            embed_coords,
            dataset_name=ctx.embed_triplets_name,
            triplet_dir=ctx.experiment_name,
        )

        if "triplets" not in self.export_dict:
            self.export_dict["triplets"] = []
        self.export_dict["triplets"].append(
            dict(
                request=triplets_request,
                paths=nbio.stringify_paths(triplet_output_paths),
            )
        )

    def cache_neighbors(self, require, embed_coords, ctx):
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
            dict(
                n_neighbors=require["n_neighbors"],
                metric=require["metric"],
            )
        )
        neighbors_output_paths = neighbors_request.create_neighbors(
            embed_coords,
            dataset_name=ctx.embed_nn_name,
            nbr_dir=ctx.experiment_name,
        )

        if "neighbors" not in self.export_dict:
            self.export_dict["neighbors"] = []
        self.export_dict["neighbors"].append(
            dict(
                request=neighbors_request,
                paths=nbio.stringify_paths(neighbors_output_paths),
            )
        )

    def export(self, embedding_result, ctx):
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


def create_exporter(export):
    if export is not None:
        if not islisty(export):
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
    method,
    data_config=None,
    plot=True,
    eval_metrics=None,
    export=None,
    verbose=False,
    embed_method_variant="",
):
    if data_config is None:
        data_config = {}
    importer = dataio.DatasetImporter(**data_config)
    # shut up pylint
    _embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotters = nbplot.create_plotters(plot)
    exporter = create_exporter(export)

    return EmbedderPipeline(
        embed_method_name=get_embedder_name(method),
        embed_method_variant=embed_method_variant,
        importer=importer,
        embedder=_embedder,
        evaluators=evaluators,
        plotters=plotters,
        exporter=exporter,
        verbose=verbose,
    )


@dataclass
class EmbedContext:
    dataset_name: str
    embed_method_name: str
    embed_method_variant: str = ""
    drnb_home: pathlib.Path = nbio.get_drnb_home()
    data_sub_dir: str = "data"
    nn_sub_dir: str = "nn"
    triplet_sub_dir: str = "triplets"
    experiment_name: str = None

    @property
    def embed_method_label(self):
        if self.embed_method_variant:
            return self.embed_method_variant
        return self.embed_method_name

    @property
    def embed_nn_name(self):
        return f"{self.dataset_name}-{self.embed_method_label}-nn"

    @property
    def embed_triplets_name(self):
        return f"{self.dataset_name}-{self.embed_method_label}-triplets"


def color_by_ko(n_neighbors, color_scale=None, normalize=True, log1p=False):
    return nbplot.ColorByKo(
        n_neighbors,
        scale=nbplot.ColorScale.new(color_scale),
        normalize=normalize,
        log1p=log1p,
    )


def color_by_so(n_neighbors, log1p=False, normalize=True, color_scale=None):
    return nbplot.ColorBySo(
        n_neighbors,
        scale=nbplot.ColorScale.new(color_scale),
        normalize=normalize,
        log1p=log1p,
    )


def color_by_nbr_pres(
    n_neighbors, normalize=True, color_scale=None, metric="euclidean"
):
    return nbplot.ColorByNbrPres(
        n_neighbors,
        normalize=normalize,
        scale=nbplot.ColorScale.new(color_scale),
        metric=metric,
    )


def color_by_rte(
    n_triplets_per_point, normalize=True, color_scale=None, metric="euclidean"
):
    return nbplot.ColorByRte(
        n_triplets_per_point=n_triplets_per_point,
        normalize=normalize,
        scale=nbplot.ColorScale.new(color_scale),
        metric=metric,
    )


def diag_plots(metric="euclidean"):
    return [
        color_by_ko(15, color_scale=dict(palette="Spectral")),
        color_by_so(15, color_scale=dict(palette="Spectral")),
        color_by_nbr_pres(15, color_scale=dict(palette="Spectral"), metric=metric),
        color_by_rte(5, color_scale=dict(palette="Spectral"), metric=metric),
    ]


def extra_plots(metric="euclidean"):
    return [
        ("nnphist", dict(metric=metric)),
        ("rthist", dict(metric=metric)),
        ("rpscatter", dict(metric=metric)),
    ]


def standard_metrics():
    return [
        "rte",
        "rpc",
        ("nnp", dict(n_neighbors=[15, 50, 150])),
        # ("unnp", dict(n_neighbors=[15, 50, 150])),
        # ("soccur", dict(n_neighbors=[15, 50, 150])),
    ]


# Automatically adds usual eval and plotting
def standard_pipeline(
    method, *, params=None, verbose=False, extra_eval=None, extra_plot=None
):
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


# Runs a one-off standard pipeline
def standard_eval(
    method,
    dataset,
    *,
    params=None,
    extra_eval=None,
    extra_plot=None,
    verbose=False,
):
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
