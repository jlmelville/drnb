import pathlib
from dataclasses import dataclass, field
from typing import Any

import drnb.io as nbio
import drnb.io.dataset as dataio
import drnb.plot as nbplot
from drnb.embed import get_embedder_name
from drnb.embed.factory import create_embedder
from drnb.eval import evaluate_embedding
from drnb.eval.factory import create_evaluators
from drnb.io.embed import create_embed_exporter
from drnb.log import log, log_verbosity
from drnb.util import dts_to_str, islisty


@dataclass
class EmbedderPipeline:
    embed_method_name: str
    # a way to refer to a specific parameterization of an embedding method,
    # e.g. the method might be umap, but you may want to refer to it as densvis
    embed_method_label: str = ""
    importer: Any = dataio.DatasetImporter()
    embedder: Any = None
    evaluators: list = field(default_factory=list)
    plotter: Any = nbplot.NoPlotter()
    exporter: Any = None
    verbose: bool = False

    def run(self, dataset_name, experiment=None, verbose=None):
        if verbose is None:
            verbose = self.verbose
        with log_verbosity(verbose):
            return self._run(dataset_name, experiment=experiment)

    def _run(self, dataset_name, experiment):
        if experiment is None:
            experiment = f"experiment-{dts_to_str()}"
            log.info("Using experiment name: %s", experiment)

        ctx = EmbedContext(
            embed_method_name=self.embed_method_name,
            embed_method_label=self.embed_method_label,
            dataset_name=dataset_name,
            drnb_home=self.importer.drnb_home,
            data_sub_dir=self.importer.sub_dir,
            experiment_name=experiment,
        )
        log.info("Getting dataset %s", ctx.dataset_name)
        x, y = self.importer.import_data(ctx.dataset_name)

        log.info("Embedding")
        embedded = self.embedder.embed(x, ctx=ctx)

        log.info("Evaluating")
        evaluations = evaluate_embedding(self.evaluators, x, embedded, ctx=ctx)

        if not isinstance(embedded, dict):
            embedded = dict(coords=embedded)
            if evaluations:
                embedded["evaluations"] = evaluations

        if self.exporter is not None:
            log.info("Exporting")
            self.exporter.export(embedding_result=embedded, ctx=ctx)

        log.info("Plotting")
        self.plotter.plot(embedded, data=x, y=y, ctx=ctx)

        return embedded


@dataclass
class EmbedPipelineExporter:
    out_types: field(default_factory=list)

    def export(self, embedding_result, ctx):
        if self.out_types is not None:
            if ctx.embed_method_label:
                embed_method_label = ctx.embed_method_label
            else:
                embed_method_label = ctx.embed_method_name
            exporters = create_embed_exporter(
                embed_method_label=embed_method_label,
                out_type=self.out_types,
                sub_dir=ctx.experiment_name,
                suffix=None,
                create_sub_dir=True,
                drnb_home=ctx.drnb_home,
                verbose=True,
            )
            for exporter in exporters:
                exporter.export(
                    name=ctx.dataset_name, embedded=embedding_result["coords"]
                )
        if "evaluations" in embedding_result:
            nbio.write_json(
                embedding_result["evaluations"],
                name=ctx.dataset_name,
                suffix=[embed_method_label, "evaluations"],
                drnb_home=ctx.drnb_home,
                sub_dir=ctx.experiment_name,
                create_sub_dir=True,
                verbose=True,
            )


# helper method to create an embedder configuration
def embedder(name, params=None, **kwargs):
    return (name, kwargs | dict(params=params))


def create_pipeline(
    method,
    data_config=None,
    plot=True,
    eval_metrics=None,
    export=None,
    verbose=False,
    embed_method_label="",
):
    if data_config is None:
        data_config = {}
    importer = dataio.DatasetImporter(**data_config)
    # shut up pylint
    _embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotter = nbplot.create_plotter(plot)
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
        exporter = EmbedPipelineExporter(out_types=export)
    else:
        exporter = None

    return EmbedderPipeline(
        embed_method_name=get_embedder_name(method),
        embed_method_label=embed_method_label,
        importer=importer,
        embedder=_embedder,
        evaluators=evaluators,
        plotter=plotter,
        exporter=exporter,
        verbose=verbose,
    )


@dataclass
class EmbedContext:
    dataset_name: str
    embed_method_name: str
    embed_method_label: str = ""
    drnb_home: pathlib.Path = nbio.get_drnb_home()
    data_sub_dir: str = "data"
    nn_sub_dir: str = "nn"
    triplet_sub_dir: str = "triplets"
    experiment_name: str = None


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
        n_triplets_per_point,
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
