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
from drnb.util import get_method_and_args


@dataclass
class EmbedderPipeline:
    importer: Any = dataio.DatasetImporter()
    embedder: Any = None
    evaluators: list = field(default_factory=list)
    plotter: Any = nbplot.NoPlotter()
    exporters: list = field(default_factory=list)
    verbose: bool = False

    def run(self, name, verbose=None):
        if verbose is None:
            verbose = self.verbose
        with log_verbosity(verbose):
            return self._run(name)

    def _run(self, name):
        ctx = DatasetContext(
            name=name,
            drnb_home=self.importer.drnb_home,
            data_sub_dir=self.importer.sub_dir,
        )
        log.info("Getting dataset %s", name)
        x, y = self.importer.import_data(name)

        log.info("Embedding")
        embedded = self.embedder.embed(x, ctx=ctx)

        log.info("Evaluating")
        evaluations = evaluate_embedding(self.evaluators, x, embedded, ctx=ctx)

        log.info("Plotting")
        self.plotter.plot(embedded, data=x, y=y, ctx=ctx)

        if self.exporters is not None:
            log.info("Exporting")
            for exporter in self.exporters:
                exporter.export(name=name, embedded=embedded)

        if not isinstance(embedded, dict):
            embedded = dict(coords=embedded)
            if evaluations:
                embedded["evaluations"] = evaluations
        return embedded


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
):
    if data_config is None:
        data_config = {}
    importer = dataio.DatasetImporter(**data_config)
    # shut up pylint
    _embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotter = nbplot.create_plotter(plot)
    if export is not None:
        out_type, export_kwargs = get_method_and_args(export)
        export_kwargs["out_type"] = out_type
        export_kwargs["embed_method"] = get_embedder_name(method)
        if "verbose" not in export_kwargs:
            export_kwargs["verbose"] = verbose
        exporters = create_embed_exporter(**export_kwargs)
    else:
        exporters = None

    return EmbedderPipeline(
        importer=importer,
        embedder=_embedder,
        evaluators=evaluators,
        plotter=plotter,
        exporters=exporters,
        verbose=verbose,
    )


@dataclass
class DatasetContext:
    name: str
    drnb_home: pathlib.Path = nbio.get_drnb_home()
    data_sub_dir: str = "data"
    nn_sub_dir: str = "nn"
    triplet_sub_dir: str = "triplets"


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


def color_by_nbr_pres(n_neighbors, normalize=True, color_scale=None):
    return nbplot.ColorByNbrPres(
        n_neighbors, normalize=normalize, scale=nbplot.ColorScale.new(color_scale)
    )


def color_by_rte(n_triplets_per_point, normalize=True, color_scale=None):
    return nbplot.ColorByRte(
        n_triplets_per_point,
        normalize=normalize,
        scale=nbplot.ColorScale.new(color_scale),
    )


def diag_plots():
    return [
        color_by_ko(15, color_scale=dict(palette="Spectral")),
        color_by_so(15, color_scale=dict(palette="Spectral")),
        color_by_nbr_pres(15, color_scale=dict(palette="Spectral")),
        color_by_rte(5, color_scale=dict(palette="Spectral")),
    ]
