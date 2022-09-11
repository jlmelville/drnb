import pathlib
from dataclasses import dataclass, field
from typing import Any

import drnb.io as nbio
import drnb.io.dataset as dataio
import drnb.io.embed as embedio
import drnb.plot as nbplot
from drnb.embed import get_embedder_name
from drnb.embed.factory import create_embedder
from drnb.eval import evaluate_embedding
from drnb.eval.factory import create_evaluators
from drnb.log import log, log_verbosity


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
            data_path=self.importer.data_path,
            data_sub_dir=self.importer.sub_dir,
        )
        log.info("Getting dataset %s", name)
        x, y = self.importer.import_data(name)

        log.info("Embedding")
        embedded = self.embedder.embed(x, ctx=ctx)

        log.info("Evaluating")
        evaluations = evaluate_embedding(self.evaluators, x, embedded, ctx=ctx)

        log.info("Plotting")
        self.plotter.plot(embedded, y, ctx=ctx)

        log.info("Exporting")
        for exporter in self.exporters:
            exporter.export(name=name, embedded=embedded)

        if not isinstance(embedded, dict):
            embedded = dict(coords=embedded)
            if evaluations:
                embedded["evaluations"] = evaluations
        return embedded


def create_pipeline(
    method,
    data_config=None,
    plot=True,
    eval_metrics=None,
    export=False,
    verbose=False,
):
    if data_config is None:
        data_config = {}
    importer = dataio.DatasetImporter(**data_config)
    embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotter = nbplot.create_plotter(plot)
    exporters = embedio.create_embed_exporters(get_embedder_name(method), export)

    return EmbedderPipeline(
        importer=importer,
        embedder=embedder,
        evaluators=evaluators,
        plotter=plotter,
        exporters=exporters,
        verbose=verbose,
    )


@dataclass
class DatasetContext:
    name: str
    data_path: pathlib.Path = nbio.DATA_ROOT
    data_sub_dir: str = "xy"
    nn_sub_dir: str = "nn"
    triplet_sub_dir: str = "triplets"
