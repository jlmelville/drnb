import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import drnb.io as nbio
import drnb.plot as nbplot
from drnb.embed import get_embedder_name
from drnb.embed.factory import create_embedder
from drnb.eval import evaluate_embedding
from drnb.eval.factory import create_evaluators
from drnb.log import log


@dataclass
class Pipeline:
    importer: Any = nbio.DatasetImporter()
    embedder: Any = None
    evaluators: list = field(default_factory=list)
    plotter: Any = nbplot.NoPlotter()
    exporters: list = field(default_factory=list)
    verbose: bool = False

    def run(self, name):
        if self.verbose:
            log.setLevel(logging.INFO)
        else:
            log.setLevel(logging.WARNING)

        ctx = DatasetContext(name=name)

        log.info("Getting dataset %s", name)
        x, y = self.importer.import_data(name)

        log.info("Embedding")
        embedded = self.embedder.embed(x, ctx=ctx)

        log.info("Evaluating")
        evaluations = evaluate_embedding(self.evaluators, x, embedded, ctx=ctx)

        log.info("Plotting")
        self.plotter.plot(embedded, y)

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
    plot=True,
    eval_metrics=None,
    export=False,
    verbose=False,
):
    embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotter = nbplot.create_plotter(plot)
    exporters = nbio.create_exporters(get_embedder_name(method), export)

    return Pipeline(
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
