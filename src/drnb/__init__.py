import logging

from drnb.embed import get_embedder_name
from drnb.embed.factory import create_embedder
from drnb.eval import evaluate_embedding
from drnb.eval.factory import create_evaluators
from drnb.io import create_exporters, create_importer
from drnb.log import log
from drnb.plot import create_plotter
from drnb.preprocess import numpyfy

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

# helper method to create an embedder configuration
def embedder(name, params=None, **kwargs):
    return (name, kwargs | dict(params=params))


def embed_data(
    name,
    method,
    x=None,
    y=None,
    import_kwargs=None,
    plot=True,
    eval_metrics=None,
    export=False,
    verbose=False,
):
    old_log_level = log.level
    if verbose:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)

    importer = create_importer(x, import_kwargs)
    exporters = create_exporters(get_embedder_name(method), export)
    # zero reason to call embedder helper method here so don't care we are shadowing it
    # pylint: disable=redefined-outer-name
    embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotter = create_plotter(plot)

    log.info("Getting data")
    x, y = importer.import_data(name, x, y)
    x = numpyfy(x)

    log.info("Embedding")
    embedded = embedder.embed(x)

    log.info("Evaluating")
    evaluations = evaluate_embedding(evaluators, x, embedded)

    log.info("Plotting")
    plotter.plot(embedded, y)

    log.info("Exporting")
    for exporter in exporters:
        exporter.export(name=name, embedded=embedded)

    if not isinstance(embedded, dict):
        embedded = dict(coords=embedded)
        if evaluations:
            embedded["evaluations"] = evaluations
    log.setLevel(old_log_level)
    return embedded
