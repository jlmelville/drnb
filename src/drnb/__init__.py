from drnb.embed import get_embedder_name
from drnb.embed.factory import create_embedder
from drnb.eval import evaluate_embedding
from drnb.eval.factory import create_evaluators
from drnb.io import create_exporter, create_importer, numpyfy
from drnb.plot import create_plotter

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


def embed_data(
    name,
    method,
    x=None,
    y=None,
    import_kwargs=None,
    plot=True,
    eval_metrics=None,
    export=False,
):
    importer = create_importer(x, import_kwargs)
    exporter = create_exporter(get_embedder_name(method), export)
    embedder = create_embedder(method)
    evaluators = create_evaluators(eval_metrics)
    plotter = create_plotter(plot)

    x, y = importer.import_data(name, x, y)
    x = numpyfy(x)
    embedded = embedder.embed(x)
    evaluations = evaluate_embedding(evaluators, x, embedded)

    plotter.plot(embedded, y)
    exporter.export(name=name, embedded=embedded)

    if not isinstance(embedded, dict):
        embedded = dict(coords=embedded)
        if evaluations:
            embedded["_evaluations"] = evaluations
    return embedded
