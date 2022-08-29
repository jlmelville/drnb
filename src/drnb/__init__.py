from drnb.embed import get_embedder_name
from drnb.embed.factory import create_embedder
from drnb.io import create_exporter, create_importer
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
    export=False,
):
    importer = create_importer(x, import_kwargs)
    exporter = create_exporter(get_embedder_name(method), export)
    embedder = create_embedder(method)
    plotter = create_plotter(plot)

    x, y = importer.import_data(name, x, y)
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    embedded = embedder.embed(x)
    plotter.plot(embedded, y)
    exporter.export(name=name, embedded=embedded)

    return embedded
