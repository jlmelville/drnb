import numpy as np
import pacmap

from drnb.embed import create_embedder
from drnb.eval import get_triplets
from drnb.io import create_exporter, create_importer, read_dataxy, write_csv
from drnb.plot import create_plotter, sns_embed_plot

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


# PACMAP

PACMAP_DEFAULTS = dict(
    n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, verbose=False
)


def pacmapper(x, y=None, pacmap_kwargs=None, plot_kwargs=None):
    if y is None and isinstance(x, tuple) and len(x) == 2:
        y = x[1]
        x = x[0]

    if pacmap_kwargs is None:
        pacmap_kwargs = PACMAP_DEFAULTS
    if "init" in pacmap_kwargs:
        init = pacmap_kwargs["init"]
        del pacmap_kwargs["init"]
    else:
        init = "pca"
    pacmap_reducer = pacmap.PaCMAP(**pacmap_kwargs)
    print(x.to_numpy().shape)
    pacmap_embed = pacmap_reducer.fit_transform(x.to_numpy(), init=init)

    if plot_kwargs is None:
        plot_kwargs = {}
    if len(pacmap_embed.shape) == 3:
        for i in range(pacmap_embed.shape[0]):
            sns_embed_plot(pacmap_embed[i, :, :], y, **plot_kwargs)
    else:
        sns_embed_plot(pacmap_embed, y, **plot_kwargs)
    return pacmap_embed


def pacmap_data(
    name,
    pacmap_kwargs=None,
    plot_kwargs=None,
    export=False,
    export_dir="pacmap",
    export_triplets=False,
    seed=None,
    n_neighbors=None,
    n_iters=450,
    suffix=None,
    intermediates=None,
    repickle=False,
):
    x, y = read_dataxy(name, repickle=repickle)

    #    Using int seed seems to give horrible results with PaCMAP
    if pacmap_kwargs is None:
        pacmap_kwargs = PACMAP_DEFAULTS

    embedded = pacmapper(
        x,
        y=y,
        pacmap_kwargs=pacmap_kwargs.update(
            n_neighbors=n_neighbors, num_iters=n_iters, intermediate=intermediates
        ),
        plot_kwargs=plot_kwargs,
    )
    if export_triplets:
        triplets = get_triplets(x, seed=seed)
        write_csv(
            np.vstack(triplets), name=f"{name}{suffix}-triplets", sub_dir=export_dir
        )
    else:
        triplets = None
    # if intermediates is not None and intermediates:
    #     print(random_triplet_eval(x.to_numpy(), embedded[-1, :, :], triplets=triplets))
    # else:
    #     print(random_triplet_eval(x.to_numpy(), embedded, triplets=triplets))

    if suffix is None:
        suffix = export_dir
    if suffix[0] not in ("-", "_"):
        suffix = f"-{suffix}"

    if export:
        if intermediates is not None and intermediates:
            for i, inter in enumerate(intermediates):
                write_csv(
                    embedded[i, :, :],
                    name=f"{name}{suffix}-it{inter}",
                    sub_dir=export_dir,
                )
        else:
            write_csv(embedded, name=f"{name}{suffix}", sub_dir=export_dir)
    return embedded


def embed_data(
    name,
    method,
    x=None,
    y=None,
    import_kwargs=None,
    embed_kwargs=None,
    plot=True,
    plot_kwargs=None,
    export=False,
    export_kwargs=None,
):
    importer = create_importer(x, import_kwargs)
    exporter = create_exporter(method, export, export_kwargs)
    embedder = create_embedder(method, embed_kwargs)
    plotter = create_plotter(plot, plot_kwargs)

    x, y = importer.import_data(name, x, y)
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    embedded = embedder.embed(x)
    plotter.plot(embedded, y)
    exporter.export(name=name, coords=embedded)

    return embedded
