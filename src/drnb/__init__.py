import numpy as np
import pacmap
import pymde
import sklearn
import torch
import trimap
import umap

from drnb.embed import create_embedder
from drnb.eval import get_triplets, random_triplet_eval
from drnb.io import (
    create_exporter,
    create_importer,
    export_coords,
    get_xy,
    get_xy_data,
    read_dataxy,
    write_csv,
)
from drnb.plot import create_plotter, plot_embedded, sns_embed_plot
from drnb.preprocess import center

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


# UMAP


def umapper(
    x,
    y=None,
    n_neighbors=15,
    init="spectral",
    densmap=False,
    plot_kwargs=None,
    seed=None,
):
    if y is None and isinstance(x, tuple) and len(x) == 2:
        y = x[1]
        x = x[0]

    umap_kwargs = {}
    if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
        umap_kwargs["metric"] = "precomputed"
    else:
        if hasattr(x, "to_numpy"):
            x = x.to_numpy()

    embedder = umap.UMAP(
        random_state=seed,
        n_neighbors=n_neighbors,
        init=init,
        densmap=densmap,
        output_dens=densmap,
        **umap_kwargs,
    )
    embedded = embedder.fit_transform(x)
    if isinstance(embedded, tuple):
        coords = embedded[0]
    else:
        coords = embedded
    if plot_kwargs is None:
        plot_kwargs = {}
    sns_embed_plot(coords, y, **plot_kwargs)
    return embedded


def umap_data(
    name,
    plot_kwargs=None,
    export=False,
    export_dir="umap",
    seed=None,
    repickle=False,
    suffix=None,
    init="spectral",
    densmap=False,
    n_neighbors=15,
    x=None,
    y=None,
):
    if x is None:
        x, y = read_dataxy(name, repickle=repickle)
    if y is None:
        y = range(x.shape[0])

    embedded = umapper(
        x,
        y=y,
        plot_kwargs=plot_kwargs,
        init=init,
        densmap=densmap,
        seed=seed,
        n_neighbors=n_neighbors,
    )
    if isinstance(embedded, tuple):
        coords = embedded[0]
    else:
        coords = embedded

    if export:
        if suffix is None:
            suffix = export_dir
        if not suffix[0] in ("-", "_"):
            suffix = f"-{suffix}"
        write_csv(coords, name=f"{name}{suffix}", sub_dir=export_dir)
        if densmap:
            write_csv(embedded[1], name=f"{name}{suffix}-ro", sub_dir=export_dir)
            write_csv(embedded[2], name=f"{name}{suffix}-re", sub_dir=export_dir)
    return embedded


# DENSMAP


def densmap_data(
    name,
    plot_kwargs=None,
    export=False,
    export_dir="densmap",
    seed=None,
    repickle=False,
    suffix=None,
    init="spectral",
    n_neighbors=30,
    x=None,
    y=None,
):
    return umap_data(
        name=name,
        plot_kwargs=plot_kwargs,
        export=export,
        export_dir=export_dir,
        seed=seed,
        repickle=repickle,
        suffix=suffix,
        init=init,
        densmap=True,
        n_neighbors=n_neighbors,
        x=x,
        y=y,
    )


# Truncated SVD


def tsvd_data(
    name,
    plot_kwargs=None,
    export=False,
    export_dir="tsvd",
    seed=None,
    repickle=False,
    suffix=None,
    x=None,
    y=None,
):
    x, y = get_xy_data(name, x, y, repickle=repickle)

    embedded = tsvd(
        x,
        y=y,
        plot_kwargs=plot_kwargs,
        seed=seed,
    )

    if export:
        export_coords(embedded, name, export_dir, suffix)

    return embedded


def tsvd(
    x,
    y=None,
    do_plot=True,
    plot_kwargs=None,
    seed=None,
    n_oversamples=10,
    n_iter=5,
    power_iteration_normalizer="auto",
):
    x, y = get_xy(x, y)

    x = center(x)

    embedder = sklearn.decomposition.TruncatedSVD(
        random_state=seed,
        n_components=2,
        n_oversamples=n_oversamples,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
    )
    embedded = embedder.fit_transform(x)

    if do_plot:
        plot_embedded(embedded, y, plot_kwargs)

    return embedded


# PCA


def pca_data(
    name,
    plot_kwargs=None,
    export=False,
    export_dir="pca",
    seed=None,
    repickle=False,
    suffix=None,
    x=None,
    y=None,
):

    x, y = get_xy_data(name, x, y, repickle=repickle)

    embedded = pca(
        x,
        y=y,
        plot_kwargs=plot_kwargs,
        seed=seed,
    )

    if export:
        export_coords(embedded, name, export_dir, suffix)

    return embedded


def pca(
    x,
    y=None,
    do_plot=True,
    plot_kwargs=None,
    seed=None,
):
    x, y = get_xy(x, y)

    embedder = sklearn.decomposition.PCA(
        random_state=seed,
        n_components=2,
    )
    embedded = embedder.fit_transform(x)

    if do_plot:
        plot_embedded(embedded, y, plot_kwargs)

    return embedded


# PyMDE


def pymde_nbrs(
    x,
    y=None,
    n_neighbors=None,
    do_plot=True,
    plot_kwargs=None,
    seed=None,
):
    x, y = get_xy(x, y)
    x = torch.from_numpy(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        pymde.seed(seed)
    embedder = pymde.preserve_neighbors(
        x, device=device, embedding_dim=2, n_neighbors=n_neighbors
    )
    embedded = embedder.embed().cpu().data.numpy()

    if do_plot:
        plot_embedded(embedded, y, plot_kwargs)

    return embedded


def pymde_data(
    name,
    plot_kwargs=None,
    export=False,
    export_dir="pymde",
    seed=None,
    n_neighbors=None,
    repickle=False,
    suffix=None,
    x=None,
    y=None,
):

    x, y = get_xy_data(name, x, y, repickle=repickle)

    embedded = pymde_nbrs(
        x,
        y=y,
        n_neighbors=n_neighbors,
        plot_kwargs=plot_kwargs,
        seed=seed,
    )

    if export:
        export_coords(embedded, name, export_dir, suffix)

    return embedded


# TriMAP


def trimap_data(
    name,
    plot_kwargs=None,
    export=False,
    export_dir="pca",
    repickle=False,
    suffix=None,
    x=None,
    y=None,
):

    x, y = get_xy_data(name, x, y, repickle=repickle)

    embedded = trimap_embed(x, y=y, plot_kwargs=plot_kwargs)

    if export:
        export_coords(embedded, name, export_dir, suffix)

    return embedded


def trimap_embed(
    x,
    y=None,
    do_plot=True,
    plot_kwargs=None,
):
    x, y = get_xy(x, y)

    embedder = trimap.TRIMAP(
        n_dims=2,
    )
    embedded = embedder.fit_transform(x)

    if do_plot:
        plot_embedded(embedded, y, plot_kwargs)

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
