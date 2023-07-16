from typing import List, Optional, Tuple, Union, cast

import numpy as np
import openTSNE.initialization
import scipy.sparse.csgraph
import scipy.sparse.linalg
import sklearn.decomposition
import umap
from sklearn.utils import check_random_state

import drnb.neighbors as nbrs
import drnb.neighbors.random
from drnb.graph import umap_graph_binary
from drnb.log import log
from drnb.neighbors import n_connected_components
from drnb.neighbors.nbrinfo import NearestNeighbors
from drnb.neighbors.random import logn_neighbors


def scale_coords(coords, max_coord=10.0):
    expansion = max_coord / np.abs(coords).max()
    return (coords * expansion).astype(np.float32)


def noisy_scale_coords(coords, max_coord=10.0, noise=0.0001, seed=None):
    return scale_coords(coords, max_coord=max_coord) + add_noise(
        coords, noise=noise, seed=seed
    )


def add_noise(coords, noise=0.0001, seed=None):
    rng = np.random.default_rng(seed=seed)
    return coords + rng.normal(scale=noise, size=coords.shape).astype(np.float32)


def spca(data, stdev=None):
    log.info("Initializing via openTSNE (scaled) PCA")
    coords = openTSNE.initialization.pca(data)
    if stdev is not None:
        coords *= stdev / 1e-4
    return coords


def pca(data, whiten=False):
    log.info("Initializing via (unscaled) PCA")
    return sklearn.decomposition.PCA(n_components=2, whiten=whiten).fit_transform(data)


def umap_random_init(n, random_state=42, max_coord=10.0):
    log.info("Initializing via UMAP-style uniform random")
    return (
        check_random_state(random_state)
        .uniform(low=-max_coord, high=max_coord, size=(n, 2))
        .astype(np.float32)
    )


def umap_graph_spectral_init(
    x: Optional[np.ndarray] = None,
    knn: Optional[List[np.ndarray], Tuple[np.ndarray], NearestNeighbors] = None,
    metric="euclidean",
    n_neighbors=15,
    global_neighbors=None,
    n_global_neighbors=None,
    op="intersection",
    global_weight=0.1,
    random_state=42,
    tsvdw=False,
    tsvdw_tol=1e-5,
    jitter=True,
):
    if x is None and knn is None:
        raise ValueError("One of x or knn must be provided")
    if x is None and global_neighbors is not None:
        raise ValueError("x may not be None if global neighbors are used")
    if knn is None:
        knn = nbrs.calculate_neighbors(
            x,
            n_neighbors=n_neighbors,
            metric=metric,
            method="pynndescent",
            return_distance=True,
            method_kwds=dict(random_state=random_state),
        )
    if isinstance(knn, NearestNeighbors):
        knn.dist = cast(np.ndarray, knn.dist)
        knn = [knn.idx, knn.dist]
    knn = cast(List[np.ndarray], knn)

    knn_fss, _, _ = umap.umap_.fuzzy_simplicial_set(
        X=x,
        knn_indices=knn[0],
        knn_dists=knn[1],
        n_neighbors=knn[0].shape[1],
        random_state=None,
        metric=None,
        return_dists=None,
    )

    if global_neighbors is not None:
        x = cast(np.ndarray, x)
        if n_global_neighbors is None:
            n_global_neighbors = logn_neighbors(x.shape[0])

        if global_neighbors == "random":
            global_nn = drnb.neighbors.random.random_neighbors(
                x,
                n_neighbors=n_global_neighbors,
                distance=metric,
                random_state=random_state,
            )
        else:
            global_nn = drnb.neighbors.random.mid_near_neighbors(
                data=x,
                n_neighbors=n_global_neighbors,
                metric=metric,
                random_state=random_state,
            )

        global_fss, _, _ = umap.umap_.fuzzy_simplicial_set(
            X=x,
            knn_indices=global_nn.idx,
            knn_dists=global_nn.dist,
            n_neighbors=global_nn.idx.shape[1],
            random_state=None,
            metric=None,
            return_dists=None,
        )

        log.info(
            "Creating UMAP %s graph with %d %s neighbors with weight %.3g",
            op,
            n_global_neighbors,
            global_neighbors,
            global_weight,
        )

        if op == "intersection":
            graph = umap.umap_.general_simplicial_set_intersection(
                knn_fss, global_fss, weight=global_weight
            )
            graph = umap.umap_.reset_local_connectivity(graph, reset_local_metric=True)
        elif op == "union":
            graph = umap.umap_.general_simplicial_set_union(knn_fss, global_fss)
            graph = umap.umap_.reset_local_connectivity(graph, reset_local_metric=True)
        elif op == "difference":
            graph = umap.umap_.general_simplicial_set_intersection(
                knn_fss, global_fss, weight=global_weight, right_complement=True
            )
            # https://github.com/lmcinnes/umap/discussions/841
            # "resetting local connectivity there ... pretty much eliminated anything
            # __sub__ did, so I just didn't do it in that case."
            graph = umap.umap_.reset_local_connectivity(graph, reset_local_metric=False)
        elif op == "linear":
            graph = global_weight * global_fss + (1.0 - global_weight) * knn_fss
            graph = umap.umap_.reset_local_connectivity(graph, reset_local_metric=True)
        else:
            raise ValueError(f"Unknown set operation: '{op}'")
    else:
        graph = knn_fss

    nc = n_connected_components(graph)
    if nc > 1:
        log.warning("global-weighted graph has %d components", nc)

    return spectral_graph_embed(graph, random_state, tsvdw, tsvdw_tol, jitter)


def binary_graph_spectral_init(
    x=None,
    knn=None,
    metric="euclidean",
    n_neighbors=15,
    global_neighbors=None,
    n_global_neighbors=None,
    global_weight=0.1,
    random_state=42,
    tsvdw=False,
    tsvdw_tol=1e-5,
    jitter=True,
):
    if x is None and knn is None:
        raise ValueError("One of x or knn must be provided")
    if x is None and global_neighbors is not None:
        raise ValueError("x may not be None if global neighbors are used")
    if knn is None:
        nbr_data = nbrs.calculate_neighbors(
            x,
            n_neighbors=n_neighbors,
            metric=metric,
            method="pynndescent",
            return_distance=True,
            method_kwds=dict(random_state=random_state),
        )
        knn = [nbr_data.idx, nbr_data.dist]
    knn_graph = umap_graph_binary(knn)

    if global_neighbors is not None:
        x = cast(np.ndarray, x)
        if n_global_neighbors is None:
            n_global_neighbors = logn_neighbors(x.shape[0])

        log.info(
            "Using %d %s neighbors with edge weight %f",
            n_global_neighbors,
            global_neighbors,
            global_weight,
        )
        if global_neighbors == "random":
            global_nn = drnb.neighbors.random.random_neighbors(
                x,
                n_neighbors=n_global_neighbors,
                distance=metric,
                random_state=random_state,
            )
        else:
            global_nn = drnb.neighbors.random.mid_near_neighbors(
                data=x,
                n_neighbors=n_global_neighbors,
                metric=metric,
                random_state=random_state,
            )
        global_graph = umap_graph_binary(global_nn, edge_weight=global_weight)
        knn_graph += global_graph

    nc = n_connected_components(knn_graph)
    if nc > 1:
        log.warning("UMAP graph has %d components", nc)

    return spectral_graph_embed(knn_graph, random_state, tsvdw, tsvdw_tol, jitter)


def spectral_graph_embed(
    graph, random_state=42, tsvdw=False, tsvdw_tol=1e-5, jitter=True
):
    random_state = check_random_state(random_state)
    if tsvdw:
        log.info("Using tsvdw solver")
        return tsvd_warm_spectral(
            graph=graph,
            dim=2,
            random_state=random_state,
            tol=tsvdw_tol,
            jitter=jitter,
        )
    init = umap.umap_.spectral_layout(
        data=None,
        graph=graph,
        dim=2,
        random_state=random_state,
    )
    if jitter:
        return noisy_scale_coords(
            init, seed=random_state.randint(np.iinfo(np.uint32).max)
        )
    return init


def scale1(x):
    return x / np.linalg.norm(x)


def tsvd_warm_spectral(
    graph,
    dim=2,
    random_state: Union[int, np.random.RandomState] = 42,
    tol=1e-5,
    jitter=True,
):
    n_components, _ = scipy.sparse.csgraph.connected_components(graph)
    if n_components > 1:
        raise ValueError("Multiple components detected")

    diag_data = np.asarray(graph.sum(axis=0))
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    # actually a shifted Lsym: I - Lsym
    L = D * graph * D

    k = dim + 1

    tsvd = sklearn.decomposition.TruncatedSVD(
        n_components=k, random_state=random_state, algorithm="arpack", tol=1e-2
    )
    guess = tsvd.fit_transform(L)
    guess[:, 0] = np.sqrt(scale1(diag_data[0]))

    Eye = scipy.sparse.identity(graph.shape[0], dtype=np.float64)  # type: ignore
    eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
        Eye - L,
        guess,
        largest=False,
        tol=tol,
        maxiter=graph.shape[0] * 5,
    )

    order = np.argsort(eigenvalues)[0:k]
    init = eigenvectors[:, order[1:]]
    if jitter:
        return noisy_scale_coords(
            init, seed=check_random_state(random_state).randint(np.iinfo(np.uint32).max)
        )
    return init


def standard_neighbor_init(
    init, nobs, random_state=42, knn_idx=None, X=None, init_scale=None
):
    # basic init options that can work for any with anything with access to the knn
    if isinstance(init, np.ndarray):
        if init.shape != (nobs, 2):
            raise ValueError("Initialization array has incorrect shape")
        log.info("Using pre-supplied initialization coordinates")
        Y = init
    elif init == "pca":
        if X is None:
            raise ValueError("Must provide X if init='pca'")
        Y = pca(X)
    elif init == "rand":
        Y = umap_random_init(nobs, random_state)
    elif init == "spectral":
        if knn_idx is None:
            raise ValueError("Must provide knn_idx if init='spectral'")
        Y = binary_graph_spectral_init(knn=knn_idx)
    else:
        raise ValueError(f"Unknown init option '{init}'")
    if init_scale is not None:
        Y = scale_coords(Y, max_coord=init_scale)
    return Y.astype(np.float32, order="C")
