import numpy as np
import openTSNE.initialization
import scipy.sparse.csgraph
import sklearn.decomposition
import umap
from sklearn.utils import check_random_state

import drnb.neighbors as nbrs
import drnb.neighbors.random
from drnb.graph import umap_graph_binary
from drnb.log import log
from drnb.neighbors import n_connected_components


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


def spca(data):
    log.info("Initializing via openTSNE (scaled) PCA")
    return openTSNE.initialization.pca(data)


def gspectral(
    x,
    knn,
    metric="euclidean",
    op="intersection",
    weight=0.2,
    random_state=42,
    spectral_algorithm="umap",
    global_neighbors="random",
):
    if global_neighbors == "random":
        global_nn = drnb.neighbors.random.random_neighbors(
            x, distance=metric, random_state=random_state
        )
    else:
        global_nn = drnb.neighbors.random.mid_near_neighbors(
            data=x,
            n_neighbors=drnb.neighbors.random.logn_neighbors(x),
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
    )

    knn_fss, _, _ = umap.umap_.fuzzy_simplicial_set(
        X=x,
        knn_indices=knn[0],
        knn_dists=knn[1],
        n_neighbors=knn[0].shape[1],
        random_state=None,
        metric=None,
    )

    log.info(
        "Creating random weighted graph with random n_neighbors = %d",
        global_nn.idx.shape[1],
    )

    if op == "intersection":
        log.info("Combining sets by intersection with weight %.2f", weight)
        result = umap.umap_.general_simplicial_set_intersection(
            knn_fss, global_fss, weight=weight
        )
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=True)
    elif op == "union":
        log.info("Combining sets by union")
        result = umap.umap_.general_simplicial_set_union(knn_fss, global_fss)
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=True)
    elif op == "difference":
        log.info("Combining sets by difference with weight %.2f", weight)
        result = umap.umap_.general_simplicial_set_intersection(
            knn_fss, global_fss, weight=weight, right_complement=True
        )
        # https://github.com/lmcinnes/umap/discussions/841
        # "resetting local connectivity there ... pretty much eliminated anything
        # __sub__ did, so I just didn't do it in that case."
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=False)
    elif op == "linear":
        result = weight * global_fss + (1.0 - weight) * knn_fss
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=True)
    else:
        raise ValueError(f"Unknown set operation: '{op}'")

    nc = scipy.sparse.csgraph.connected_components(result)[0]
    if nc > 1:
        log.warning("global-weighted graph has %d components", nc)

    return spectral_graph_embed(
        result, random_state, tsvdw=spectral_algorithm == "umap", jitter=True
    )


def binary_graph_spectral_init(
    x,
    knn=None,
    metric="euclidean",
    n_neighbors=15,
    random_state=42,
    tsvdw=False,
    tsvdw_tol=1e-5,
    jitter=True,
):
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
    knn_fss = umap_graph_binary(knn)

    nc = n_connected_components(knn_fss)
    if nc > 1:
        log.warning("UMAP graph has %d components", nc)

    return spectral_graph_embed(knn_fss, random_state, tsvdw, tsvdw_tol, jitter)


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


def tsvd_warm_spectral(graph, dim=2, random_state=42, tol=1e-5, jitter=True):
    n_components, _ = scipy.sparse.csgraph.connected_components(graph)
    if n_components > 1:
        raise ValueError("Multiple components detected")

    diag_data = np.asarray(graph.sum(axis=0))
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = D * graph * D

    k = dim + 1

    tsvd = sklearn.decomposition.TruncatedSVD(
        n_components=k, random_state=random_state, algorithm="arpack", tol=1e-2
    )
    guess = tsvd.fit_transform(L)
    guess[:, 0] = np.sqrt(scale1(diag_data[0]))

    Eye = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
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
