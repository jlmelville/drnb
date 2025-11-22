from typing import List, Literal, Tuple, cast

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import sklearn.decomposition
import umap
from numpy.typing import NDArray
from sklearn.utils import check_random_state

import drnb.neighbors as nbrs
import drnb.neighbors.random
from drnb.graph import umap_graph_binary
from drnb.log import log
from drnb.neighbors import n_connected_components
from drnb.neighbors.nbrinfo import NearestNeighbors
from drnb.neighbors.random import logn_neighbors


def tsne_scale_coords(coords: np.ndarray, target_std: float = 1e-4) -> np.ndarray:
    """Rescale coordinates to a fixed standard deviation, t-SNE style."""

    # Copied from the openTSNE initialization module to avoid dependency
    x = np.array(coords, copy=True)
    x /= np.std(x[:, 0]) / target_std

    return x


def scale_coords(
    coords: NDArray[np.float32], max_coord: float = 10.0
) -> NDArray[np.float32]:
    """Ensure that the maximum absolute value of the coordinates is `max_coord`."""
    expansion = max_coord / np.abs(coords).max()
    return (coords * expansion).astype(np.float32)


def noisy_scale_coords(
    coords: NDArray[np.float32],
    max_coord: float = 10.0,
    noise: float = 0.0001,
    seed=None,
) -> NDArray[np.float32]:
    """Scale the coordinates so that the maximum absolute value is `max_coord`
    and add noise."""
    return scale_coords(coords, max_coord=max_coord) + add_noise(
        coords, noise=noise, seed=seed
    )


def add_noise(
    coords: NDArray[np.float32], noise: float = 0.0001, seed: int | None = None
) -> NDArray[np.float32]:
    """Add Gaussian noise to the coordinates."""
    rng = np.random.default_rng(seed=seed)
    return coords + rng.normal(scale=noise, size=coords.shape).astype(np.float32)


def add_scaled_noise(
    coords: NDArray[np.float32], scale: float = 0.01, seed: int | None = None
) -> NDArray[np.float32]:
    """
    Add Gaussian noise to the coordinates using relative scaling. Matches openTSNE
    logic for the `jitter` function.
    """
    rng = np.random.default_rng(seed=seed)

    # standard deviation is based on the first dimension
    data_std = np.std(coords[:, 0])
    target_std = data_std * scale

    return coords + rng.normal(scale=target_std, size=coords.shape).astype(np.float32)


def spca(data: NDArray[np.float32], stdev: float = None) -> NDArray[np.float32]:
    """Initialize the embedding using PCA, scaling the coordinates by `stdev` divided by
    1e-4. If `stdev` is None, the coordinates are not scaled after PCA."""
    log.info("Initializing via openTSNE-style (scaled) PCA")
    coords = sklearn.decomposition.PCA(n_components=2).fit_transform(data)

    # rescale
    if stdev is not None:
        coords *= stdev / 1e-4

    # jitter
    coords = add_scaled_noise(coords)
    return coords


def pca(data: NDArray[np.float32], whiten: bool = False) -> NDArray[np.float32]:
    """Initialize the embedding using PCA. If `whiten` is True, the data is whitened."""
    log.info("Initializing via (unscaled) PCA")
    return sklearn.decomposition.PCA(n_components=2, whiten=whiten).fit_transform(data)


def umap_random_init(n: int, random_state: int = 42, max_coord: float = 10.0):
    """Initialize the embedding with random coordinates in the range [-`max_coord`,
    `max_coord`]."""
    log.info("Initializing via UMAP-style uniform random")
    return (
        check_random_state(random_state)
        .uniform(low=-max_coord, high=max_coord, size=(n, 2))
        .astype(np.float32)
    )


def umap_graph_spectral_init(
    x: np.ndarray | None = None,
    knn: List[np.ndarray] | Tuple[np.ndarray] | NearestNeighbors | None = None,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    global_neighbors: Literal["random", "mid"] | None = None,
    n_global_neighbors: int | None = None,
    op: Literal["intersection", "union", "difference", "linear"]
    | None = "intersection",
    global_weight: float = 0.1,
    random_state: int = 42,
    tsvdw: bool = False,
    tsvdw_tol: float = 1e-5,
    jitter: bool = True,
) -> np.ndarray:
    """Initialize the embedding using UMAP with a spectral layout. The graph is
    constructed using the kNN graph and optionally a global graph. The global graph is
    constructed using either random or mid-nearest neighbors. The global graph is then
    combined with the kNN graph using one of the following operations: intersection,
    union, difference, or linear combination. The resulting graph is then embedded using
    a spectral layout."""
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
            method_kwds={"random_state": random_state},
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
    x: np.ndarray | None = None,
    knn: List[np.ndarray] | Tuple[np.ndarray] | NearestNeighbors | None = None,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    global_neighbors: Literal["random", "mid"] | None = None,
    n_global_neighbors: int | None = None,
    global_weight: float = 0.1,
    random_state: int = 42,
    tsvdw: bool = False,
    tsvdw_tol: float = 1e-5,
    jitter: bool = True,
) -> np.ndarray:
    """Initialize the embedding using a binary graph with a spectral layout. The graph
    is constructed using the kNN graph and optionally a global graph with binary (0/1)
    edges. The global graph is constructed using either random or mid-nearest neighbors.
    The global graph is then combined with the kNN graph using a linear combination.
    The resulting graph is then embedded using a spectral layout."""
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
            method_kwds={"random_state": random_state},
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
    graph: scipy.sparse.coo_matrix,
    random_state: int = 42,
    tsvdw: bool = False,
    tsvdw_tol: float = 1e-5,
    jitter: bool = True,
) -> np.ndarray:
    """Embed the graph using a spectral layout. If `tsvdw` is True, the embedding is
    computed using a truncated SVD warm start."""
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


def scale1(x: np.ndarray) -> np.ndarray:
    """Scale the input vector to have an L2 norm of 1."""
    return x / np.linalg.norm(x)


def tsvd_warm_spectral(
    graph: scipy.sparse.coo_matrix,
    dim: int = 2,
    random_state: int = 42,
    tol: float = 1e-5,
    jitter: bool = True,
) -> np.ndarray:
    """Embed the graph using a spectral layout with a truncated SVD warm start. If
    `jitter` is True, the coordinates are jittered."""
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
    init: Literal["pca", "rand", "spectral", "gspectral"] | np.ndarray,
    nobs: int,
    random_state: int = 42,
    knn_idx: List[np.ndarray] | Tuple[np.ndarray] | NearestNeighbors | None = None,
    X: NDArray[np.float32] | None = None,
    init_scale: float | None = None,
):
    """Initialize the embedding using a standard method based on the `init` parameter.
    The initialization is scaled to have a maximum absolute value of `init_scale`."""
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
    elif init == "gspectral":
        if knn_idx is None:
            raise ValueError("Must provide knn_idx if init='gspectral'")
        Y = binary_graph_spectral_init(knn=knn_idx, global_neighbors="random", x=X)
    else:
        raise ValueError(f"Unknown init option '{init}'")
    if init_scale is not None:
        Y = scale_coords(Y, max_coord=init_scale)
    return Y.astype(np.float32, order="C")
