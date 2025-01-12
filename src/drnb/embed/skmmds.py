from dataclasses import dataclass
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numba
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from umap.utils import tau_rand_int

import drnb.embed
import drnb.embed.base
import drnb.neighbors as nbrs
import drnb.neighbors.random
from drnb.embed.context import EmbedContext
from drnb.embed.mixins import InitMixin, KNNMixin
from drnb.embed.umap.utils import euclidean
from drnb.log import log
from drnb.neighbors import calculate_exact_neighbors
from drnb.neighbors.distances import neighbor_distances
from drnb.neighbors.hubness import nn_to_sparse
from drnb.neighbors.random import random_neighbors, sort_neighbors
from drnb.optim import create_opt
from drnb.preprocess import pca as pca_reduce
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.types import EmbedResult
from drnb.yinit import standard_neighbor_init


@numba.njit(fastmath=True)
def l2dist(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff
    return np.sqrt(result)


def embedded_nbr_dist(
    data: np.ndarray,
    nbr_idx: np.ndarray,
    n_components: int = 2,
    algorithm: Literal["arpack", "randomized"] = "arpack",
    random_state: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Calculate the distances between embedded neighbors, and the variance explained
    by the embedding, using a truncated SVD on the neighbors of each point.
    """
    if data.shape[0] != nbr_idx.shape[0]:
        raise ValueError("Incompatible nbr idx and data shape")
    embedded_dist = np.zeros(nbr_idx.shape, np.float32)
    varex = np.zeros(nbr_idx.shape[0])
    tsvd = TruncatedSVD(
        n_components=n_components, random_state=random_state, algorithm=algorithm
    )
    for i in range(nbr_idx.shape[0]):
        # nbr_idx does *not* contain i itself, so we need to add it the SVD
        embedded_nbrs = tsvd.fit_transform(data[np.r_[i, nbr_idx[i]], :])
        varex[i] = np.sum(tsvd.explained_variance_ratio_)

        # calculate the distances from embedded i to the other embedded points
        for j in range(1, embedded_nbrs.shape[0]):
            embedded_dist[i, j - 1] = l2dist(embedded_nbrs[0, :], embedded_nbrs[j, :])
    return sort_neighbors(nbr_idx, embedded_dist), varex


def center(Y: np.ndarray):
    """Center the data in Y."""
    ndim = Y.shape[1]
    for d in range(ndim):
        Yd = Y[:, d]
        Y[:, d] = Yd - np.mean(Yd)


def setup_opt(
    nobs, learning_rate=1.0, opt="adam", optargs: dict | None = None
) -> drnb.optim.OptimizerProtocol:
    """Setup the optimizer."""
    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate
    return create_opt(nobs, opt, optargs)


def knn_pca(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    n_components: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform PCA reduction on the data if requested, and update the nearest neighbor
    data accordingly."""
    if n_components is not None:
        if np.min(X.shape) > n_components:
            X = pca_reduce(X, n_components=n_components)
            n_neighbors = knn_idx.shape[1]
            log.info("Calculating new nearest neighbor data after PCA reduction")
            pca_nn = nbrs.calculate_exact_neighbors(
                data=X, metric="euclidean", n_neighbors=n_neighbors + 1
            )
            knn_idx, knn_dist = pca_nn.idx, pca_nn.dist
        else:
            log.info(
                "Requested PCA with n_components=%d is not possible, skipping",
                n_components,
            )
    return X, knn_idx[:, 1:], knn_dist[:, 1:]


def knn_init(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"],
    random_state: int,
    init_scale: float | Literal["knn"] | None,
) -> np.ndarray:
    """Initialize the embedding. If init is a string, use the corresponding method to
    initialize the embedding. If it is an array, use it as the initial embedding. If
    init_scale = "knn", use the mean of the knn distances as the scale for the
    initialization."""
    if isinstance(init_scale, str) and init_scale == "knn":
        init_scale = np.mean(knn_dist)
        log.info("Using knn distance mean for init_scale: %f", init_scale)
    return standard_neighbor_init(
        init,
        nobs=knn_idx.shape[0],
        random_state=random_state,
        knn_idx=knn_idx,
        X=X,
        init_scale=init_scale,
    )


def symmetrize_nn(
    nbr_idx: np.ndarray,
    nbr_dist: np.ndarray,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
) -> csr_matrix:
    """Symmetrize the nearest neighbor graph."""
    dmat = nn_to_sparse([nbr_idx, nbr_dist], symmetrize=symmetrize)
    dmat.eliminate_zeros()
    return dmat.tocsr()


def symmetrized_random_neighbors(
    X: np.ndarray,
    n_samples: int,
    nnbr_dist_strategy: Literal["localmean", "mean", "raw"] | None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
) -> csr_matrix:
    """Create a symmetrized random non-neighbors graph. If nnbr_dist_strategy is
    "localmean", the distance to all neighbors is set as the mean of the neighbors of
    that item. If it is "mean", the global mean distance is used. If it is "raw", the
    raw distances are retained. This allows for experimentation around how accurate
    non-neighbor distances need to be."""
    rnbrs = random_neighbors(
        X, n_neighbors=n_samples, distance="euclidean", random_state=42
    )
    rnbrs_idx = rnbrs.idx
    rnbrs_dist = rnbrs.dist
    if nnbr_dist_strategy is not None:
        if nnbr_dist_strategy == "localmean":
            rnbrs_dist = np.mean(rnbrs_dist, axis=1)[:, np.newaxis].repeat(
                rnbrs_dist.shape[1], axis=1
            )
        elif nnbr_dist_strategy == "mean":
            rnbrs_dist = np.full_like(rnbrs_dist, np.mean(rnbrs_dist))

    return symmetrize_nn(rnbrs_idx, rnbrs_dist, symmetrize=symmetrize)


def random_neighbors_init(
    X: np.ndarray,
    n_random_neighbors: int,
    nnbr_dist_strategy: Literal["localmean", "mean", "raw"] | None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
) -> csr_matrix:
    """Create a random non-neighbors graph. If nnbr_dist_strategy is "localmean", the
    distance to all neighbors is set as the mean of the neighbors of that item. If it is
    "mean", the global mean distance is used. If it is "raw", the raw distances are
    retained. If n_random_neighbors is 0, return an empty graph."""
    if n_random_neighbors == 0:
        log.info("Using dynamically allocated random neighbors only")
        nobs = X.shape[0]
        return csr_matrix((nobs, nobs))
    log.info("Using %d pre-allocated random non-neighbors graph", n_random_neighbors)
    return symmetrized_random_neighbors(
        X, n_random_neighbors, nnbr_dist_strategy, symmetrize=symmetrize
    )


def lcmmds(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    n_epochs: int = 500,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "spectral",
    n_samples: int = 5,
    sample_strategy: Literal["unif", "inc", "dec"] | None = None,
    random_state: int = 42,
    learning_rate: float = 1.0,
    opt: str = "adam",
    optargs: dict | None = None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    init_scale: float | Literal["knn"] | None = 10.0,
    pca: int | None = None,
    eps: float = 1e-10,
    n_random_neighbors: int = 0,
    nnbr_dist_strategy: Literal["dynamic", "fixed", "localmean", "mean"] = "dynamic",
    nbr_dist_strategy: Literal["localmean", "mean", "raw"] | None = None,
    tnbr: float = 1.0,
    tnnbr: float = 1.0,
) -> np.ndarray:
    """MMDS with a log-cosh loss function."""
    nobs = X.shape[0]

    opt = setup_opt(nobs, learning_rate, opt, optargs)

    rng_state = setup_rngn(nobs, random_state)

    X, knn_idx, knn_dist = knn_pca(X, knn_idx, knn_dist, n_components=pca)

    Y = knn_init(X, knn_idx, knn_dist, init, random_state, init_scale)

    nbr_graph = symmetrize_nn(knn_idx, knn_dist, symmetrize=symmetrize)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    rnbr_graph = random_neighbors_init(
        X, n_random_neighbors, nnbr_dist_strategy, symmetrize
    )

    if nbr_dist_strategy is None:
        nbr_dist_strategy = "raw"
    if nbr_dist_strategy == "localmean":
        nnz = np.diff(nbr_graph.indptr)
        row_means = np.array(nbr_graph.sum(axis=1)).flatten() / nnz
        nbr_graph.data = row_means[np.repeat(np.arange(nbr_graph.shape[0]), nnz)]
    elif nbr_dist_strategy == "mean":
        nbr_graph.data = np.full_like(nbr_graph.data, np.mean(nbr_graph.data))
    elif nbr_dist_strategy == "raw":
        pass
    else:
        raise ValueError(f"Unknown nbr_dist_strategy: {nbr_dist_strategy}")

    rng_state = setup_rngn(nobs, random_state)
    return _lcmmds(
        X, Y, n_epochs, opt, samples, rng_state, nbr_graph, rnbr_graph, eps, tnbr, tnnbr
    )


def _lcmmds(
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    nbr_graph: csr_matrix,
    rnbr_graph: csr_matrix,
    eps: float,
    tnbr: float,
    tnnbr: float,
) -> np.ndarray:
    nobs, ndim = Y.shape
    graph_j, graph_dist, ptr = nbr_graph.indices, nbr_graph.data, nbr_graph.indptr
    rgraph_j, rgraph_dist, rptr = rnbr_graph.indices, rnbr_graph.data, rnbr_graph.indptr

    n_rand_nbrs = np.diff(rptr)

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        _glcmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, tnbr, grads)
        _glcmmds_non_nbrs_fixed(
            Y, samples[n], rgraph_j, rgraph_dist, rptr, eps, tnnbr, grads
        )
        _glcmmds_non_nbrs(
            X, Y, n_rand_nbrs, samples[n], ptr, rng_state, eps, tnnbr, grads
        )

        Y = opt.opt(Y, grads, n, n_epochs)
        center(Y)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _glcmmds_nbrs(
    Y: np.ndarray,
    graph_j: np.ndarray,
    graph_dist: np.ndarray,
    ptr: np.ndarray,
    eps: float,
    t: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            j = graph_j[edge]
            rij = graph_dist[edge]
            Yj = Y[j]

            dij = euclidean(Yi, Yj)
            gc = np.exp(2.0 * t * (rij - dij))
            grad_coeff = (t * (1.0 - gc)) / ((1.0 + gc) * (dij + eps))

            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _glcmmds_non_nbrs(
    X: np.ndarray,
    Y: np.ndarray,
    n_rand_nbrs: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    eps: float,
    t: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    nobs_minus_1 = nobs - 1
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        n_nbr_edges = ptr[i + 1] - ptr[i]
        n_nnbr_edges_to_sample = (n_nbr_edges * n_samples) - n_rand_nbrs[i]
        if n_nnbr_edges_to_sample <= 0:
            continue
        n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
        Xi = X[i]
        Yi = Y[i]
        # number of edges in i is ptr[i + 1] - ptr[i]
        for _ in range(n_nnbr_edges_to_sample):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            Yj = Y[j]

            rij = euclidean(Xi, X[j])
            dij = euclidean(Yi, Yj)
            gc = np.exp(2.0 * t * (rij - dij))
            grad_coeff = (t * (1.0 - gc)) / ((1.0 + gc) * (dij + eps))

            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _glcmmds_non_nbrs_fixed(
    Y: np.ndarray,
    n_samples: int,
    graph_j: np.ndarray,
    graph_dist: np.ndarray,
    ptr: np.ndarray,
    eps: float,
    t: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            if edge > n_samples:
                break

            j = graph_j[edge]
            rij = graph_dist[edge]
            Yj = Y[j]

            dij = euclidean(Yi, Yj)

            gc = np.exp(2.0 * t * (rij - dij))
            grad_coeff = (t * (1.0 - gc)) / ((1.0 + gc) * (dij + eps))
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@dataclass
class Lcmmds(InitMixin, KNNMixin, drnb.embed.base.Embedder):
    """Log-cosh Metric MMDS"""

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.handle_precomputed_init(params)
        params = self.handle_precomputed_knn(x, params, ctx=ctx)

        log.info("Running LCMMDS")
        params["X"] = x
        embedded = lcmmds(**params)
        log.info("Embedding completed")

        return embedded


def skmmds(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    n_epochs: int = 500,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "spectral",
    n_samples: int = 5,
    sample_strategy: Literal["unif", "inc", "dec"] | None = None,
    random_state: int = 42,
    learning_rate: float = 1.0,
    opt: str = "adam",
    optargs: dict | None = None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    init_scale: float | Literal["knn"] | None = 10.0,
    pca: int | None = None,
    eps: float = 1e-10,
    n_random_neighbors: int = 0,
    nnbr_dist_strategy: Literal["dynamic", "fixed", "localmean", "mean"] = "dynamic",
    nbr_dist_strategy: Literal["localmean", "mean", "raw"] | None = None,
) -> np.ndarray:
    """Stochastic k-Metric Multidimensional Scaling"""
    nobs = X.shape[0]

    opt = setup_opt(nobs, learning_rate, opt, optargs)

    rng_state = setup_rngn(nobs, random_state)

    X, knn_idx, knn_dist = knn_pca(X, knn_idx, knn_dist, n_components=pca)

    Y = knn_init(X, knn_idx, knn_dist, init, random_state, init_scale)

    nbr_graph = symmetrize_nn(knn_idx, knn_dist, symmetrize=symmetrize)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    rnbr_graph = random_neighbors_init(
        X, n_random_neighbors, nnbr_dist_strategy, symmetrize
    )

    if nbr_dist_strategy is None:
        nbr_dist_strategy = "raw"
    if nbr_dist_strategy == "localmean":
        nnz = np.diff(nbr_graph.indptr)
        row_means = np.array(nbr_graph.sum(axis=1)).flatten() / nnz
        nbr_graph.data = row_means[np.repeat(np.arange(nbr_graph.shape[0]), nnz)]
    elif nbr_dist_strategy == "mean":
        nbr_graph.data = np.full_like(nbr_graph.data, np.mean(nbr_graph.data))
    elif nbr_dist_strategy == "raw":
        pass
    else:
        raise ValueError(f"Unknown nbr_dist_strategy: {nbr_dist_strategy}")

    rng_state = setup_rngn(nobs, random_state)
    return _skmmds(
        X,
        Y,
        n_epochs,
        opt,
        samples,
        rng_state,
        nbr_graph,
        rnbr_graph,
        eps,
    )


def _skmmds(
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    nbr_graph: csr_matrix,
    rnbr_graph: csr_matrix,
    eps: float,
) -> np.ndarray:
    nobs, ndim = Y.shape
    graph_j, graph_dist, ptr = nbr_graph.indices, nbr_graph.data, nbr_graph.indptr
    rgraph_j, rgraph_dist, rptr = rnbr_graph.indices, rnbr_graph.data, rnbr_graph.indptr

    n_rand_nbrs = np.diff(rptr)

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        _skmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, grads)
        _skmmds_non_nbrs_fixed(Y, samples[n], rgraph_j, rgraph_dist, rptr, eps, grads)
        _skmmds_non_nbrs(X, Y, n_rand_nbrs, samples[n], ptr, rng_state, eps, grads)

        Y = opt.opt(Y, grads, n, n_epochs)
        center(Y)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _skmmds_nbrs(
    Y: np.ndarray,
    graph_j: np.ndarray,
    graph_dist: np.ndarray,
    ptr: np.ndarray,
    eps: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            j = graph_j[edge]
            rij = graph_dist[edge]
            Yj = Y[j]

            dij = euclidean(Yi, Yj)
            grad_coeff = (dij - rij) / (dij + eps)

            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _skmmds_non_nbrs(
    X: np.ndarray,
    Y: np.ndarray,
    n_rand_nbrs: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    eps: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    nobs_minus_1 = nobs - 1
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        n_nbr_edges = ptr[i + 1] - ptr[i]
        n_nnbr_edges_to_sample = (n_nbr_edges * n_samples) - n_rand_nbrs[i]
        if n_nnbr_edges_to_sample <= 0:
            continue
        n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
        Xi = X[i]
        Yi = Y[i]
        # number of edges in i is ptr[i + 1] - ptr[i]
        for _ in range(n_nnbr_edges_to_sample):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            Yj = Y[j]

            rij = euclidean(Xi, X[j])
            dij = euclidean(Yi, Yj)

            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _skmmds_non_nbrs_fixed(
    Y: np.ndarray,
    n_samples: int,
    graph_j: np.ndarray,
    graph_dist: np.ndarray,
    ptr: np.ndarray,
    eps: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            if edge > n_samples:
                break

            j = graph_j[edge]
            rij = graph_dist[edge]
            Yj = Y[j]

            dij = euclidean(Yi, Yj)

            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@dataclass
class Skmmds(InitMixin, KNNMixin, drnb.embed.base.Embedder):
    """Stochastic k-Metric Multidimensional Scaling
    Attractive forces are on the symmetrized k-nearest neighbor edges
    Repulsive forces are on per-epoch random negative sampling plus some fixed random
    negatives determined before the optimization begins.
    """

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.handle_precomputed_init(params)
        params = self.handle_precomputed_knn(x, params, ctx=ctx)

        log.info("Running SKMMDS")
        params["X"] = x
        embedded = skmmds(**params)
        log.info("Embedding completed")

        return embedded


### SIKMMDS


def sikmmds(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    n_epochs: int = 500,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "spectral",
    n_samples: int = 5,
    sample_strategy: Literal["unif", "inc", "dec"] | None = None,
    random_state: int = 42,
    learning_rate: float = 1.0,
    opt: str = "adam",
    optargs: dict | None = None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    init_scale: float | Literal["knn"] | None = 10.0,
    pca: int | None = None,
    eps: float = 1e-10,
    n_random_neighbors: int = 0,
    nnbr_dist_strategy: Literal["dynamic", "fixed", "localmean", "mean"] = "dynamic",
    nbr_dist_strategy: Literal["localmean", "mean", "raw"] | None = None,
) -> np.ndarray:
    """Stochastic Isometric k-Metric Multidimensional Scaling"""
    nobs = X.shape[0]

    opt = setup_opt(nobs, learning_rate, opt, optargs)

    rng_state = setup_rngn(nobs, random_state)

    X, knn_idx, knn_dist = knn_pca(X, knn_idx, knn_dist, n_components=pca)

    Y = knn_init(X, knn_idx, knn_dist, init, random_state, init_scale)

    nbr_graph = symmetrize_nn(knn_idx, knn_dist, symmetrize=symmetrize)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    rnbr_graph = random_neighbors_init(
        X, n_random_neighbors, nnbr_dist_strategy, symmetrize
    )

    rng_state = setup_rngn(nobs, random_state)

    if nbr_dist_strategy is None:
        nbr_dist_strategy = "raw"
    if nbr_dist_strategy == "localmean":
        nnz = np.diff(nbr_graph.indptr)
        row_means = np.array(nbr_graph.sum(axis=1)).flatten() / nnz
        nbr_graph.data = row_means[np.repeat(np.arange(nbr_graph.shape[0]), nnz)]
    elif nbr_dist_strategy == "mean":
        nbr_graph.data = np.full_like(nbr_graph.data, np.mean(nbr_graph.data))
    elif nbr_dist_strategy == "raw":
        pass
    else:
        raise ValueError(f"Unknown nbr_dist_strategy: {nbr_dist_strategy}")

    return _sikmmds(
        X,
        Y,
        n_epochs,
        opt,
        samples,
        rng_state,
        nbr_graph,
        rnbr_graph,
        eps,
    )


def _sikmmds(
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    nbr_graph: csr_matrix,
    rnbr_graph: csr_matrix,
    eps: float,
) -> np.ndarray:
    nobs, ndim = Y.shape
    graph_j, graph_dist, ptr = nbr_graph.indices, nbr_graph.data, nbr_graph.indptr
    rgraph_j, rgraph_dist, rptr = rnbr_graph.indices, rnbr_graph.data, rnbr_graph.indptr
    n_rand_nbrs = np.diff(rptr)

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        _skmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, grads)
        _sikmmds_non_nbrs_fixed(Y, samples[n], rgraph_j, rgraph_dist, rptr, eps, grads)
        _sikmmds_non_nbrs(X, Y, n_rand_nbrs, samples[n], ptr, rng_state, eps, grads)

        Y = opt.opt(Y, grads, n, n_epochs)
        center(Y)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _sikmmds_non_nbrs(
    X: np.ndarray,
    Y: np.ndarray,
    n_rand_nbrs: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    eps: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    nobs_minus_1 = nobs - 1
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        n_nbr_edges = ptr[i + 1] - ptr[i]
        n_nnbr_edges_to_sample = (n_nbr_edges * n_samples) - n_rand_nbrs[i]
        if n_nnbr_edges_to_sample <= 0:
            continue
        n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
        Xi = X[i]
        Yi = Y[i]
        # number of edges in i is ptr[i + 1] - ptr[i]
        for _ in range(n_nnbr_edges_to_sample):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            Yj = Y[j]

            rij = euclidean(Xi, X[j])
            dij = euclidean(Yi, Yj)

            # assuming rij are meaningful for lower dimensions (they aren't but...)
            # then rij is the lower-bound on what dij should be: if dij is smaller,
            # update it to match rij, if dij is larger then we don't update
            if dij >= rij:
                continue

            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _sikmmds_non_nbrs_fixed(
    Y: np.ndarray,
    n_samples: int,
    graph_j: np.ndarray,
    graph_dist: np.ndarray,
    ptr: np.ndarray,
    eps: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            if edge > n_samples:
                break

            j = graph_j[edge]
            rij = graph_dist[edge]
            Yj = Y[j]

            dij = euclidean(Yi, Yj)
            if dij >= rij:
                continue

            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@dataclass
class Sikmmds(InitMixin, KNNMixin, drnb.embed.base.Embedder):
    """Stochastic Isometric k-Metric Multidimensional Scaling
    Attractive forces are on the symmetrized k-nearest neighbor edges
    Repulsive forces are on per-epoch random negative sampling plus some fixed random
    negatives determined before the optimization begins.
    If the embedded distance is larger than the ambient distance, the repulsive
    force is not updated.
    """

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.handle_precomputed_init(params)
        params = self.handle_precomputed_knn(x, params, ctx=ctx)

        log.info("Running SIKMMDS")
        params["X"] = x
        embedded = sikmmds(**params)
        log.info("Embedding completed")

        return embedded


# RSIKMMDS


def rsikmmds(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    n_epochs: int = 500,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "spectral",
    n_samples: int = 5,
    sample_strategy: Literal["unif", "inc", "dec"] | None = None,
    random_state: int = 42,
    learning_rate: float = 1.0,
    opt: str = "adam",
    optargs: dict | None = None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    init_scale: float | Literal["knn"] | None = 10.0,
    pca: int | None = None,
    eps: float = 1e-10,
    scale_dist: bool = False,
    local_pca: bool = False,
    far_weight: float = 1.0,
    iso_nbrs: Tuple[float, float] | bool = False,
    iso_nnbrs: Tuple[float, float] | bool = True,
    lc_nbrs: float | Tuple[float, float] | None = None,
    lc_nnbrs: float | Tuple[float, float] | None = None,
    plot_high: bool = False,
    plot_low: bool = False,
) -> np.ndarray:
    """Radius Stochastic Isometric k-Metric Multidimensional Scaling"""

    nobs = X.shape[0]

    opt = setup_opt(nobs, learning_rate, opt, optargs)

    rng_state = setup_rngn(nobs, random_state)

    X, knn_idx, knn_dist = knn_pca(X, knn_idx, knn_dist, n_components=pca)

    Y = knn_init(X, knn_idx, knn_dist, init, random_state, init_scale)

    axes = []
    if plot_high:
        ncols = 1
        if local_pca:
            ncols = 3

        _, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = (axes,)
        sns.histplot(
            knn_dist.flatten(),
            ax=axes[0],
        )
        axes[0].set_title("Ambient Neighbor Distances")

    if local_pca:
        ((lknn_idx, lknn_dist), varex) = embedded_nbr_dist(
            X, knn_idx, n_components=2, algorithm="arpack", random_state=random_state
        )
        log.info("Mean variances explained of local PCA: %f", np.mean(varex))
        if plot_high:
            sns.histplot(lknn_dist.flatten(), ax=axes[1])
            axes[1].set_title(
                f"LPCA Neighbor Distances varex={np.mean(varex) * 100.0:.2f}%"
            )
            sns.scatterplot(
                x=knn_dist.ravel(), y=lknn_dist.ravel(), ax=axes[2], s=8.0, alpha=0.5
            )
            axes[2].set_title("LPCA v Ambient Neighbor Distances")
            axes[2].set_xlabel("Ambient Neighbor Distances")
            axes[2].set_ylabel("LPCA Neighbor Distances")
            knn_idx, knn_dist = lknn_idx, lknn_dist
            plt.tight_layout()
            plt.show()

    if scale_dist:
        knn_dist = knn_dist / np.max(knn_dist)

    nbr_graph = symmetrize_nn(knn_idx, knn_dist, symmetrize=symmetrize)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    # rnbrs = random_neighbors(X, n_neighbors=knn_idx.shape[1], distance="euclidean")
    # random_radii = np.mean(rnbrs.dist, axis=1)
    # nbr_radii = np.mean(knn_dist, axis=1)
    # sns.histplot(random_radii / nbr_radii)
    # plt.show()
    # sns.histplot(
    #     calculate_exact_neighbors(Y, n_neighbors=knn_dist.shape[1] + 1)
    #     .dist[:, 1:]
    #     .flatten()
    # )
    # plt.show()
    Y = _rsikmmds(
        Y,
        n_epochs,
        opt,
        samples,
        rng_state,
        nbr_graph,
        eps,
        far_weight,
        iso_nbrs,
        iso_nnbrs,
        lc_nbrs,
        lc_nnbrs,
    )

    if plot_low:
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(4 * 3, 4))

        Ydist = neighbor_distances(Y, knn_idx)

        sns.scatterplot(
            x=knn_dist.ravel(), y=Ydist.ravel(), ax=axes[0], s=8.0, alpha=0.5
        )
        axes[0].set_title("Low v High Neighbor Distances")
        axes[0].set_xlabel("High D Neighbor Distances")
        axes[0].set_ylabel("Low D Neighbor Distances")

        sns.histplot(Ydist.ravel(), ax=axes[1])
        axes[1].set_title("Embedded Neighbor Distances")

        sns.histplot(
            calculate_exact_neighbors(Y, n_neighbors=knn_dist.shape[1] + 1)
            .dist[:, 1:]
            .flatten(),
            ax=axes[2],
        )
        axes[2].set_title("LowD Neighbor Distances")

        plt.tight_layout()
        plt.show()

    return Y


def _rsikmmds(
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    nbr_graph: csr_matrix,
    eps: float,
    far_weight: float,
    iso_nbrs: Tuple[float, float] | bool = False,
    iso_nnbrs: Tuple[float, float] | bool = True,
    lc_nbrs: float | Tuple[float, float] | None = None,
    lc_nnbrs: float | Tuple[float, float] | None = None,
) -> np.ndarray:
    nobs, ndim = Y.shape
    graph_j, graph_dist, ptr = nbr_graph.indices, nbr_graph.data, nbr_graph.indptr
    radii = nbr_graph.max(axis=1).toarray().flatten()
    if lc_nbrs is not None:
        if isinstance(lc_nbrs, tuple):
            nbr_func = _create_glcmmds_nbrs2(*lc_nbrs)
        elif iso_nbrs:
            nbr_func = _create_glcmmds_iso_nbrs(lc_nbrs)
        else:
            nbr_func = _create_glcmmds_nbrs(lc_nbrs)
    else:
        if iso_nbrs:
            nbr_func = _rsikmmds_nbrs
        else:
            nbr_func = _skmmds_nbrs
    if lc_nnbrs is not None:
        if isinstance(lc_nnbrs, tuple):
            nnbr_func = _create_glcmmds_nnbrs2(*lc_nnbrs)
        elif iso_nnbrs:
            nnbr_func = _create_glcmmds_iso_nnbrs(lc_nnbrs)
        else:
            nnbr_func = _create_glcmmds_nnbrs(lc_nnbrs)
    else:
        if iso_nnbrs:
            nnbr_func = _rsikmmds_non_nbrs
        else:
            nnbr_func = _rskmmds_non_nbrs

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        nbr_func(Y, graph_j, graph_dist, ptr, eps, grads)
        if np.any(np.isnan(grads)):
            raise ValueError(f"NaNs in nbr gradient epoch {n}")
        nnbr_func(radii, Y, samples[n], ptr, rng_state, eps, far_weight, grads)
        if np.any(np.isnan(grads)):
            raise ValueError(f"NaNs in nnbr gradient epoch {n}")
        Y = opt.opt(Y, grads, n, n_epochs)
        if np.any(np.isnan(Y)):
            raise ValueError(f"NaNs in Y epoch {n}")
        center(Y)

    return Y


def _create_glcmmds_nbrs2(delta1: float, delta2: float) -> callable:
    t1 = 1.2311 / delta1
    t2 = 1.2311 / delta2

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _glcmmds_nbrs(
        Y: np.ndarray,
        graph_j: np.ndarray,
        graph_dist: np.ndarray,
        ptr: np.ndarray,
        eps: float,
        grads: np.ndarray,
    ):
        nobs, ndim = Y.shape
        # pylint:disable=not-an-iterable
        for i in numba.prange(nobs):
            Yi = Y[i]
            for edge in range(ptr[i], ptr[i + 1]):
                j = graph_j[edge]
                rij = graph_dist[edge]
                Yj = Y[j]

                dij = euclidean(Yi, Yj)
                (t, delta) = (t2, delta2) if dij < rij else (t1, delta1)
                grad_coeff = -2.0 * delta * np.tanh(t * (rij - dij)) / (dij + eps)

                for d in range(ndim):
                    grads[i, d] += grad_coeff * (Yi[d] - Yj[d])

    return _glcmmds_nbrs


def _create_glcmmds_nbrs(delta: float) -> callable:
    t = 1.2311 / delta
    delta2 = -2.0 * delta

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _glcmmds_nbrs(
        Y: np.ndarray,
        graph_j: np.ndarray,
        graph_dist: np.ndarray,
        ptr: np.ndarray,
        eps: float,
        grads: np.ndarray,
    ):
        nobs, ndim = Y.shape
        # pylint:disable=not-an-iterable
        for i in numba.prange(nobs):
            Yi = Y[i]
            for edge in range(ptr[i], ptr[i + 1]):
                j = graph_j[edge]
                rij = graph_dist[edge]
                Yj = Y[j]

                dij = euclidean(Yi, Yj)
                grad_coeff = delta2 * np.tanh(t * (rij - dij)) / (dij + eps)
                # grad_coeff = delta2 / (dij + eps)

                for d in range(ndim):
                    grads[i, d] += grad_coeff * (Yi[d] - Yj[d])

    return _glcmmds_nbrs


def _create_glcmmds_iso_nbrs(delta: float) -> callable:
    t = 1.2311 / delta
    delta2 = -2.0 * delta

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _glcmmds_iso_nbrs(
        Y: np.ndarray,
        graph_j: np.ndarray,
        graph_dist: np.ndarray,
        ptr: np.ndarray,
        eps: float,
        grads: np.ndarray,
    ):
        nobs, ndim = Y.shape
        # pylint:disable=not-an-iterable
        for i in numba.prange(nobs):
            Yi = Y[i]
            for edge in range(ptr[i], ptr[i + 1]):
                j = graph_j[edge]
                rij = graph_dist[edge]
                Yj = Y[j]

                dij = euclidean(Yi, Yj)
                if dij < rij:
                    continue
                grad_coeff = delta2 * np.tanh(t * (rij - dij)) / (dij + eps)
                # grad_coeff = delta2 / (dij + eps)

                for d in range(ndim):
                    grads[i, d] += grad_coeff * (Yi[d] - Yj[d])

    return _glcmmds_iso_nbrs


def _create_glcmmds_iso_nnbrs(delta: float) -> callable:
    t = 1.2311 / delta
    delta2 = -2.0 * delta

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _glcmmds_non_nbrs(
        radii: np.ndarray,
        Y: np.ndarray,
        n_samples: int,
        ptr: np.ndarray,
        rng_state: np.ndarray,
        eps: float,
        far_weight: float,
        grads: np.ndarray,
    ):
        nobs, ndim = Y.shape
        nobs_minus_1 = nobs - 1

        # pylint:disable=not-an-iterable
        for i in numba.prange(nobs):
            n_nbr_edges = ptr[i + 1] - ptr[i]
            n_nnbr_edges_to_sample = n_nbr_edges * n_samples
            ri = radii[i]
            n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
            Yi = Y[i]
            for _ in range(n_nnbr_edges_to_sample):
                j = tau_rand_int(rng_state[i]) % nobs
                if i == j:
                    continue
                Yj = Y[j]

                rij = far_weight * (ri + radii[j])
                dij = euclidean(Yi, Yj)

                if dij >= rij:
                    continue

                grad_coeff = delta2 * np.tanh(t * (rij - dij)) / (dij + eps)
                # grad_coeff = delta2 / (dij + eps)
                for d in range(ndim):
                    grads[i, d] += grad_coeff * (Yi[d] - Yj[d])

    return _glcmmds_non_nbrs


def _create_glcmmds_nnbrs(delta: float) -> callable:
    t = 1.2311 / delta
    delta2 = -2.0 * delta

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _glcmmds_non_nbrs(
        radii: np.ndarray,
        Y: np.ndarray,
        n_samples: int,
        ptr: np.ndarray,
        rng_state: np.ndarray,
        eps: float,
        far_weight: float,
        grads: np.ndarray,
    ):
        nobs, ndim = Y.shape
        nobs_minus_1 = nobs - 1

        # pylint:disable=not-an-iterable
        for i in numba.prange(nobs):
            n_nbr_edges = ptr[i + 1] - ptr[i]
            n_nnbr_edges_to_sample = n_nbr_edges * n_samples
            ri = radii[i]
            n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
            Yi = Y[i]
            for _ in range(n_nnbr_edges_to_sample):
                j = tau_rand_int(rng_state[i]) % nobs
                if i == j:
                    continue
                Yj = Y[j]

                rij = far_weight * (ri + radii[j])
                dij = euclidean(Yi, Yj)

                grad_coeff = delta2 * np.tanh(t * (rij - dij)) / (dij + eps)
                # grad_coeff = delta2 / (dij + eps)
                for d in range(ndim):
                    grads[i, d] += grad_coeff * (Yi[d] - Yj[d])

    return _glcmmds_non_nbrs


def _create_glcmmds_nnbrs2(delta1: float, delta2: float) -> callable:
    t1 = 1.2311 / delta1
    t2 = 1.2311 / delta2

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _glcmmds_non_nbrs(
        radii: np.ndarray,
        Y: np.ndarray,
        n_samples: int,
        ptr: np.ndarray,
        rng_state: np.ndarray,
        eps: float,
        far_weight: float,
        grads: np.ndarray,
    ):
        nobs, ndim = Y.shape
        nobs_minus_1 = nobs - 1

        # pylint:disable=not-an-iterable
        for i in numba.prange(nobs):
            n_nbr_edges = ptr[i + 1] - ptr[i]
            n_nnbr_edges_to_sample = n_nbr_edges * n_samples
            ri = radii[i]
            n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
            Yi = Y[i]
            for _ in range(n_nnbr_edges_to_sample):
                j = tau_rand_int(rng_state[i]) % nobs
                if i == j:
                    continue
                Yj = Y[j]

                rij = far_weight * (ri + radii[j])
                dij = euclidean(Yi, Yj)
                (t, delta) = (t1, delta1) if dij < rij else (t2, delta2)

                grad_coeff = -2.0 * delta * np.tanh(t * (rij - dij)) / (dij + eps)
                for d in range(ndim):
                    grads[i, d] += grad_coeff * (Yi[d] - Yj[d])

    return _glcmmds_non_nbrs


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _rsikmmds_nbrs(
    Y: np.ndarray,
    graph_j: np.ndarray,
    graph_dist: np.ndarray,
    ptr: np.ndarray,
    eps: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            j = graph_j[edge]
            rij = graph_dist[edge]
            Yj = Y[j]

            dij = euclidean(Yi, Yj)
            if dij < rij:
                continue

            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _rsikmmds_non_nbrs(
    radii: np.ndarray,
    Y: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    eps: float,
    far_weight: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    nobs_minus_1 = nobs - 1

    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        n_nbr_edges = ptr[i + 1] - ptr[i]
        n_nnbr_edges_to_sample = n_nbr_edges * n_samples
        ri = radii[i]
        n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
        Yi = Y[i]
        for _ in range(n_nnbr_edges_to_sample):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            Yj = Y[j]

            rij = far_weight * (ri + radii[j])
            dij = euclidean(Yi, Yj)

            if dij >= rij:
                continue

            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _rskmmds_non_nbrs(
    radii: np.ndarray,
    Y: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    eps: float,
    far_weight: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    nobs_minus_1 = nobs - 1

    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        n_nbr_edges = ptr[i + 1] - ptr[i]
        n_nnbr_edges_to_sample = n_nbr_edges * n_samples
        ri = radii[i]
        n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
        Yi = Y[i]
        for _ in range(n_nnbr_edges_to_sample):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            Yj = Y[j]

            rij = far_weight * (ri + radii[j])
            dij = euclidean(Yi, Yj)

            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@dataclass
class Rsikmmds(InitMixin, KNNMixin, drnb.embed.base.Embedder):
    """Radius Stochastic Isometric k-Metric Multidimensional Scaling"""

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.handle_precomputed_init(params)
        params = self.handle_precomputed_knn(x, params, ctx=ctx)

        log.info("Running RSIKMMDS")
        params["X"] = x
        embedded = rsikmmds(**params)
        log.info("Embedding completed")

        return embedded


def mrsikmmds(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    n_epochs: int = 500,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "spectral",
    n_samples: int = 5,
    sample_strategy: Literal["unif", "inc", "dec"] | None = None,
    random_state: int = 42,
    learning_rate: float = 1.0,
    opt: str = "adam",
    optargs: dict | None = None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    init_scale: float | Literal["knn"] | None = 10.0,
    pca: int | None = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Multi-Radius Stochastic Isometric k-Metric Multidimensional Scaling"""
    nobs = X.shape[0]

    opt = setup_opt(nobs, learning_rate, opt, optargs)

    rng_state = setup_rngn(nobs, random_state)

    X, knn_idx, knn_dist = knn_pca(X, knn_idx, knn_dist, n_components=pca)

    Y = knn_init(
        X,
        knn_idx,
        knn_dist,
        init,
        random_state,
        init_scale,
    )

    nbr_graph = symmetrize_nn(knn_idx, knn_dist, symmetrize=symmetrize)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    rnbrs = random_neighbors(X, n_neighbors=knn_idx.shape[1], distance="euclidean")
    radii = np.mean(rnbrs.dist, axis=1)
    # radii = np.mean(rnbrs.dist)

    Y = _mrsikmmds(Y, n_epochs, opt, samples, rng_state, nbr_graph, radii, eps)

    return Y


def _mrsikmmds(
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    nbr_graph: csr_matrix,
    radii: np.ndarray,
    eps: float,
) -> np.ndarray:
    nobs, ndim = Y.shape
    graph_j, graph_dist, ptr = nbr_graph.indices, nbr_graph.data, nbr_graph.indptr
    lradii = nbr_graph.max(axis=1).toarray().flatten()

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        _skmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, grads)
        _mrsikmmds_non_nbrs(radii, lradii, Y, samples[n], ptr, rng_state, eps, grads)

        Y = opt.opt(Y, grads, n, n_epochs)
        center(Y)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _mrsikmmds_non_nbrs(
    radii: np.ndarray,
    lradii: np.ndarray,
    Y: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    eps: float,
    grads: np.ndarray,
):
    nobs, ndim = Y.shape
    nobs_minus_1 = nobs - 1

    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        n_nbr_edges = ptr[i + 1] - ptr[i]
        n_nnbr_edges_to_sample = n_nbr_edges * n_samples
        ri = radii[i]
        lri = lradii[i]
        n_nnbr_edges_to_sample = min(n_nnbr_edges_to_sample, nobs_minus_1)
        Yi = Y[i]
        for _ in range(n_nnbr_edges_to_sample):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            Yj = Y[j]

            rij = ri + radii[j]
            dij = euclidean(Yi, Yj)

            if dij >= rij:
                continue

            rij = 0.5 * (lri + lradii[j])
            grad_coeff = (dij - rij) / (dij + eps)
            for d in range(ndim):
                grads[i, d] += grad_coeff * (Yi[d] - Yj[d])


@dataclass
class Mrsikmmds(InitMixin, KNNMixin, drnb.embed.base.Embedder):
    """Multi-Radius Stochastic Isometric k-Metric Multidimensional Scaling"""

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.handle_precomputed_init(params)
        params = self.handle_precomputed_knn(x, params, ctx=ctx)

        log.info("Running MRSIKMMDS")
        params["X"] = x
        embedded = mrsikmmds(**params)
        log.info("Embedding completed")

        return embedded
