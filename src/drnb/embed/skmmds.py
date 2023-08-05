from dataclasses import dataclass
from typing import Literal, Optional

import numba
import numpy as np
from scipy.sparse import csr_matrix
from umap.utils import tau_rand_int

import drnb.embed
import drnb.neighbors as nbrs
import drnb.neighbors.random
from drnb.embed import EmbedContext
from drnb.embed.mixins import InitMixin, KNNMixin
from drnb.embed.umap.utils import euclidean
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.neighbors.random import random_neighbors
from drnb.optim import create_opt
from drnb.preprocess import pca as pca_reduce
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.yinit import standard_neighbor_init


def center(Y):
    ndim = Y.shape[1]
    for d in range(ndim):
        Yd = Y[:, d]
        Y[:, d] = Yd - np.mean(Yd)


def setup_opt(nobs, learning_rate=1.0, opt="adam", optargs: Optional[dict] = None):
    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate
    return create_opt(nobs, opt, optargs)


def knn_pca(X, knn_idx, knn_dist, n_components=None):
    if n_components is not None:
        if np.min(X.shape) > n_components:
            X = pca_reduce(X, n_components=n_components)
            n_neighbors = knn_idx.shape[1]
            log.info("Calculating new nearest neighbor data after PCA reduction")
            pca_nn = nbrs.calculate_exact_neighbors(
                data=X, metric="euclidean", n_neighbors=n_neighbors
            )
            knn_idx, knn_dist = pca_nn.idx, pca_nn.dist
        else:
            log.info(
                "Requested PCA with n_components=%d is not possible, skipping",
                n_components,
            )
    return X, knn_idx, knn_dist


def knn_init(X, knn_idx, knn_dist, init, random_state, init_scale):
    if init_scale == "knn":
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


def symmetrize_nn(nbr_idx, nbr_dist, symmetrize="or") -> csr_matrix:
    dmat = nn_to_sparse([nbr_idx, nbr_dist], symmetrize=symmetrize)
    dmat.eliminate_zeros()
    return dmat.tocsr()


def symmetrized_random_neighbors(
    X, n_samples, nnbr_dist_strategy, symmetrize="or"
) -> csr_matrix:
    rnbrs = random_neighbors(
        X, n_neighbors=n_samples, distance="euclidean", random_state=42
    )
    rnbrs_idx = rnbrs.idx
    rnbrs_dist = rnbrs.dist
    if nnbr_dist_strategy == "localmean":
        rnbrs_dist = np.mean(rnbrs_dist, axis=1)[:, np.newaxis].repeat(
            rnbrs_dist.shape[1], axis=1
        )
    elif nnbr_dist_strategy == "mean":
        rnbrs_dist = np.full_like(rnbrs_dist, np.mean(rnbrs_dist))

    return symmetrize_nn(rnbrs_idx, rnbrs_dist, symmetrize=symmetrize)


def random_neighbors_init(X, n_random_neighbors: int, nnbr_dist_strategy, symmetrize):
    if n_random_neighbors == 0:
        log.info("Using dynamically allocated random neighbors only")
        nobs = X.shape[0]
        return csr_matrix((nobs, nobs))
    log.info("Using %d pre-allocated random non-neighbors graph", n_random_neighbors)
    return symmetrized_random_neighbors(
        X, n_random_neighbors, nnbr_dist_strategy, symmetrize=symmetrize
    )


def skmmds(
    X,
    knn_idx,
    knn_dist,
    n_epochs=500,
    init="spectral",
    n_samples=5,
    sample_strategy=None,
    random_state=42,
    learning_rate=1.0,
    opt="adam",
    optargs: Optional[dict] = None,
    symmetrize="or",
    init_scale: float | Literal["knn"] = 10.0,
    pca: Optional[int] = None,
    eps: float = 1e-10,
    n_random_neighbors: int = 0,
    nnbr_dist_strategy: Literal["dynamic", "fixed", "localmean", "mean"] = "dynamic",
):
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
    X,
    Y,
    n_epochs,
    opt,
    samples,
    rng_state,
    nbr_graph,
    rnbr_graph,
    eps,
):
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
def _skmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, grads):
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
def _skmmds_non_nbrs(X, Y, n_rand_nbrs, n_samples, ptr, rng_state, eps, grads):
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
def _skmmds_non_nbrs_fixed(Y, n_samples, graph_j, graph_dist, ptr, eps, grads):
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
class Skmmds(InitMixin, KNNMixin, drnb.embed.Embedder):
    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: Optional[EmbedContext] = None
    ):
        params = self.handle_precomputed_init(params)
        params = self.handle_precomputed_knn(x, params, ctx=ctx)

        return embed_skmmds(x, params)


def embed_skmmds(
    x,
    params,
):
    log.info("Running SKMMDS")
    params["X"] = x
    embedded = skmmds(**params)
    log.info("Embedding completed")

    return embedded


### SIKMMDS


def sikmmds(
    X,
    knn_idx,
    knn_dist,
    n_epochs=500,
    init="spectral",
    n_samples=5,
    sample_strategy=None,
    random_state=42,
    learning_rate=1.0,
    opt="adam",
    optargs: Optional[dict] = None,
    symmetrize="or",
    init_scale: float | Literal["knn"] = 10.0,
    pca: Optional[int] = None,
    eps: float = 1e-10,
    n_random_neighbors: int = 0,
    nnbr_dist_strategy: Literal["dynamic", "fixed", "localmean", "mean"] = "dynamic",
):
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
    X,
    Y,
    n_epochs,
    opt,
    samples,
    rng_state,
    nbr_graph,
    rnbr_graph,
    eps,
):
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
def _sikmmds_non_nbrs(X, Y, n_rand_nbrs, n_samples, ptr, rng_state, eps, grads):
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
def _sikmmds_non_nbrs_fixed(Y, n_samples, graph_j, graph_dist, ptr, eps, grads):
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
class Sikmmds(InitMixin, KNNMixin, drnb.embed.Embedder):
    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: Optional[EmbedContext] = None
    ):
        params = self.handle_precomputed_init(params)
        params = self.handle_precomputed_knn(x, params, ctx=ctx)

        return embed_sikmmds(x, params)


def embed_sikmmds(
    x,
    params,
):
    log.info("Running SIKMMDS")
    params["X"] = x
    embedded = sikmmds(**params)
    log.info("Embedding completed")

    return embedded
