from dataclasses import dataclass
from typing import Literal, Optional, cast

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
import drnb.neighbors as nbrs
import drnb.neighbors.random
from drnb.distances import distance_function
from drnb.embed.umap.utils import euclidean
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.preprocess import pca as pca_reduce
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.yinit import pca as pca_init
from drnb.yinit import scale_coords, standard_neighbor_init, umap_random_init


def smmds(
    X,
    metric="euclidean",
    n_epochs=200,
    init="pca",
    n_samples=3,
    sample_strategy=None,
    random_state=42,
    learning_rate=None,
    opt="adam",
    optargs=None,
    eps=1e-10,
    pca=None,
    init_scale=None,
):
    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate

    nobs = X.shape[0]
    optim = create_opt(nobs, opt, optargs)

    rng_state = setup_rngn(nobs, random_state)
    xdfun = distance_function(metric)

    if pca is not None:
        X = pca_reduce(X, n_components=pca)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    Y = mmds_init(init, nobs, X, init_scale, random_state)

    Y = _smmds(X, Y, n_epochs, eps, xdfun, optim, samples, rng_state)

    return Y


def _smmds(X, Y, n_epochs, eps, xdfun, opt, samples, rng_state):
    nobs, ndim = Y.shape

    for n in range(n_epochs):
        grads = _smmds_epoch(
            X,
            Y,
            eps,
            xdfun,
            rng_state,
            samples[n],
            nobs,
            ndim,
        )

        Y = opt.opt(Y, grads, n, n_epochs)

        for d in range(ndim):
            Yd = Y[:, d]
            Y[:, d] = Yd - np.mean(Yd)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _smmds_epoch(
    X,
    Y,
    eps,
    xdfun,
    rng_state,
    n_samples,
    nobs,
    ndim,
):
    grads = np.zeros((nobs, ndim), dtype=np.float32)
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Xi = X[i]
        Yi = Y[i]
        for _ in range(n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            Yj = Y[j]
            rij = xdfun(Xi, X[j])
            dij = euclidean(Yi, Yj)
            # by not updating grads[j] this is safe to parallelize
            for d in range(ndim):
                grads[i, d] = grads[i, d] + ((dij - rij) / (dij + eps)) * (
                    Yi[d] - Yj[d]
                )
    return grads


def mmds_init(init, nobs, X=None, init_scale=None, random_state=42):
    if isinstance(init, np.ndarray):
        if init.shape != (nobs, 2):
            raise ValueError("Initialization array has incorrect shape")
        log.info("Using pre-supplied initialization coordinates")
        Y = init
    elif init == "pca":
        Y = pca_init(X)
    elif init == "pcaw":
        Y = pca_init(X, whiten=True)
    elif init == "rand":
        Y = umap_random_init(nobs, random_state)
    else:
        raise ValueError(f"Unknown init option '{init}'")
    Y = Y.astype(np.float32, order="C")
    if init_scale is not None:
        Y = scale_coords(Y, max_coord=init_scale)
    return Y


@dataclass
class Smmds(drnb.embed.Embedder):
    precomputed_init: Optional[np.ndarray] = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init
        return embed_smmds(x, params)


def embed_smmds(
    x,
    params,
):
    log.info("Running SMMDS")
    params["X"] = x
    embedded = smmds(**params)
    log.info("Embedding completed")

    return embedded


def snmds(
    X,
    metric="euclidean",
    n_epochs=200,
    init="pca",
    n_samples=3,
    random_state=42,
    learning_rate=None,
    opt="adam",
    optargs=None,
    eps=1e-10,
    init_scale=None,
    pca=None,
):
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        if optargs is None:
            optargs = {}
        optargs["alpha"] = learning_rate
    nobs = X.shape[0]
    optim = create_opt(nobs, opt, optargs)
    rng_state = setup_rngn(nobs, random_state)

    xdfun = distance_function(metric)

    Y = mmds_init(init, nobs, X, init_scale, random_state)
    if pca is not None:
        X = pca_reduce(X, n_components=pca)
    Y = _snmds(X, Y, n_epochs, eps, xdfun, optim, n_samples, rng_state)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _snmds(X, Y, n_epochs, eps, xdfun, opt, n_samples, rng_state):
    nobs, ndim = Y.shape
    js = np.empty(n_samples, dtype=np.int32)
    pq = np.empty(n_samples, dtype=np.float32)
    rs = np.empty(n_samples, dtype=np.float32)
    ds = np.empty(n_samples, dtype=np.float32)
    for n in range(n_epochs):
        grads = _snmds_epoch(
            X, Y, eps, xdfun, n_samples, nobs, ndim, rng_state, js, pq, rs, ds
        )

        Y = opt.opt(Y, grads, n, n_epochs)
        for d in range(ndim):
            Yd = Y[:, d]
            Y[:, d] = Yd - np.mean(Yd)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _snmds_epoch(X, Y, eps, xdfun, n_samples, nobs, ndim, rng_state, js, pq, rs, ds):
    grads = np.zeros((nobs, ndim), dtype=np.float32)

    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Xi = X[i]
        rsum = 0.0

        Yi = Y[i]
        dsum = 0.0

        for k in range(n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            js[k] = j

            if i == j:
                rij = eps
                dij = eps
            else:
                rij = xdfun(Xi, X[j])
                dij = euclidean(Yi, Y[j])

            rs[k] = rij
            rsum += rij

            ds[k] = dij
            dsum += dij

        pqq = 0.0
        for k in range(n_samples):
            q = ds[k] / dsum
            pq[k] = (rs[k] / rsum) - q
            pqq += pq[k] * q
            # re-use the ds vector
            ds[k] = ds[k] * dsum + eps

        for k in range(n_samples):
            j = js[k]
            Yj = Y[j]
            if i == j:
                continue
            for d in range(ndim):
                gy = ((pqq - pq[k]) / ds[k]) * (Yi[d] - Yj[d])
                grads[i, d] = grads[i, d] + gy
                grads[j, d] = grads[j, d] - gy

    return grads


@dataclass
class Snmds(drnb.embed.Embedder):
    precomputed_init: Optional[np.ndarray] = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init
        return embed_snmds(x, params)


def embed_snmds(
    x,
    params,
):
    log.info("Running SMMDS")
    params["X"] = x
    embedded = snmds(**params)
    log.info("Embedding completed")

    return embedded


def mmds(
    X,
    metric="euclidean",
    n_epochs=200,
    init="pca",
    random_state=42,
    learning_rate=None,
    opt="adam",
    optargs=None,
    eps=1e-10,
    init_scale=None,
):
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        if optargs is None:
            optargs = {}
        optargs["alpha"] = learning_rate
    nobs = X.shape[0]
    optim = create_opt(nobs, opt, optargs)
    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")

    Y = mmds_init(init, nobs, X, init_scale, random_state)

    Y = _mmds(X, Y, n_epochs, eps, xdfun, ydfun, optim)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _mmds(X, Y, n_epochs, eps, xdfun, ydfun, opt):
    nobs, ndim = Y.shape

    for n in range(n_epochs):
        grads = _mmds_epoch(X, Y, eps, xdfun, ydfun, nobs, ndim)

        Y = opt.opt(Y, grads, n, n_epochs)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _mmds_epoch(X, Y, eps, xdfun, ydfun, nobs, ndim):
    grads = np.zeros((nobs, ndim), dtype=np.float32)
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Xi = X[i]
        Yi = Y[i]
        for j in range(nobs):
            if i == j:
                continue

            Yj = Y[j]
            rij = xdfun(Xi, X[j])
            dij = ydfun(Yi, Yj)
            # by not updating grads[j] this is safe to parallelize
            grads[i] = grads[i] + ((dij - rij) / (dij + eps)) * (Yi - Yj)
    return grads


@dataclass
class Mmds(drnb.embed.Embedder):
    precomputed_init: Optional[np.ndarray] = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init
        return embed_mmds(x, params)


def embed_mmds(
    x,
    params,
):
    log.info("Running MMDS")
    params["X"] = x
    embedded = mmds(**params)
    log.info("Embedding completed")

    return embedded


# SKMMDS


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
):
    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate

    nobs = X.shape[0]
    optim = create_opt(nobs, opt, optargs)
    rng_state = setup_rngn(nobs, random_state)

    if init_scale == "knn":
        init_scale = np.mean(knn_dist)
    Y = standard_neighbor_init(
        init,
        nobs=nobs,
        random_state=random_state,
        knn_idx=knn_idx,
        X=X,
        init_scale=init_scale,
    )

    dmat = nn_to_sparse([knn_idx, knn_dist], symmetrize=symmetrize)
    dmat.eliminate_zeros()

    graph_j = dmat.col
    graph_dist = dmat.data
    ptr = dmat.tocsr().indptr

    # this is probably bad because now the non-neighbors have distances
    # drawn from a different distribution to the neighbors
    if pca is not None and np.min(X.shape) > pca:
        X = pca_reduce(X, n_components=pca)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    Y = _skmmds(
        X,
        Y,
        n_epochs,
        optim,
        samples,
        rng_state,
        graph_j,
        graph_dist,
        ptr,
        eps,
    )

    return Y


def _skmmds(X, Y, n_epochs, opt, samples, rng_state, graph_j, graph_dist, ptr, eps):
    nobs, ndim = Y.shape
    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        _skmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, grads)
        _skmmds_non_nbrs(X, Y, samples[n], ptr, rng_state, eps, grads)
        Y = opt.opt(Y, grads, n, n_epochs)

        for d in range(ndim):
            Yd = Y[:, d]
            Y[:, d] = Yd - np.mean(Yd)
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
            for d in range(ndim):
                grads[i, d] = grads[i, d] + ((dij - rij) / (dij + eps)) * (
                    Yi[d] - Yj[d]
                )


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _skmmds_non_nbrs(X, Y, n_samples, ptr, rng_state, eps, grads):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Xi = X[i]
        Yi = Y[i]
        # number of edges in i is ptr[i + 1] - ptr[i]
        for _ in range((ptr[i + 1] - ptr[i]) * n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            Yj = Y[j]

            rij = euclidean(Xi, X[j])
            dij = euclidean(Yi, Yj)

            for d in range(ndim):
                grads[i, d] = grads[i, d] + ((dij - rij) / (dij + eps)) * (
                    Yi[d] - Yj[d]
                )


@dataclass
class Skmmds(drnb.embed.Embedder):
    precomputed_init: Optional[np.ndarray] = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init

        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = 15

        precomputed_knn = nbrs.get_neighbors_with_ctx(
            x, params.get("metric", "euclidean"), n_neighbors + 1, ctx=ctx
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        precomputed_knn.dist = cast(np.ndarray, precomputed_knn.dist)
        params["knn_dist"] = precomputed_knn.dist[:, 1:]
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
):
    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate

    nobs = X.shape[0]
    optim = create_opt(nobs, opt, optargs)
    rng_state = setup_rngn(nobs, random_state)

    if init_scale == "knn":
        init_scale = np.mean(knn_dist)
    Y = standard_neighbor_init(
        init,
        nobs=nobs,
        random_state=random_state,
        knn_idx=knn_idx,
        X=X,
        init_scale=init_scale,
    )

    dmat = nn_to_sparse([knn_idx, knn_dist], symmetrize=symmetrize)
    dmat.eliminate_zeros()

    graph_j = dmat.col
    graph_dist = dmat.data
    ptr = dmat.tocsr().indptr

    if pca is not None and np.min(X.shape) > pca:
        X = pca_reduce(X, n_components=pca)

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    Y = _sikmmds(
        X,
        Y,
        n_epochs,
        optim,
        samples,
        rng_state,
        graph_j,
        graph_dist,
        ptr,
        eps,
    )

    return Y


def _sikmmds(
    X,
    Y,
    n_epochs,
    opt,
    samples,
    rng_state,
    graph_j,
    graph_dist,
    ptr,
    eps,
):
    nobs, ndim = Y.shape

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)

        _sikmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, grads)
        _sikmmds_non_nbrs(X, Y, samples[n], ptr, rng_state, eps, grads)

        Y = opt.opt(Y, grads, n, n_epochs)
        for d in range(ndim):
            Yd = Y[:, d]
            Y[:, d] = Yd - np.mean(Yd)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _sikmmds_nbrs(Y, graph_j, graph_dist, ptr, eps, grads):
    ndim = Y.shape[1]
    # pylint:disable=not-an-iterable
    for i in numba.prange(len(ptr) - 1):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            j = graph_j[edge]
            rij = graph_dist[edge]
            Yj = Y[j]

            dij = euclidean(Yi, Yj)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + ((dij - rij) / (dij + eps)) * (
                    Yi[d] - Yj[d]
                )


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _sikmmds_non_nbrs(X, Y, n_samples, ptr, rng_state, eps, grads):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Xi = X[i]
        Yi = Y[i]
        # number of edges in i is ptr[i + 1] - ptr[i]
        for _ in range((ptr[i + 1] - ptr[i]) * n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            Yj = Y[j]

            rij = euclidean(Xi, X[j])
            dij = euclidean(Yi, Yj)
            # assuming rij are meaningful for lower dimensions (they aren't but...)
            # then rij is the lower-bound on what dij should be: if dij is smaller,
            # update it to match rij, if dij is larger then we don't update
            # if dij > rij then make the grad coeff zero by setting rij = dij
            rij = max(dij, rij)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + ((dij - rij) / (dij + eps)) * (
                    Yi[d] - Yj[d]
                )


@dataclass
class Sikmmds(drnb.embed.Embedder):
    precomputed_init: Optional[np.ndarray] = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init

        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = 15

        precomputed_knn = nbrs.get_neighbors_with_ctx(
            x, params.get("metric", "euclidean"), n_neighbors + 1, ctx=ctx
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        precomputed_knn.dist = cast(np.ndarray, precomputed_knn.dist)
        params["knn_dist"] = precomputed_knn.dist[:, 1:]
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
