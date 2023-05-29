from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
import drnb.neighbors as nbrs
from drnb.distances import distance_function
from drnb.embed.umap.utils import euclidean
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.yinit import standard_neighbor_init


@numba.njit()
def clip(val, max_val=4.0, min_val=-4.0):
    if val > max_val:
        return max_val
    if val < min_val:
        return min_val
    return val


def ivhd(
    knn_idx,
    n_random=1,
    n_epochs=200,
    near_dist=0.0,
    far_dist=1.0,
    far_weight=0.01,
    init="pca",
    learning_rate=None,
    opt="adam",
    optargs: Optional[dict] = None,
    sample_strategy="unif",
    symmetrize: Optional[str] = "or",
    eps=1e-5,
    random_state=42,
    X=None,
    init_scale=None,
    clip_grad=4.0,
):
    if optargs is None:
        optargs = {}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate

    nobs = knn_idx.shape[0]
    optim = create_opt(nobs, opt, optargs)
    rng_state = setup_rngn(nobs, random_state)

    Y = standard_neighbor_init(
        init,
        nobs=nobs,
        random_state=random_state,
        knn_idx=knn_idx,
        X=X,
        init_scale=init_scale,
    )

    dmat = nn_to_sparse(knn_idx, symmetrize=symmetrize)
    dmat.eliminate_zeros()

    knn_i = dmat.row
    knn_j = dmat.col
    ptr = dmat.tocsr().indptr

    samples = create_sample_plan(
        n_samples=n_random, n_epochs=n_epochs, strategy=sample_strategy
    )

    Y = _ivhd(
        Y,
        n_epochs,
        optim,
        samples,
        rng_state,
        knn_i,
        knn_j,
        ptr,
        near_dist,
        far_dist,
        far_weight,
        eps,
        clip_grad,
    )

    return Y


@numba.jit(nopython=True, fastmath=True)
def _ivhd(
    Y,
    n_epochs,
    opt,
    samples,
    rng_state,
    knn_i,
    knn_j,
    ptr,
    near_dist,
    far_dist,
    far_weight,
    eps,
    clip_grad,
):
    nobs, ndim = Y.shape
    degrees = np.zeros((nobs,), dtype=np.int32)
    for i in knn_i:
        degrees[i] = degrees[i] + 1

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        # neighbors
        _ivhd_nbrs(Y, knn_i, knn_j, ptr, near_dist, eps, grads, clip_grad)

        # non-neighbors
        _ivhd_inner_nnbr(
            Y,
            samples[n],
            degrees,
            rng_state,
            far_dist,
            far_weight,
            eps,
            grads,
            clip_grad,
        )

        Y = opt.opt(Y, grads, n, n_epochs)

    for d in range(ndim):
        Yd = Y[:, d]
        Y[:, d] = Yd - np.mean(Yd)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _ivhd_nbrs(Y, knn_i, knn_j, ptr, near_dist, eps, grads, clip_grad):
    ndim = Y.shape[1]
    # pylint:disable=not-an-iterable
    for p in numba.prange(len(ptr) - 1):
        for edge in range(ptr[p], ptr[p + 1]):
            i = knn_i[edge]
            Yi = Y[i]

            j = knn_j[edge]
            dij = euclidean(Yi, Y[j])

            grad_coeff = (dij - near_dist) / (dij + eps)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(
                    grad_coeff * (Yi[d] - Y[j, d]),
                    max_val=clip_grad,
                    min_val=-clip_grad,
                )


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _ivhd_inner_nnbr(
    Y, nsamples, degrees, rng_state, far_dist, far_weight, eps, grads, clip_grad
):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for _ in range(degrees[i] * nsamples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            dij = euclidean(Yi, Y[j])

            grad_coeff = far_weight * (dij - far_dist) / (dij + eps)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(
                    grad_coeff * (Yi[d] - Y[j, d]),
                    max_val=clip_grad,
                    min_val=-clip_grad,
                )


@dataclass
class Ivhd(drnb.embed.Embedder):
    precomputed_init: Optional[np.ndarray] = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init

        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = 2

        precomputed_knn = nbrs.get_neighbors_with_ctx(
            x,
            params.get("metric", "euclidean"),
            n_neighbors + 1,
            return_distance=False,
            ctx=ctx,
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        return embed_ivhd(x, params)


def embed_ivhd(
    x,
    params,
):
    log.info("Running IVHD")
    params["X"] = x
    embedded = ivhd(**params)
    log.info("Embedding completed")

    return embedded


def xvhd(
    knn_idx,
    n_random=1,
    n_epochs=200,
    near_dist=0.0,
    far_dist=1.0,
    far_weight=0.01,
    power=1.0,
    init="pca",
    learning_rate=None,
    opt="adam",
    optargs=None,
    sample_strategy="unif",
    symmetrize=None,
    eps=1e-5,
    random_state=42,
    X=None,
    init_scale=None,
):
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        if optargs is None:
            optargs = {}
        optargs["alpha"] = learning_rate
    nobs = knn_idx.shape[0]
    optim = create_opt(nobs, opt, optargs)
    rng_state = setup_rngn(nobs, random_state)

    Y = standard_neighbor_init(
        init,
        nobs=nobs,
        random_state=random_state,
        knn_idx=knn_idx,
        X=X,
        init_scale=init_scale,
    )

    ydfun = distance_function("euclidean")

    dmat = nn_to_sparse(knn_idx, symmetrize=symmetrize)
    dmat.eliminate_zeros()
    knn_i = dmat.row
    knn_j = dmat.col

    samples = create_sample_plan(
        n_samples=n_random, n_epochs=n_epochs, strategy=sample_strategy
    )

    Y = _xvhd(
        Y,
        n_epochs,
        ydfun,
        optim,
        samples,
        rng_state,
        knn_i,
        knn_j,
        near_dist,
        far_dist,
        far_weight,
        power,
        eps,
    )

    return Y


@numba.jit(nopython=True, fastmath=True)
def _xvhd_inner_nbr(
    Y, ydfun, knn_i, knn_j, near_dist, eps, nedges, grads, radii, power, counts
):
    # neighbors
    for edge in range(nedges):
        i = knn_i[edge]
        j = knn_j[edge]
        dij = ydfun(Y[i], Y[j])
        pdij = pow(dij, power)
        radii[i] = radii[i] + pdij
        radii[j] = radii[j] + pdij
        grad_coeff = (dij - near_dist) / (dij + eps) * (Y[i] - Y[j])
        grads[i] = grads[i] + grad_coeff
        grads[j] = grads[j] - grad_coeff

    p1 = 1.0 / power
    for i in range(radii.shape[0]):
        radii[i] = pow(radii[i] / counts[i], p1)
    return grads


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _xvhd_inner_nnbr(
    Y,
    ydfun,
    rng_state,
    nsamples,
    far_dist,
    far_weight,
    eps,
    nobs,
    grads,
    radii,
):
    # non-neighbors
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        for _ in range(nsamples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            dij = ydfun(Y[i], Y[j])

            rij = radii[i] + radii[j]
            if dij < rij:
                grad_coeff = far_weight * (dij - rij * far_dist) / (dij + eps)
                # grads[i] = grads[i] + grad_coeff * (Y[i] - Y[j])
                grads[j] = grads[j] - grad_coeff * (Y[i] - Y[j])
    return grads


@numba.jit(nopython=True, fastmath=True)
def _xvhd(
    Y,
    n_epochs,
    ydfun,
    opt,
    samples,
    rng_state,
    knn_i,
    knn_j,
    near_dist,
    far_dist,
    far_weight,
    power,
    eps,
):
    nobs, ndim = Y.shape
    nedges = len(knn_i)
    radii = np.zeros(nobs, dtype=np.float32)

    counts = np.zeros(radii.shape[0], dtype=np.float32)
    for edge in range(nedges):
        i = knn_i[edge]
        j = knn_j[edge]
        counts[i] = counts[i] + 1
        counts[j] = counts[j] + 1

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        nsamples = samples[n]
        # neighbors
        grads = _xvhd_inner_nbr(
            Y, ydfun, knn_i, knn_j, near_dist, eps, nedges, grads, radii, power, counts
        )

        # non-neighbors
        grads = _xvhd_inner_nnbr(
            Y, ydfun, rng_state, nsamples, far_dist, far_weight, eps, nobs, grads, radii
        )

        Y = opt.opt(Y, grads, n, n_epochs)
    for d in range(ndim):
        Y[:, d] -= np.mean(Y[:, d])
    return Y


@dataclass
class Xvhd(drnb.embed.Embedder):
    precomputed_init: Optional[np.ndarray] = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init

        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = 2

        precomputed_knn = nbrs.get_neighbors_with_ctx(
            x,
            params.get("metric", "euclidean"),
            n_neighbors + 1,
            return_distance=False,
            ctx=ctx,
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        return embed_xvhd(x, params)


def embed_xvhd(
    x,
    params,
):
    log.info("Running XVHD")
    params["X"] = x
    embedded = xvhd(**params)
    log.info("Embedding completed")

    return embedded
