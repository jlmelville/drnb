import math
from dataclasses import dataclass

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
import drnb.neighbors as nbrs
from drnb.dimension import mle_global
from drnb.embed.umap.utils import clip, rdist
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.yinit import binary_graph_spectral_init, pca, scale_coords, umap_random_init


def spacemap(
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
    optargs=None,
    symmetrize="or",
    init_scale=10.0,
    dglobal="auto",
):
    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate
    nobs = X.shape[0]
    optim = create_opt(nobs, opt, optargs)
    rng_state = setup_rngn(nobs, random_state)

    if isinstance(init, np.ndarray):
        if init.shape != (nobs, 2):
            raise ValueError("Initialization array has incorrect shape")
        log.info("Using pre-supplied initialization coordinates")
        Y = init
    elif init == "pca":
        Y = pca(X)
    elif init == "rand":
        Y = umap_random_init(nobs, random_state)
    elif init == "spectral":
        Y = binary_graph_spectral_init(knn=knn_idx)
    else:
        raise ValueError(f"Unknown init option '{init}'")
    if init_scale is not None:
        Y = scale_coords(Y, max_coord=init_scale)
    Y = Y.astype(np.float32, order="C")

    dmat = nn_to_sparse(knn_idx, symmetrize=symmetrize)
    dmat.eliminate_zeros()

    knn_i = dmat.row
    knn_j = dmat.col
    ptr = dmat.tocsr().indptr

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    if isinstance(dglobal, str):
        if dglobal == "auto":
            dglobal = mle_global(knn_dist)
            log.info("estimated intrinsic dimensionality = %.2f", dglobal)
        else:
            raise ValueError(f"Unknown dglobal option: {dglobal}")

    Y = _spacemap(
        Y,
        n_epochs,
        optim,
        samples,
        rng_state,
        ptr,
        knn_i,
        knn_j,
        dglobal,
    )

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _spacemap(
    Y,
    n_epochs,
    opt,
    samples,
    rng_state,
    ptr,
    knn_i,
    knn_j,
    dof,
):
    nobs, ndim = Y.shape

    degrees = np.zeros((nobs,), dtype=np.int32)
    for i in knn_i:
        degrees[i] = degrees[i] + 1

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)

        _spacemap_nbrs(
            Y,
            knn_i,
            knn_j,
            ptr,
            dof,
            grads,
        )
        _spacemap_non_nbrs(
            Y,
            samples[n],
            degrees,
            rng_state,
            dof,
            grads,
        )

        Y = opt.opt(Y, grads, n, n_epochs)
        for d in range(ndim):
            Yd = Y[:, d]
            Y[:, d] = Yd - np.mean(Yd)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _spacemap_nbrs(
    Y,
    knn_i,
    knn_j,
    ptr,
    dof,
    grads,
):
    ndim = Y.shape[1]
    twoddof = 2.0 / dof
    # pylint:disable=not-an-iterable
    for p in numba.prange(len(ptr) - 1):
        for edge in range(ptr[p], ptr[p + 1]):
            i = knn_i[edge]
            Yi = Y[i]

            j = knn_j[edge]
            dij2 = rdist(Yi, Y[j])

            wij = math.exp(-pow(dij2, twoddof))

            grad_coeff = -4.0 * math.log(wij) / (0.001 + dof * dij2)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(grad_coeff * (Yi[d] - Y[j, d]))


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _spacemap_non_nbrs(
    Y,
    n_samples,
    degrees,
    rng_state,
    dof,
    grads,
):
    nobs, ndim = Y.shape
    twoddof = 2.0 / dof
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for _ in range(degrees[i] * n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            dij2 = rdist(Yi, Y[j])
            wij = math.exp(-pow(dij2, twoddof))

            grad_coeff = (4.0 * math.log(wij) * wij) / (
                0.001 + dof * dij2 * (1.0 - wij)
            )
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(grad_coeff * (Yi[d] - Y[j, d]))


@dataclass
class Spacemap(drnb.embed.Embedder):
    precomputed_init: np.ndarray = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init

        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = 70

        precomputed_knn = nbrs.get_neighbors_with_ctx(
            x, params.get("metric", "euclidean"), n_neighbors + 1, ctx=ctx
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        params["knn_dist"] = precomputed_knn.dist[:, 1:]
        return embed_spacemap(x, params)


def embed_spacemap(
    x,
    params,
):
    log.info("Running SpaceMAP")
    params["X"] = x
    embedded = spacemap(**params)
    log.info("Embedding completed")

    return embedded
