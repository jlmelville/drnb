import math
from dataclasses import dataclass
from typing import Optional, cast

import numba
import numpy as np
import sklearn.preprocessing
from umap.utils import tau_rand_int

import drnb.embed
import drnb.neighbors as nbrs
from drnb.embed.umap.utils import clip, rdist
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.yinit import standard_neighbor_init


def leopold(
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
    init_scale=10.0,
    dens_scale=0.0,
    dof=1.0,
    anneal_dens=0,
    anneal_dof=0,
):
    if anneal_dens > n_epochs:
        raise ValueError("anneal_dens must be <= n_epochs")
    if anneal_dof > n_epochs:
        raise ValueError("anneal_dof must be <= n_epochs")

    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate

    nobs = X.shape[0]
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

    # local densities
    mean_d = np.mean(knn_dist, axis=1)
    min_scale_d = math.sqrt(1.0e-2)
    max_scale_d = math.sqrt(100.0)
    mean_d[mean_d < min_scale_d] = min_scale_d
    beta_unscaled = 1.0 / mean_d
    beta_scaled = sklearn.preprocessing.minmax_scale(
        beta_unscaled, feature_range=(min_scale_d, max_scale_d)  # type: ignore
    )

    dmat = nn_to_sparse(knn_idx, symmetrize=symmetrize)
    dmat.eliminate_zeros()

    knn_j = dmat.col
    ptr = dmat.tocsr().indptr

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    Y = _leopold(
        Y,
        n_epochs,
        optim,
        samples,
        rng_state,
        knn_j,
        ptr,
        beta_scaled,
        dens_scale,
        dof,
        anneal_dens,
        anneal_dof,
    )

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _leopold(
    Y,
    n_epochs,
    opt,
    samples,
    rng_state,
    knn_j,
    ptr,
    prec,
    dens_scale,
    dof,
    anneal_dens,
    anneal_dof,
):
    nobs, ndim = Y.shape

    # Anneal over anneal_dens epochs and then pad out with the dens_scale to n_epochs
    epoch_dens_scale = np.linspace(0.0, dens_scale, anneal_dens)
    epoch_dens_scale = np.append(
        epoch_dens_scale, np.repeat(dens_scale, n_epochs - len(epoch_dens_scale))
    )

    # dof > 2.0 is not visually that different
    max_dof = max(2.0, dof)
    # space out the annealing of dof non-linearly
    epoch_dof = np.flip(np.exp(np.linspace(np.log(dof), np.log(max_dof), anneal_dof)))
    epoch_dof = np.append(epoch_dof, np.repeat(dof, n_epochs - len(epoch_dof)))

    for n in range(n_epochs):
        beta = (1.0 - epoch_dens_scale[n]) + epoch_dens_scale[n] * prec
        edof = epoch_dof[n]
        grads = np.zeros((nobs, ndim), dtype=np.float32)

        _leopold_nbrs(Y, knn_j, ptr, beta, edof, grads)
        _leopold_non_nbrs(Y, samples[n], ptr, rng_state, beta, edof, grads)

        Y = opt.opt(Y, grads, n, n_epochs)
        for d in range(ndim):
            Yd = Y[:, d]
            Y[:, d] = Yd - np.mean(Yd)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _leopold_nbrs(Y, knn_j, ptr, prec, dof, grads):
    ndim = Y.shape[1]
    # pylint:disable=not-an-iterable
    for i in numba.prange(len(ptr) - 1):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            j = knn_j[edge]
            dij2 = rdist(Yi, Y[j])

            beta = prec[i] * prec[j]

            grad_coeff = (2.0 * beta) / (1.0 + (beta / dof) * dij2)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(grad_coeff * (Yi[d] - Y[j, d]))


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _leopold_non_nbrs(Y, n_samples, ptr, rng_state, prec, dof, grads):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        # number of edges in i is ptr[i + 1] - ptr[i]
        for _ in range((ptr[i + 1] - ptr[i]) * n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            dij2 = rdist(Yi, Y[j])
            beta = prec[i] * prec[j]

            wwij = 1.0 + (beta / dof) * dij2
            wij = pow(wwij, -dof)
            grad_coeff = (-2.0 * beta * wij) / (wwij * (1.001 - wij))
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(grad_coeff * (Yi[d] - Y[j, d]))


@dataclass
class Leopold(drnb.embed.Embedder):
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
        return embed_leopold(x, params)


def embed_leopold(
    x,
    params,
):
    log.info("Running LEOPOLD")
    params["X"] = x
    embedded = leopold(**params)
    log.info("Embedding completed")

    return embedded
