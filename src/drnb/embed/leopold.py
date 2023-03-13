import math
from dataclasses import dataclass

import numba
import numpy as np
import sklearn.preprocessing
from umap.utils import tau_rand_int

import drnb.embed
import drnb.neighbors as nbrs
from drnb.distances import distance_function
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.yinit import binary_graph_spectral_init, pca, scale_coords, umap_random_init


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
    optargs=None,
    symmetrize="or",
    init_scale=10.0,
    dens_scale=0.0,
    dof=1.0,
):
    if optargs is None:
        optargs = {"decay_alpha": True}
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        optargs["alpha"] = learning_rate
    optim = create_opt(X, opt, optargs)

    nobs = X.shape[0]
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

    ydfun = distance_function("squared_euclidean")

    mean_d = np.mean(knn_dist, axis=1)
    min_scale_d = math.sqrt(1.0e-2)
    max_scale_d = math.sqrt(100.0)
    mean_d[mean_d < min_scale_d] = min_scale_d
    beta_unscaled = 1.0 / mean_d
    beta_scaled = sklearn.preprocessing.minmax_scale(
        beta_unscaled, feature_range=(min_scale_d, max_scale_d)
    )
    beta = (1.0 - dens_scale) + dens_scale * beta_scaled

    dmat = nn_to_sparse(knn_idx, symmetrize=symmetrize)
    dmat.eliminate_zeros()

    knn_i = dmat.row
    knn_j = dmat.col

    samples = create_sample_plan(n_samples, n_epochs, strategy=sample_strategy)

    Y = _leopold(Y, n_epochs, ydfun, optim, samples, rng_state, knn_i, knn_j, beta, dof)

    return Y


@numba.njit()
def clip(val):
    if val > 4.0:
        return 4.0
    if val < -4.0:
        return -4.0
    return val


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _leopold(Y, n_epochs, ydfun, opt, samples, rng_state, knn_i, knn_j, prec, dof):
    nobs, ndim = Y.shape

    degrees = np.zeros((nobs,), dtype=np.int32)
    for i in knn_i:
        degrees[i] = degrees[i] + 1

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)

        _leopold_nbrs(Y, ydfun, knn_i, knn_j, prec, dof, grads)
        _leopold_non_nbrs(Y, ydfun, samples[n], degrees, rng_state, prec, dof, grads)

        Y = opt.opt(Y, grads, n, n_epochs)
        for d in range(ndim):
            Yd = Y[:, d]
            Y[:, d] = Yd - np.mean(Yd)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _leopold_nbrs(Y, ydfun, knn_i, knn_j, prec, dof, grads):
    nedges = len(knn_i)
    ndim = Y.shape[1]
    for edge in range(nedges):
        i = knn_i[edge]
        Yi = Y[i]

        j = knn_j[edge]
        dij2 = ydfun(Yi, Y[j])

        beta = prec[i] * prec[j]

        grad_coeff = (2.0 * beta) / (1.0 + (beta / dof) * dij2)
        for d in range(ndim):
            grads[i, d] = grads[i, d] + clip(grad_coeff * (Yi[d] - Y[j, d]))


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _leopold_non_nbrs(Y, ydfun, n_samples, degrees, rng_state, prec, dof, grads):
    nobs, ndim = Y.shape
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Yi = Y[i]
        for _ in range(degrees[i] * n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            dij2 = ydfun(Yi, Y[j])
            beta = prec[i] * prec[j]
            wwij = 1.0 + (beta / dof) * dij2
            wij = pow(wwij, -dof)
            grad_coeff = (-2.0 * beta * wij) / (wwij * (1.001 - wij))
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(grad_coeff * (Yi[d] - Y[j, d]))
    # return grads


@dataclass
class Leopold(drnb.embed.Embedder):
    precomputed_init: np.ndarray = None

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
