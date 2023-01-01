from dataclasses import dataclass

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
import drnb.neighbors as nbrs
from drnb.distances import distance_function
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import ncvis_negative_plan
from drnb.yinit import binary_graph_spectral_init, pca, scale_coords, umap_random_init


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
    optargs=None,
    sample_strategy=None,
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
    optim = create_opt(X, opt, optargs)

    nobs = knn_idx.shape[0]
    rng_state = setup_rngn(nobs, random_state)

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
        Y = binary_graph_spectral_init(knn=knn_idx)
    else:
        raise ValueError(f"Unknown init option '{init}'")
    if init_scale is not None:
        Y = scale_coords(Y, max_coord=init_scale)

    Y = Y.astype(np.float32, order="C")

    ydfun = distance_function("euclidean")

    dmat = nn_to_sparse(knn_idx, symmetrize=symmetrize)
    dmat.eliminate_zeros()
    knn_i = dmat.row
    knn_j = dmat.col

    if sample_strategy is not None:
        if sample_strategy == "inc":
            samples = ncvis_negative_plan(n_random, n_epochs)
        elif sample_strategy == "dec":
            samples = np.flip(ncvis_negative_plan(n_random, n_epochs))
        else:
            raise ValueError(f"Unknown sample strategy {sample_strategy}")
    else:
        samples = np.array([n_random] * n_epochs, dtype=np.int)
    Y = _ivhd(
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
        eps,
    )

    return Y


@numba.jit(nopython=True, fastmath=True)
def _ivhd_inner_nbr(Y, ydfun, knn_i, knn_j, near_dist, eps, nedges, grads):
    # neighbors
    for edge in range(nedges):
        i = knn_i[edge]
        j = knn_j[edge]
        dij = ydfun(Y[i], Y[j])

        grad_coeff = (dij - near_dist) / (dij + eps)
        grads[i] = grads[i] + grad_coeff * (Y[i] - Y[j])
    return grads


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _ivhd_inner_nnbr(
    Y,
    ydfun,
    rng_state,
    nsamples,
    far_dist,
    far_weight,
    eps,
    nobs,
    grads,
):
    # non-neighbors
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        for _ in range(nsamples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            dij = ydfun(Y[i], Y[j])

            grad_coeff = far_weight * (dij - far_dist) / (dij + eps)
            grads[i] = grads[i] + grad_coeff * (Y[i] - Y[j])
    return grads


@numba.jit(nopython=True, fastmath=True)
def _ivhd(
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
    eps,
):
    nobs, ndim = Y.shape
    nedges = len(knn_i)

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        nsamples = samples[n]
        # neighbors
        grads = _ivhd_inner_nbr(Y, ydfun, knn_i, knn_j, near_dist, eps, nedges, grads)

        # non-neighbors
        grads = _ivhd_inner_nnbr(
            Y,
            ydfun,
            rng_state,
            nsamples,
            far_dist,
            far_weight,
            eps,
            nobs,
            grads,
        )

        Y = opt.opt(Y, grads, n, n_epochs)
        for d in range(ndim):
            Y[:, d] -= np.mean(Y[:, d])
    return Y


@dataclass
class Ivhd(drnb.embed.Embedder):
    precomputed_init: np.ndarray = None

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
