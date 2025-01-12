import math
from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np
import sklearn.preprocessing
from umap.utils import tau_rand_int

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.embed.umap.utils import clip, rdist
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.types import EmbedResult
from drnb.yinit import standard_neighbor_init


def leopold(
    X: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    n_epochs: int = 500,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "spectral",
    n_samples: int = 5,
    sample_strategy: str | None = None,
    random_state: int = 42,
    learning_rate: float | None = 1.0,
    opt: str = "adam",
    optargs: dict | None = None,
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    init_scale: float = 10.0,
    dens_scale: float = 0.0,
    dof: float = 1.0,
    anneal_dens: int = 0,
    anneal_dof: int = 0,
):
    """Run the LEOPOLD (Lightweight Estimate Of Preservation Of Local Density)
    embedding algorithm.

    Parameters
    ----------
    X : np.ndarray
        The input data to embed.
    knn_idx : np.ndarray
        The nearest neighbors indices.
    knn_dist : np.ndarray
        The nearest neighbors distances.
    n_epochs : int, optional
        The number of epochs to run, by default 500.
    init : np.ndarray or str, optional
        The initial embedding, by default "spectral".
    n_samples : int, optional
        The number of samples to take per non-neighbor, by default 5.
    sample_strategy : str, optional
        The sampling strategy to use, by default None.
    random_state : int, optional
        The random state, by default 42.
    learning_rate : float, optional
        The learning rate, by default 1.0.
    opt : str, optional
        The optimizer to use, by default "adam".
    optargs : dict, optional
        The optimizer arguments, by default None.
    symmetrize : str, optional
        The symmetrization strategy to use, by default "or".
    init_scale : float, optional
        The initial scale, by default 10.0.
    dens_scale : float, optional
        The density scale, by default 0.0.
    dof : float, optional
        The degrees of freedom, by default 1.0.
    anneal_dens : int, optional
        The number of epochs to anneal the density, by default 0.
    anneal_dof : int, optional
        The number of epochs to anneal the degrees of freedom, by default 0.
    """
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
        beta_unscaled,
        feature_range=(min_scale_d, max_scale_d),  # type: ignore
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
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    knn_j: np.ndarray,
    ptr: np.ndarray,
    prec: float,
    dens_scale: float,
    dof: float,
    anneal_dens: int,
    anneal_dof: int,
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
def _leopold_nbrs(
    Y: np.ndarray,
    knn_j: np.ndarray,
    ptr: np.ndarray,
    prec: np.ndarray,
    dof,
    grads: np.ndarray,
):
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
def _leopold_non_nbrs(
    Y: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    prec: np.ndarray,
    dof: float,
    grads: np.ndarray,
):
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
class Leopold(drnb.embed.base.Embedder):
    """
    The Leopold embedder uses the LEOPOLD technique for dimensionality reduction of
    high-dimensional data. If precomputed_init is provided, those coordinates are used
    as the initial embedding.

    Attributes:
        precomputed_init: Optional array of initial coordinates for embedding.
    """

    precomputed_init: np.ndarray | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init

        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = 15

        precomputed_knn = get_neighbors_with_ctx(
            x, params.get("metric", "euclidean"), n_neighbors + 1, ctx=ctx
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        params["knn_dist"] = precomputed_knn.dist[:, 1:]

        log.info("Running LEOPOLD")
        params["X"] = x
        embedded = leopold(**params)
        log.info("Embedding completed")

        return embedded
