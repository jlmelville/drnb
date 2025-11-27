import math
from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
import drnb.embed.base
from drnb.dimension import mle_global
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.embed.umap.utils import clip, rdist
from drnb.log import log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.types import EmbedResult
from drnb.yinit import binary_graph_spectral_init, pca, scale_coords, umap_random_init


def spacemap(
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
    dglobal: str = "auto",
) -> np.ndarray:
    """Run the SpaceMAP embedding algorithm."""
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
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    ptr: np.ndarray,
    knn_i: np.ndarray,
    knn_j: np.ndarray,
    dof: float,
) -> np.ndarray:
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
    Y: np.ndarray,
    knn_i: np.ndarray,
    knn_j: np.ndarray,
    ptr: np.ndarray,
    dof: np.ndarray,
    grads: np.ndarray,
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
    Y: np.ndarray,
    n_samples: int,
    degrees: np.ndarray,
    rng_state: np.ndarray,
    dof: float,
    grads: np.ndarray,
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
class Spacemap(drnb.embed.base.Embedder):
    """Spacemap embedding implementation.

    Attributes:
        precomputed_init (numpy.ndarray | None): Optional precomputed initial
        coordinates
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
            n_neighbors = 70

        precomputed_knn = get_neighbors_with_ctx(
            x, params.get("metric", "euclidean"), n_neighbors + 1, ctx=ctx
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        params["knn_dist"] = precomputed_knn.dist[:, 1:]

        log.info("Running SpaceMAP")
        params["X"] = x
        embedded = spacemap(**params)
        log.info("Embedding completed")

        return embedded
