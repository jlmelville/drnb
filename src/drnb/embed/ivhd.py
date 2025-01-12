from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
import drnb.embed.base
import drnb.optim
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.embed.umap.utils import euclidean
from drnb.log import is_progress_report_iter, log
from drnb.neighbors.hubness import nn_to_sparse
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.types import EmbedResult
from drnb.yinit import standard_neighbor_init


@numba.njit()
def clip(val: float, max_val: float = 4.0, min_val: float = -4.0) -> float:
    """Clip a value to a range between min_val and max_val."""
    if val > max_val:
        return max_val
    if val < min_val:
        return min_val
    return val


def ivhd(
    knn_idx: np.ndarray,
    n_random: int = 1,
    n_epochs: int = 200,
    near_dist: float = 0.0,
    far_dist: float = 1.0,
    far_weight: float = 0.01,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "pca",
    learning_rate: float | None = None,
    opt: str = "adam",
    optargs: dict | None = None,
    sample_strategy: str = "unif",
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    eps: float = 1e-5,
    random_state: int = 42,
    X: np.ndarray | None = None,
    init_scale: float = None,
    clip_grad: float = 4.0,
) -> np.ndarray:
    """Embed data using IVHD."""
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
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    knn_j: np.ndarray,
    ptr: np.ndarray,
    near_dist: float,
    far_dist: float,
    far_weight: float,
    eps: float,
    clip_grad: float,
) -> np.ndarray:
    nobs, ndim = Y.shape

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        # neighbors
        _ivhd_nbrs(Y, knn_j, ptr, near_dist, eps, grads, clip_grad)

        # non-neighbors
        _ivhd_inner_nnbr(
            Y,
            samples[n],
            ptr,
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
def _ivhd_nbrs(
    Y: np.ndarray,
    knn_j: np.ndarray,
    ptr: np.ndarray,
    near_dist: float,
    eps: float,
    grads: np.ndarray,
    clip_grad: float,
):
    ndim = Y.shape[1]
    # pylint:disable=not-an-iterable
    for i in numba.prange(len(ptr) - 1):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
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
    Y: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    far_dist: float,
    far_weight: float,
    eps: float,
    grads: np.ndarray,
    clip_grad: float,
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
            dij = euclidean(Yi, Y[j])

            grad_coeff = far_weight * (dij - far_dist) / (dij + eps)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(
                    grad_coeff * (Yi[d] - Y[j, d]),
                    max_val=clip_grad,
                    min_val=-clip_grad,
                )


@dataclass
class Ivhd(drnb.embed.base.Embedder):
    """
    The IVHD Embedder

    Attributes:
        precomputed_init: Optional array of initial coordinates for embedding. If None,
            random initialization is used.
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
            n_neighbors = 2

        precomputed_knn = get_neighbors_with_ctx(
            x,
            params.get("metric", "euclidean"),
            n_neighbors + 1,
            return_distance=False,
            ctx=ctx,
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]

        log.info("Running IVHD")
        params["X"] = x
        embedded = ivhd(**params)
        log.info("Embedding completed")

        return embedded


def xvhd(
    knn_idx: np.ndarray,
    n_random: int = 1,
    n_epochs: int = 200,
    near_dist: float = 0.0,
    far_weight: float = 0.01,
    init: np.ndarray | Literal["pca", "rand", "spectral", "gspectral"] = "pca",
    learning_rate: float | None = None,
    opt: str = "adam",
    optargs: dict | None = None,
    sample_strategy: str = "unif",
    symmetrize: Literal["or", "and", "mean"] | None = "or",
    eps: float = 1e-5,
    random_state: int = 42,
    X: np.ndarray | None = None,
    init_scale: float = None,
    clip_grad: float = 4.0,
) -> np.ndarray:
    """Embed data using XVHD."""
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

    knn_j = dmat.col
    ptr = dmat.tocsr().indptr

    samples = create_sample_plan(
        n_samples=n_random, n_epochs=n_epochs, strategy=sample_strategy
    )

    Y = _xvhd(
        Y,
        n_epochs,
        optim,
        samples,
        rng_state,
        knn_j,
        ptr,
        near_dist,
        far_weight,
        eps,
        clip_grad,
    )

    return Y


def _xvhd(
    Y: np.ndarray,
    n_epochs: int,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
    knn_j: np.ndarray,
    ptr: np.ndarray,
    near_dist: float,
    far_weight: float,
    eps: float,
    clip_grad: float,
) -> np.ndarray:
    nobs, ndim = Y.shape

    for n in range(n_epochs):
        radii = np.zeros((nobs,), dtype=np.float32)
        grads = np.zeros((nobs, ndim), dtype=np.float32)

        # neighbors
        _xvhd_nbrs(Y, knn_j, ptr, near_dist, eps, clip_grad, grads, radii)

        # non-neighbors
        _xvhd_non_nbrs(
            Y,
            radii,
            samples[n],
            ptr,
            rng_state,
            far_weight,
            eps,
            clip_grad,
            grads,
        )
        if is_progress_report_iter(n, n_epochs, 10):
            log.info("epoch %d %f", n + 1, radii[0])
        Y = opt.opt(Y, grads, n, n_epochs)

    for d in range(ndim):
        Yd = Y[:, d]
        Y[:, d] = Yd - np.mean(Yd)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _xvhd_nbrs(
    Y: np.ndarray,
    knn_j: np.ndarray,
    ptr: np.ndarray,
    near_dist: float,
    eps: float,
    clip_grad: float,
    grads: np.ndarray,
    radii: np.ndarray,
):
    ndim = Y.shape[1]
    # pylint:disable=not-an-iterable
    for i in numba.prange(len(ptr) - 1):
        Yi = Y[i]
        for edge in range(ptr[i], ptr[i + 1]):
            j = knn_j[edge]
            dij = euclidean(Yi, Y[j])
            if dij > radii[i]:
                radii[i] = dij
            grad_coeff = (dij - near_dist) / (dij + eps)
            for d in range(ndim):
                grads[i, d] = grads[i, d] + clip(
                    grad_coeff * (Yi[d] - Y[j, d]),
                    max_val=clip_grad,
                    min_val=-clip_grad,
                )


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _xvhd_non_nbrs(
    Y: np.ndarray,
    radii: np.ndarray,
    n_samples: int,
    ptr: np.ndarray,
    rng_state: np.ndarray,
    far_weight: float,
    eps: float,
    clip_grad: float,
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
            dij = euclidean(Yi, Y[j])
            rad = radii[i]
            if radii[j] > rad:
                rad = radii[j]
            if dij < rad:
                grad_coeff = far_weight * (dij - 2.0 * rad) / (dij + eps)
                for d in range(ndim):
                    grads[i, d] = grads[i, d] + clip(
                        grad_coeff * (Yi[d] - Y[j, d]),
                        max_val=clip_grad,
                        min_val=-clip_grad,
                    )


@dataclass
class Xvhd(drnb.embed.base.Embedder):
    """
    The XVHD Embedder

    Attributes:
        precomputed_init: Optional array of initial coordinates for embedding. If None,
            random initialization is used.
    """

    precomputed_init: np.ndarray | None = None

    def embed_impl(self, x, params, ctx=None):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init

        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = 2

        precomputed_knn = get_neighbors_with_ctx(
            x,
            params.get("metric", "euclidean"),
            n_neighbors + 1,
            return_distance=False,
            ctx=ctx,
        )
        params["knn_idx"] = precomputed_knn.idx[:, 1:]

        log.info("Running XVHD")
        params["X"] = x
        embedded = xvhd(**params)
        log.info("Embedding completed")

        return embedded
