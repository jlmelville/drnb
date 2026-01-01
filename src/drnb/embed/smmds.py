from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np
from numpy.typing import NDArray
from umap.utils import tau_rand_int

import drnb.embed
import drnb.embed.base
from drnb.distances import distance_function
from drnb.embed.context import EmbedContext
from drnb.embed.umap.utils import euclidean
from drnb.log import log
from drnb.optim import create_opt
from drnb.preprocess import pca as pca_reduce
from drnb.rng import setup_rngn
from drnb.sampling import create_sample_plan
from drnb.types import DistanceFunc, EmbedResult
from drnb.yinit import pca as pca_init
from drnb.yinit import scale_coords, umap_random_init


def mmds_init(
    init: np.ndarray | Literal["pca", "pcaw", "rand"],
    nobs: int,
    X: np.ndarray | None = None,
    init_scale: float | None = None,
    random_state: int = 42,
) -> NDArray[np.float32]:
    """Initialize the embedding for Metric Multidimensional Scaling (MMDS) methods."""
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


def smmds(
    X: np.ndarray,
    metric: str = "euclidean",
    n_epochs: int = 200,
    init: np.ndarray | Literal["pca", "pcaw", "rand"] = "pca",
    n_samples: int = 3,
    sample_strategy: Literal["unif", "inc", "dec"] | None = None,
    random_state: int = 42,
    learning_rate: float | None = None,
    opt: str = "adam",
    optargs: dict | None = None,
    eps: float = 1e-10,
    pca: int | None = None,
    init_scale: float | None = None,
):
    """Run the Stochastic Metric Multidimensional Scaling (SMMDS) embedding algorithm."""
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


def _smmds(
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int,
    eps: float,
    xdfun: DistanceFunc,
    opt: drnb.optim.OptimizerProtocol,
    samples: np.ndarray,
    rng_state: np.ndarray,
) -> np.ndarray:
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
    X: np.ndarray,
    Y: np.ndarray,
    eps: float,
    xdfun: DistanceFunc,
    rng_state: np.ndarray,
    n_samples: int,
    nobs: int,
    ndim: int,
) -> np.ndarray:
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


@dataclass
class Smmds(drnb.embed.base.Embedder):
    """Stochastic Metric MDS (SMMDS) embedding implementation.

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

        log.info("Running SMMDS")
        params["X"] = x
        embedded = smmds(**params)
        log.info("Embedding completed")

        return embedded


def snmds(
    X: np.ndarray,
    metric: str = "euclidean",
    n_epochs: int = 200,
    init: np.ndarray | Literal["pca", "pcaw", "rand"] = "pca",
    n_samples: int = 3,
    random_state: int = 42,
    learning_rate: float | None = None,
    opt: str = "adam",
    optargs: dict | None = None,
    eps: float = 1e-10,
    pca: int | None = None,
    init_scale: float | None = None,
) -> np.ndarray:
    """Run the Stochastic Normalized Multidimensional Scaling (SNMDS) embedding
    algorithm."""
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        if optargs is None:
            optargs = {}
        optargs["alpha"] = learning_rate
    nobs = X.shape[0]
    optim = create_opt(nobs, opt, optargs)
    rng_state = setup_rngn(nobs, random_state)

    xdfun = distance_function(metric)

    if pca is not None:
        X = pca_reduce(X, n_components=pca)
    Y = mmds_init(init, nobs, X, init_scale, random_state)
    Y = _snmds(X, Y, n_epochs, eps, xdfun, optim, n_samples, rng_state)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _snmds(
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int,
    eps: float,
    xdfun: DistanceFunc,
    opt: drnb.optim.OptimizerProtocol,
    n_samples: int,
    rng_state: np.ndarray,
) -> np.ndarray:
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
def _snmds_epoch(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float,
    xdfun: DistanceFunc,
    n_samples: int,
    nobs: int,
    ndim: int,
    rng_state: np.ndarray,
    js: np.ndarray,
    pq: np.ndarray,
    rs: np.ndarray,
    ds: np.ndarray,
) -> np.ndarray:
    grads = np.zeros((nobs, ndim), dtype=np.float32)

    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Xi = X[i]
        rsum = 0.0

        Yi = Y[i]
        dsum = eps

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
class Snmds(drnb.embed.base.Embedder):
    """Stochastic Normalized MDS (SNMDS) embedding implementation.

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

        log.info("Running SNMDS")
        params["X"] = x
        embedded = snmds(**params)
        log.info("Embedding completed")

        return embedded


def mmds(
    X: np.ndarray,
    metric: str = "euclidean",
    n_epochs: int = 200,
    init: np.ndarray | Literal["pca", "pcaw", "rand"] = "pca",
    random_state: int = 42,
    learning_rate: float | None = None,
    opt: str = "adam",
    optargs: dict | None = None,
    eps: float = 1e-10,
    init_scale: float | None = None,
):
    """Run the Metric Multidimensional Scaling (MMDS) embedding algorithm."""
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
def _mmds(
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int,
    eps: float,
    xdfun: DistanceFunc,
    ydfun: DistanceFunc,
    opt: drnb.optim.OptimizerProtocol,
) -> np.ndarray:
    nobs, ndim = Y.shape

    for n in range(n_epochs):
        grads = _mmds_epoch(X, Y, eps, xdfun, ydfun, nobs, ndim)

        Y = opt.opt(Y, grads, n, n_epochs)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _mmds_epoch(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float,
    xdfun: DistanceFunc,
    ydfun: DistanceFunc,
    nobs: int,
    ndim: int,
) -> np.ndarray:
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
class Mmds(drnb.embed.base.Embedder):
    """Metric Multidimensional Scaling (MMDS) embedding using numba."""

    precomputed_init: np.ndarray | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, _: EmbedContext | None = None
    ) -> EmbedResult:
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init
        log.info("Running MMDS")
        params["X"] = x
        embedded = mmds(**params)
        log.info("Embedding completed")

        return embedded
