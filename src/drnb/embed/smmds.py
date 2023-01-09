from dataclasses import dataclass

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
from drnb.distances import distance_function
from drnb.log import log
from drnb.optim import create_opt
from drnb.rng import setup_rng, setup_rngn
from drnb.yinit import pca, scale_coords, umap_random_init


def smmds(
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
):
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        if optargs is None:
            optargs = {}
        optargs["alpha"] = learning_rate
    optim = create_opt(X, opt, optargs)

    nobs = X.shape[0]
    rng_state = setup_rng(random_state)
    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")

    Y = mmds_init(init, nobs, X, init_scale, random_state)

    Y = _smmds(X, Y, n_epochs, eps, xdfun, ydfun, optim, n_samples, rng_state)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _smmds(X, Y, n_epochs, eps, xdfun, ydfun, opt, n_samples, rng_state):
    nobs, ndim = Y.shape
    epoch_samples = np.zeros(n_samples, dtype=np.int32)

    for n in range(n_epochs):
        # to save on time in the PRNG use the same set of samples within each epoch
        for j in range(n_samples):
            epoch_samples[j] = tau_rand_int(rng_state) % nobs

        grads = _smmds_epoch(X, Y, eps, xdfun, ydfun, epoch_samples, nobs, ndim)

        Y = opt.opt(Y, grads, n, n_epochs)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _smmds_epoch(X, Y, eps, xdfun, ydfun, epoch_samples, nobs, ndim):
    grads = np.zeros((nobs, ndim), dtype=np.float32)
    # pylint:disable=not-an-iterable
    for i in numba.prange(nobs):
        Xi = X[i]
        Yi = Y[i]
        for j in epoch_samples:
            if i == j:
                continue

            Yj = Y[j]
            rij = xdfun(Xi, X[j])
            dij = ydfun(Yi, Yj)
            # by not updating grads[j] this is safe to parallelize
            grads[i] = grads[i] + ((dij - rij) / (dij + eps)) * (Yi - Yj)
    return grads


def mmds_init(init, nobs, X=None, init_scale=None, random_state=42):
    if isinstance(init, np.ndarray):
        if init.shape != (nobs, 2):
            raise ValueError("Initialization array has incorrect shape")
        log.info("Using pre-supplied initialization coordinates")
        Y = init
    elif init == "pca":
        Y = pca(X)
    elif init == "pcaw":
        Y = pca(X, whiten=True)
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
    precomputed_init: np.ndarray = None

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
):
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        if optargs is None:
            optargs = {}
        optargs["alpha"] = learning_rate
    optim = create_opt(X, opt, optargs)

    nobs = X.shape[0]
    rng_state = setup_rngn(nobs, random_state)

    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")

    Y = mmds_init(init, nobs, X, init_scale, random_state)

    Y = _snmds(X, Y, n_epochs, eps, xdfun, ydfun, optim, n_samples, rng_state)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _snmds(X, Y, n_epochs, eps, xdfun, ydfun, opt, n_samples, rng_state):
    nobs, ndim = Y.shape
    js = np.empty(n_samples, dtype=np.int32)
    pq = np.empty(n_samples, dtype=np.float32)
    rs = np.empty(n_samples, dtype=np.float32)
    ds = np.empty(n_samples, dtype=np.float32)
    for n in range(n_epochs):
        grads = _snmds_epoch(
            X, Y, eps, xdfun, ydfun, n_samples, nobs, ndim, rng_state, js, pq, rs, ds
        )

        Y = opt.opt(Y, grads, n, n_epochs)
    return Y


@numba.jit(nopython=True, fastmath=True, parallel=False)
def _snmds_epoch(
    X, Y, eps, xdfun, ydfun, n_samples, nobs, ndim, rng_state, js, pq, rs, ds
):
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
                dij = ydfun(Yi, Y[j])

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
            if i == j:
                continue
            gy = ((pqq - pq[k]) / ds[k]) * (Yi - Y[j])
            grads[i] = grads[i] + gy
            grads[j] = grads[j] - gy
    return grads


@dataclass
class Snmds(drnb.embed.Embedder):
    precomputed_init: np.ndarray = None

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
