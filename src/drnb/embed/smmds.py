from dataclasses import dataclass

import numba
import numpy as np
from umap.utils import tau_rand_int

import drnb.embed
from drnb.distances import distance_function
from drnb.log import log
from drnb.optim import create_opt
from drnb.rng import setup_rngn
from drnb.yinit import pca, umap_random_init


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
):
    # specifying learning_rate takes precedence over any value of alpha set in opt args
    if learning_rate is not None:
        if optargs is None:
            optargs = {}
        optargs["alpha"] = learning_rate
    optim = create_opt(X, opt, optargs)

    rng_state = setup_rngn(X.shape[0], random_state)

    if isinstance(init, np.ndarray):
        if init.shape != (X.shape[0], 2):
            raise ValueError("Initialization array has incorrect shape")
        log.info("Using pre-supplied initialization coordinates")
        Y = init
    elif init == "pca":
        Y = pca(X)
    elif init == "rand":
        Y = umap_random_init(random_state)
    else:
        raise ValueError(f"Unknown init option '{init}'")
    Y = Y.astype(np.float32, order="C")

    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")

    Y = _smmds(X, Y, n_epochs, eps, xdfun, ydfun, optim, n_samples, rng_state)

    return Y


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _smmds(X, Y, n_epochs, eps, xdfun, ydfun, opt, n_samples, rng_state):
    nobs, ndim = Y.shape

    for n in range(n_epochs):
        grads = np.zeros((nobs, ndim), dtype=np.float32)
        # pylint:disable=not-an-iterable
        for i in numba.prange(nobs):
            for _ in range(n_samples):
                j = tau_rand_int(rng_state[i]) % nobs
                if i == j:
                    continue

                rij = xdfun(X[i], X[j])
                dij = ydfun(Y[i], Y[j])
                grad_coeff = (dij - rij) / (dij + eps)
                ydiff = Y[i] - Y[j]
                grads[i] += grad_coeff * ydiff
                grads[j] -= grad_coeff * ydiff
        Y = opt.opt(Y, grads, n, n_epochs)
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
