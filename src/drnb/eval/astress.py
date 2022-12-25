from dataclasses import dataclass

import numba
import numpy as np
from umap.utils import tau_rand_int

from drnb.distances import distance_function
from drnb.eval import EvalResult
from drnb.rng import setup_rngn

from .base import EmbeddingEval


def approx_stress(X, coords, metric="euclidean", n_samples=3, random_state=42):
    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")
    rng_state = setup_rngn(X.shape[0], random_state)

    return _approx_stress(X, coords, xdfun, ydfun, n_samples, rng_state)


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _approx_stress(X, Y, xdfun, ydfun, n_samples, rng_state):
    nobs, _ = X.shape
    stressv = np.zeros(nobs)
    # pylint: disable=not-an-iterable
    for i in numba.prange(nobs):
        for _ in range(n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            rij = xdfun(X[i], X[j])
            dij = ydfun(Y[i], Y[j])
            stressv[i] += (rij - dij) ** 2
    return np.sqrt(np.mean(stressv))


@dataclass
class ApproxStressEval(EmbeddingEval):
    random_state: int = None
    metric: str = "euclidean"
    n_samples: int = 3

    def evaluate(self, X, coords, ctx=None):
        result = approx_stress(
            X, coords, self.metric, self.n_samples, self.random_state
        )

        return EvalResult(
            eval_type="ApproxStress",
            label=str(self),
            info=dict(metric=self.metric, n_samples=self.n_samples),
            value=result,
        )

    def __str__(self):
        return f"astress-{self.metric}-{self.n_samples}"
